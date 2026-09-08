import Foundation

/// The Stage 1 replay gate (ADR-0063, ticket #477): the per-turn judgement
/// over a **Canonical-Echo Fidelity** walk's Emitted Path account. Pure —
/// the corpus suite feeds it the walk's boundary reports and asserts the
/// failure list is empty; every failure names the turn, the leaf source and
/// its reason, the prefilled count and the tail, so a red run reads without
/// re-running the walk.
///
/// The rules, per boundary the walk reduced from adjacent recordings:
///
/// 1. **Leaf source.** In a tool stretch (the echoed turn emitted tool
///    calls) the source is `live`. A stop turn is `live` too, or `boundary`
///    with the one explained reason — the think-stripping user boundary
///    (ADR-0063 decision 12), where the next user message re-renders the
///    stretch and the canonical-user leaf is synthesized as before.
/// 2. **Registration.** A live-decided turn registered its path. The only
///    tolerated skip is `promptNotTokenPrefix`: the recording keeps no fed
///    ids, and when the canonical encode of the stored render does not
///    extend request N's prompt token-for-token the harness cannot simulate
///    them (the live fast path registers the fed ids themselves, which
///    always extend the prompt).
/// 3. **Fidelity.** The fidelity gate rejected nothing: the echoed message
///    and the template's rendering of it agree.
/// 4. **Overwrites.** No key was registered twice: a recorded session
///    never regenerates from the same parent.
/// 5. **Prefill.** The next request's indexed prefix is the whole registered
///    path — none of the stored turn is re-prefilled — and what it prefills
///    is its new messages plus at most `glueTokenAllowance` glue tokens.
/// 6. **Tail.** Below `tailBudgetPathTokens` path tokens the simulated
///    post-EOS CPU tail (stored render to bytes plus registration) stays
///    under `tailBudgetSeconds`.
nonisolated enum EmittedPathReplayGate {

    /// The glue a next request prefills beyond its new messages: the
    /// generation prompt plus the newline that closes the stored turn's
    /// marker line. The ticket's bound of three assumed a bare
    /// `<|im_start|>assistant\n` prompt; the Qwen3.8 thinking prompt is
    /// five tokens (`<|im_start|>`, `assistant`, `\n`, `<think>`, `\n`), so
    /// with the marker-line newline the allowance is six. A shallow hit —
    /// any of the stored turn re-prefilled — lands above it.
    static let glueTokenAllowance = 6

    /// Post-EOS tail budget, and the path length it applies below.
    static let tailBudgetSeconds = 0.150
    static let tailBudgetPathTokens = 20_000

    /// One turn as the gate sees it.
    struct TurnAccount: Sendable {
        let requestIndex: Int
        let kind: CanonicalEchoFidelity.Boundary.Kind
        let verdict: CanonicalEchoFidelity.EmittedPathVerdict

        /// The echoed turn emitted tool calls: a tool continuation, or the
        /// interrupt that rewound one.
        var isToolStretch: Bool { kind != .canonicalUser }

        init(
            requestIndex: Int, kind: CanonicalEchoFidelity.Boundary.Kind,
            verdict: CanonicalEchoFidelity.EmittedPathVerdict
        ) {
            self.requestIndex = requestIndex
            self.kind = kind
            self.verdict = verdict
        }

        /// `nil` for a boundary the walk learned no index over.
        init?(_ report: CanonicalEchoFidelity.BoundaryReport) {
            guard let verdict = report.emittedPath else { return nil }
            self.init(
                requestIndex: report.boundary.requestIndex, kind: report.boundary.kind,
                verdict: verdict)
        }
    }

    struct Failure: Sendable, CustomStringConvertible, Equatable {
        enum Rule: String, Sendable {
            case leafSource
            case registration
            case fidelity
            case overwrite
            case prefill
            case glue
            case tail
        }

        let rule: Rule
        let requestIndex: Int
        let detail: String
        /// The turn's whole account, rendered once per failure.
        let account: String

        var description: String {
            "request#\(requestIndex) \(rule.rawValue): \(detail) — \(account)"
        }
    }

    /// Judge a walk's boundaries; the empty list is the pass.
    static func check(_ report: CanonicalEchoFidelity.SessionReport) -> [Failure] {
        check(report.boundaries.compactMap(TurnAccount.init))
    }

    static func check(_ turns: [TurnAccount]) -> [Failure] {
        turns.flatMap(check)
    }

    static func check(_ turn: TurnAccount) -> [Failure] {
        let verdict = turn.verdict
        var failures: [Failure] = []
        func fail(_ rule: Failure.Rule, _ detail: String) {
            failures.append(
                Failure(
                    rule: rule, requestIndex: turn.requestIndex, detail: detail,
                    account: account(of: turn)))
        }

        let live =
            verdict.source == LeafStorePhase.Report.Source.live.rawValue
            || verdict.source == LeafStorePhase.Report.Source.handoff.rawValue
        if !live {
            let explained =
                verdict.boundaryReason
                == EmittedPathRegistration.SkipReason.thinkStrippingUserBoundary.rawValue
            if turn.isToolStretch {
                fail(.leafSource, "tool-stretch turn stored from the boundary")
            } else if !explained {
                fail(.leafSource, "stop turn stored from the boundary without an explained reason")
            }
        }
        if verdict.fidelityRejected {
            fail(.fidelity, "the fidelity gate rejected the turn")
        } else if live, !verdict.registered, verdict.registration != promptNotTokenPrefix {
            fail(.registration, "live turn registered nothing")
        }
        if verdict.overwrote {
            fail(.overwrite, "the key was already registered")
        }
        if verdict.registered, let pathLength = verdict.pathLength {
            let prefix = verdict.nextIndexedPrefix ?? 0
            if prefix != pathLength {
                fail(.prefill, "next request served \(prefix) of \(pathLength) path tokens")
            }
            switch verdict.glueTokens {
            case .none:
                fail(.glue, "glue unmeasured")
            case .some(let glue) where glue > glueTokenAllowance:
                fail(.glue, "\(glue) glue tokens, allowance \(glueTokenAllowance)")
            case .some:
                break
            }
            if pathLength < tailBudgetPathTokens {
                switch verdict.tailSeconds {
                case .none:
                    fail(.tail, "tail unmeasured")
                case .some(let tail) where tail >= tailBudgetSeconds:
                    fail(
                        .tail,
                        "\(milliseconds(tail)) ms, budget \(milliseconds(tailBudgetSeconds)) ms")
                case .some:
                    break
                }
            }
        }
        return failures
    }

    /// The harness's spelling for a prompt the stored render does not
    /// extend token-for-token (`CanonicalEchoFidelity.registerEmittedPath`).
    static let promptNotTokenPrefix = "promptNotTokenPrefix"

    /// One line naming everything the rules looked at.
    static func account(of turn: TurnAccount) -> String {
        let verdict = turn.verdict
        func number(_ value: Int?) -> String { value.map(String.init) ?? "-" }
        return [
            "kind=\(turn.kind.rawValue)",
            "mode=\(verdict.mode)",
            "source=\(verdict.source)",
            "boundary=\(verdict.boundaryReason ?? "-")",
            "registration=\(verdict.registration)",
            "pathLength=\(number(verdict.pathLength))",
            "nextPrefix=\(number(verdict.nextIndexedPrefix))",
            "nextMiss=\(verdict.nextMissReason ?? "-")",
            "prefilled=\(number(verdict.nextPrefilled))",
            "newMessages=\(number(verdict.nextNewMessageTokens))",
            "glue=\(number(verdict.glueTokens))",
            "tailMs=\(verdict.tailSeconds.map(milliseconds) ?? "-")",
            "renderMs=\(verdict.renderSeconds.map(milliseconds) ?? "-")",
            "registerMs=\(verdict.registerSeconds.map(milliseconds) ?? "-")",
        ].joined(separator: " ")
    }

    private static func milliseconds(_ seconds: Double) -> String {
        String(format: "%.1f", seconds * 1000)
    }
}
