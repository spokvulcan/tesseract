//
//  Proto273ReplayDriver.swift
//  tesseract
//
//  PROTOTYPE (wayfinder #273) — THROWAWAY. Streams two synthetic requests
//  through `ServerGenerationLog`'s real ingress API so the owner can react
//  to live chat-grammar streaming, phase transitions, tool calls, and the
//  Stop path without pointing a client at the server. Pacing approximates
//  a ~45 tok/s decode. Demo inbound messages ride the real capture path
//  (`RequestTrace.captureInbound`, #274) into `startRequest`.
//

import Foundation
import MLXLMCommon
import Observation

@MainActor
@Observable
final class Proto273ReplayDriver {
    static let shared = Proto273ReplayDriver()

    private(set) var isPlaying = false

    @ObservationIgnored private var playTask: Task<Void, Never>?

    private init() {}

    func play(into log: ServerGenerationLog) {
        guard !isPlaying else { return }
        isPlaying = true
        playTask = Task { @MainActor [weak self] in
            await Self.streamShowcaseRequest(into: log)
            if !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 900_000_000)
                await Self.streamShortRequest(into: log)
            }
            self?.isPlaying = false
        }
    }

    // MARK: - Request 1: the showcase exchange

    private static func streamShowcaseRequest(into log: ServerGenerationLog) async {
        let handle = log.startRequest(
            completionID: "chatcmpl-demo-\(Int.random(in: 1000...9999))",
            model: "qwen3.6-35b-a3b-paro",
            stream: true,
            sessionAffinity: "ses_demo",
            inbound: RequestTrace.captureInbound([
                .init(role: .system, content: .text(showcaseSystemPrompt)),
                .init(
                    role: .user,
                    content: .text(
                        "How does the SSD tier decide what to keep when the disk fills up?")
                ),
                .init(
                    role: .tool,
                    content: .text(
                        "# ADR-0018 — SSD snapshot tier\n\nStatus: accepted\n\n"
                            + "## Context\nThe RAM tier's pressure-reactive budget evicts KV "
                            + "snapshots under memory pressure…")
                ),
            ])
        )

        let streamTask = Task { @MainActor in
            // Lease + cache lookup: a warm RAM hit on most of the prompt.
            try await sleep(180)
            log.markLeaseAcquired(handle: handle)
            try await sleep(60)
            log.markCacheLookupFinished(
                handle: handle,
                reason: "ramHit",
                cachedTokens: 11_776,
                sharedPrefixLength: 11_802,
                promptTokens: 12_845,
                lookupMs: 2.4,
                restoreMs: 38,
                newTokensToPrefill: 1_069
            )
            log.markPrefillStarted(handle: handle)
            try await sleep(650)
            log.markPrefillFinished(handle: handle, prefillMs: 642)

            // A short thinking preamble, streamed plain.
            for chunk in chunked(showcaseThinking, size: 9) {
                try Task.checkCancellation()
                log.ingest(handle: handle, event: .thinking(chunk))
                try await sleep(22)
            }

            // The markdown body.
            for chunk in chunked(showcaseBody, size: 9) {
                try Task.checkCancellation()
                log.ingest(handle: handle, event: .text(chunk))
                try await sleep(22)
            }

            // A tool call, streamed as building deltas then finalized.
            log.ingest(
                handle: handle,
                event: .toolCallDelta(name: nil, argumentsDelta: "\n{\"name\": \"read\","))
            try await sleep(120)
            log.ingest(
                handle: handle,
                event: .toolCallDelta(
                    name: "read",
                    argumentsDelta: " \"arguments\": {\"path\": \"docs/adr/0018-"))
            try await sleep(120)
            log.ingest(
                handle: handle,
                event: .toolCallDelta(
                    name: "read",
                    argumentsDelta: "ssd-snapshot-tier.md\", \"limit\": 40}}"))
            try await sleep(140)
            log.ingest(
                handle: handle,
                event: .toolCall(
                    ToolCall(
                        function: .init(
                            name: "read",
                            arguments: [
                                "path": "docs/adr/0018-ssd-snapshot-tier.md",
                                "limit": 40,
                            ]
                        ))))

            for chunk in chunked(showcaseCoda, size: 9) {
                try Task.checkCancellation()
                log.ingest(handle: handle, event: .text(chunk))
                try await sleep(22)
            }

            log.ingest(
                handle: handle,
                event: .info(
                    .init(
                        promptTokenCount: 12_845,
                        generationTokenCount: 428,
                        promptTime: 0.68,
                        generateTime: 9.4,
                        stopReason: .stop
                    )))
            log.complete(handle: handle, finishReason: "stop")
        }

        log.registerCancelAction(handle: handle) {
            streamTask.cancel()
        }

        do {
            try await streamTask.value
        } catch {
            log.cancel(handle: handle)
        }
    }

    // MARK: - Request 2: short, with a malformed tool call

    private static func streamShortRequest(into log: ServerGenerationLog) async {
        let handle = log.startRequest(
            completionID: "chatcmpl-demo-\(Int.random(in: 1000...9999))",
            model: "qwen3.6-35b-a3b-paro",
            stream: true,
            sessionAffinity: "ses_demo",
            inbound: RequestTrace.captureInbound([
                .init(role: .system, content: .text(shortSystemPrompt)),
                .init(
                    role: .user,
                    content: .text("Summarize the cache telemetry pipeline in one line.")
                ),
            ])
        )

        let streamTask = Task { @MainActor in
            try await sleep(140)
            log.markLeaseAcquired(handle: handle)
            try await sleep(40)
            log.markCacheLookupFinished(
                handle: handle,
                reason: "miss",
                cachedTokens: 0,
                sharedPrefixLength: 0,
                promptTokens: 486,
                lookupMs: 1.1,
                restoreMs: 0,
                newTokensToPrefill: 486
            )
            log.markPrefillStarted(handle: handle)
            try await sleep(380)
            log.markPrefillFinished(handle: handle, prefillMs: 371)

            for chunk in chunked(shortBody, size: 8) {
                try Task.checkCancellation()
                log.ingest(handle: handle, event: .text(chunk))
                try await sleep(24)
            }

            // A malformed tool call: the error grammar under live traffic.
            log.ingest(
                handle: handle,
                event: .malformedToolCall(
                    "{\"name\": \"emit_summary\", \"arguments\": {\"line\": \"…\", }"))
            try await sleep(200)

            log.ingest(
                handle: handle,
                event: .info(
                    .init(
                        promptTokenCount: 486,
                        generationTokenCount: 74,
                        promptTime: 0.37,
                        generateTime: 1.7,
                        stopReason: .stop
                    )))
            log.complete(handle: handle, finishReason: "stop")
        }

        log.registerCancelAction(handle: handle) {
            streamTask.cancel()
        }

        do {
            try await streamTask.value
        } catch {
            log.cancel(handle: handle)
        }
    }

    // MARK: - Helpers

    private static func sleep(_ milliseconds: UInt64) async throws {
        try await Task.sleep(nanoseconds: milliseconds * 1_000_000)
    }

    private static func chunked(_ text: String, size: Int) -> [String] {
        var chunks: [String] = []
        var current = ""
        for character in text {
            current.append(character)
            if current.count >= size {
                chunks.append(current)
                current = ""
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }

    // MARK: - Demo copy

    private static let showcaseSystemPrompt =
        "You are Tesseract's local assistant. Answer from the repository's own "
        + "documents when tools are available. Keep answers concise and cite the "
        + "files you used."

    private static let showcaseThinking =
        "The user is asking about SSD-tier eviction. ADR-0018 covers the dynamic "
        + "budget: floor at 20 GiB equivalent, degrade when free space drops, and "
        + "the survival gate from ADR-0019 that skips writes which would not outlive "
        + "their own admission. I should outline the budget band first, then the "
        + "eviction score, then point at the ledger for the write accounting."

    private static let showcaseBody = """
        The SSD tier keeps itself inside a **dynamic budget band**, and everything \
        else follows from that.

        ## The budget band

        - The budget tracks free disk space and degrades toward a **floor** when \
        the volume fills; a full disk parks it there ("disk low").
        - Admissions that would breach the budget trigger **evict-at-admission**: \
        the lowest-utility snapshots are deleted until the incoming chain fits.
        - A snapshot's utility blends recency with **FLOP efficiency** — how much \
        prefill compute a byte of KV buys back on a hit:

        ```swift
        let utility = alpha * normalizedRecency
                    + (1 - alpha) * normalizedFlopEfficiency
        ```

        ## What never gets written

        The *survival gate* skips writes that would not survive the eviction \
        their own admission triggers — no churn for bytes with no future.

        Let me pull the exact wording from the ADR:
        """

    private static let showcaseCoda = """

        The ADR confirms it: eviction is **utility-ordered within the budget \
        band**, delete accounting lands in the endurance ledger, and a floor-bound \
        budget is reported as *disk low* rather than silently shrinking. See \
        `docs/adr/0018-ssd-snapshot-tier.md` and the ledger counters on the Cache \
        page for the write/delete history.
        """

    private static let shortSystemPrompt =
        "You are Tesseract's local assistant. Keep answers to one sentence."

    private static let shortBody =
        "Every cache decision emits one telemetry event that fans out to the UI "
        + "store, a durable JSONL sink, and the SSD endurance ledger — one pipeline, "
        + "three consumers."
}
