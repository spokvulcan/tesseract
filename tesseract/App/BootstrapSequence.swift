//
//  BootstrapSequence.swift
//  tesseract
//

import Foundation

/// The launch sequence `DependencyContainer.setup()` executes: an ordered list
/// of named steps plus the cross-step ordering invariants that were previously
/// encoded only as statement order and comments.
///
/// The point of the type is that the *declaration* — step names and invariants —
/// is a value that can be validated without running any step (the steps are
/// closures the validator never calls). `setup()` runs exactly what it declares:
/// a pure validator checks the declared order against the invariants first, and
/// only then does the runner invoke the closures in order.
///
/// No dependency-graph framework, no topological sort — the order is declared,
/// not derived. The validator just proves the declared order honours the
/// declared invariants.
nonisolated struct BootstrapSequence {
    /// A named launch step. `run` is `@MainActor` because every production step
    /// touches main-actor-isolated container state.
    struct Step {
        let name: String
        let run: @MainActor () async -> Void
    }

    /// A declared ordering constraint: `before` must run before `after`, for the
    /// stated reason. `why` is the message a reordering test surfaces.
    struct Invariant: Equatable {
        let before: String
        let after: String
        let why: String
    }

    /// A way the declared order fails to honour the declaration. Pure data so
    /// tests can assert on the exact failure.
    enum ValidationFailure: Equatable, CustomStringConvertible {
        /// A step name appears more than once in the ordered list.
        case duplicateStepName(String)
        /// An invariant names a step (`before` or `after`) that isn't in the list.
        case unknownStep(name: String, in: Invariant)
        /// An invariant's `after` step runs at or before its `before` step.
        case violatedOrder(Invariant)

        var description: String {
            switch self {
            case .duplicateStepName(let name):
                return "duplicate step name '\(name)'"
            case .unknownStep(let name, let invariant):
                return "invariant [\(invariant.before) → \(invariant.after)] names "
                    + "unknown step '\(name)'"
            case .violatedOrder(let invariant):
                return "invariant violated: '\(invariant.before)' must run before "
                    + "'\(invariant.after)' — \(invariant.why)"
            }
        }
    }

    let steps: [Step]
    let invariants: [Invariant]

    /// Check the declared order against the declared invariants. Pure: reads only
    /// step names and never calls `run`. Returns every failure found (empty when
    /// the declaration is sound).
    func validate() -> [ValidationFailure] {
        var failures: [ValidationFailure] = []

        // First occurrence index per name, and duplicate detection.
        var firstIndex: [String: Int] = [:]
        for (offset, step) in steps.enumerated() {
            if firstIndex[step.name] == nil {
                firstIndex[step.name] = offset
            } else {
                failures.append(.duplicateStepName(step.name))
            }
        }

        for invariant in invariants {
            let beforeIndex = firstIndex[invariant.before]
            let afterIndex = firstIndex[invariant.after]
            if beforeIndex == nil {
                failures.append(.unknownStep(name: invariant.before, in: invariant))
            }
            if afterIndex == nil {
                failures.append(.unknownStep(name: invariant.after, in: invariant))
            }
            if let beforeIndex, let afterIndex, beforeIndex >= afterIndex {
                failures.append(.violatedOrder(invariant))
            }
        }

        return failures
    }

    /// Validate, then run the steps in declared order. A validation failure here
    /// is a programmer error — the production declaration is wrong, which tests
    /// are meant to catch — so fail loudly rather than launch in a bad order.
    @MainActor
    func run() async {
        let failures = validate()
        if !failures.isEmpty {
            let detail = failures.map(\.description).joined(separator: "; ")
            Log.general.fault("BootstrapSequence declaration invalid: \(detail)")
            preconditionFailure("BootstrapSequence declaration invalid: \(detail)")
        }
        for step in steps {
            await step.run()
        }
    }
}
