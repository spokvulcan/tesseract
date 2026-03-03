import Foundation

/// Configuration for a benchmark run, parsed from CLI arguments.
struct BenchmarkConfig {
    enum Sweep: String {
        case quick
        case full
    }

    let sweep: Sweep
    let scenarioIDs: [String]?  // nil = all
    let outputDir: URL
    let modelDir: URL?  // nil = auto-detect
    let modelID: String?  // nil = default (see ModelDefinition.defaultAgentModelID)

    var resolvedModelID: String {
        modelID ?? ModelDefinition.defaultAgentModelID
    }
    /// Max tokens per generation round. Prevents runaway generation where the model
    /// fails to emit a stop token and generates thousands of tokens per round.
    let maxTokensPerRound: Int

    /// Parameter configurations to sweep over.
    var parameterConfigs: [AgentGenerateParameters] {
        switch sweep {
        case .quick:
            return [.default]
        case .full:
            return Self.fullSweepConfigs
        }
    }

    /// Scenarios that should be skipped for non-default param configs (too slow).
    static let slowScenarioIDs: Set<String> = ["S4"]

    private static let fullSweepConfigs: [AgentGenerateParameters] = {
        let temperatures: [Float] = [0.3, 0.6, 0.8, 1.0]
        let topPs: [Float] = [0.8, 0.9, 0.95]
        let repPenalties: [Float?] = [nil, 1.05, 1.1]

        var configs: [AgentGenerateParameters] = []
        for temp in temperatures {
            for topP in topPs {
                for repPen in repPenalties {
                    var params = AgentGenerateParameters.default
                    params.temperature = temp
                    params.topP = topP
                    params.repetitionPenalty = repPen
                    configs.append(params)
                }
            }
        }
        return configs
    }()

    /// Short label for a parameter config (used in filenames and charts).
    static func label(for params: AgentGenerateParameters) -> String {
        var parts = ["t=\(String(format: "%.1f", params.temperature))",
                     "p=\(String(format: "%.2f", params.topP))"]
        if let rp = params.repetitionPenalty {
            parts.append("rp=\(String(format: "%.2f", rp))")
        }
        return parts.joined(separator: "_")
    }

    /// Deterministic hash of parameters for filenames.
    static func paramHash(for params: AgentGenerateParameters) -> String {
        let str = label(for: params)
        var hash: UInt64 = 5381
        for byte in str.utf8 {
            hash = hash &* 33 &+ UInt64(byte)
        }
        return String(format: "%08x", UInt32(truncatingIfNeeded: hash))
    }

    /// Parse from `CommandLine.arguments`.
    static func fromCommandLine() -> BenchmarkConfig {
        let args = CommandLine.arguments

        let sweep: Sweep = {
            if let idx = args.firstIndex(of: "--bench-sweep"), idx + 1 < args.count {
                return Sweep(rawValue: args[idx + 1]) ?? .quick
            }
            return .quick
        }()

        let scenarioIDs: [String]? = {
            if let idx = args.firstIndex(of: "--bench-scenarios"), idx + 1 < args.count {
                return args[idx + 1].split(separator: ",").map(String.init)
            }
            return nil
        }()

        let outputDir: URL = {
            if let idx = args.firstIndex(of: "--bench-output"), idx + 1 < args.count {
                return URL(fileURLWithPath: args[idx + 1])
            }
            return DebugPaths.benchmark
        }()

        let modelDir: URL? = {
            if let idx = args.firstIndex(of: "--bench-model"), idx + 1 < args.count {
                return URL(fileURLWithPath: args[idx + 1])
            }
            return nil
        }()

        let modelID: String? = {
            if let idx = args.firstIndex(of: "--bench-model-id"), idx + 1 < args.count {
                return args[idx + 1]
            }
            return nil
        }()

        let maxTokensPerRound: Int = {
            if let idx = args.firstIndex(of: "--bench-max-tokens"), idx + 1 < args.count,
               let val = Int(args[idx + 1]) {
                return val
            }
            return 2048
        }()

        return BenchmarkConfig(
            sweep: sweep,
            scenarioIDs: scenarioIDs,
            outputDir: outputDir,
            modelDir: modelDir,
            modelID: modelID,
            maxTokensPerRound: maxTokensPerRound
        )
    }
}
