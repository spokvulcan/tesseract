import Foundation
import MLXLMCommon

nonisolated enum TriAttentionDenseFallbackReason: String, Sendable, Codable, Equatable {
    case unsupportedModel
    case visionMode
    case missingCalibrationArtifact
    case mismatchedCalibrationArtifact
    case unavailableCalibrationArtifact
}

nonisolated struct TriAttentionRuntimeSelection: Sendable {
    let requestedConfiguration: TriAttentionConfiguration
    let effectiveConfiguration: TriAttentionConfiguration
    let fallbackReason: TriAttentionDenseFallbackReason?
    let calibrationArtifactLookup: TriAttentionCalibrationArtifactLookupResult?

    static let disabledDefault = Self(
        requestedConfiguration: .v1Disabled,
        effectiveConfiguration: .v1Disabled,
        fallbackReason: nil,
        calibrationArtifactLookup: nil
    )

    var isEffectivelyEnabled: Bool {
        effectiveConfiguration.enabled
    }

    static func resolve(
        requestedConfiguration: TriAttentionConfiguration,
        isParoModel: Bool,
        visionMode: Bool,
        calibrationArtifactLookup: TriAttentionCalibrationArtifactLookupResult?
    ) -> Self {
        if !requestedConfiguration.enabled {
            return Self(
                requestedConfiguration: requestedConfiguration,
                effectiveConfiguration: denseConfiguration(from: requestedConfiguration),
                fallbackReason: nil,
                calibrationArtifactLookup: nil
            )
        }

        if !isParoModel {
            return Self(
                requestedConfiguration: requestedConfiguration,
                effectiveConfiguration: denseConfiguration(from: requestedConfiguration),
                fallbackReason: .unsupportedModel,
                calibrationArtifactLookup: nil
            )
        }

        if visionMode {
            return Self(
                requestedConfiguration: requestedConfiguration,
                effectiveConfiguration: denseConfiguration(from: requestedConfiguration),
                fallbackReason: .visionMode,
                calibrationArtifactLookup: nil
            )
        }

        guard let calibrationArtifactLookup else {
            fatalError(
                "TriAttention runtime selection requires a calibration artifact lookup for eligible PARO text loads"
            )
        }

        switch calibrationArtifactLookup {
        case .loaded(_, let identity, _):
            return Self(
                requestedConfiguration: requestedConfiguration,
                effectiveConfiguration: enabledConfiguration(
                    from: requestedConfiguration,
                    calibrationArtifactIdentity: identity
                ),
                fallbackReason: nil,
                calibrationArtifactLookup: calibrationArtifactLookup
            )
        case .missing:
            return Self(
                requestedConfiguration: requestedConfiguration,
                effectiveConfiguration: denseConfiguration(from: requestedConfiguration),
                fallbackReason: .missingCalibrationArtifact,
                calibrationArtifactLookup: calibrationArtifactLookup
            )
        case .fingerprintMismatch:
            return Self(
                requestedConfiguration: requestedConfiguration,
                effectiveConfiguration: denseConfiguration(from: requestedConfiguration),
                fallbackReason: .mismatchedCalibrationArtifact,
                calibrationArtifactLookup: calibrationArtifactLookup
            )
        case .unavailable:
            return Self(
                requestedConfiguration: requestedConfiguration,
                effectiveConfiguration: denseConfiguration(from: requestedConfiguration),
                fallbackReason: .unavailableCalibrationArtifact,
                calibrationArtifactLookup: calibrationArtifactLookup
            )
        }
    }

    private static func denseConfiguration(
        from requestedConfiguration: TriAttentionConfiguration
    ) -> TriAttentionConfiguration {
        TriAttentionConfiguration(
            enabled: false,
            budgetTokens: requestedConfiguration.budgetTokens,
            calibrationArtifactIdentity: nil,
            implementationVersion: requestedConfiguration.implementationVersion
        )
    }

    private static func enabledConfiguration(
        from requestedConfiguration: TriAttentionConfiguration,
        calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity
    ) -> TriAttentionConfiguration {
        TriAttentionConfiguration(
            enabled: true,
            budgetTokens: requestedConfiguration.budgetTokens,
            calibrationArtifactIdentity: calibrationArtifactIdentity,
            implementationVersion: requestedConfiguration.implementationVersion
        )
    }
}
