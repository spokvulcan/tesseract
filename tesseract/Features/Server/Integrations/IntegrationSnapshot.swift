//
//  IntegrationSnapshot.swift
//  tesseract
//

import Foundation

/// Route paths for the Integration endpoints — shared between route
/// registration, the script renderer, and the Settings UI's Setup One-liner
/// so the three can never drift.
nonisolated enum IntegrationRoutes {
    static let openCodeSetupScript = "/integrations/opencode/setup.sh"
    static let openCodeMerge = "/integrations/opencode/merge"
}

/// The facts a client-config generator needs about the live server, captured
/// as plain values (CONTEXT.md → Client integrations): the port, every
/// downloaded agent model with its capabilities, and which model a freshly
/// configured client should default to.
nonisolated struct IntegrationSnapshot: Equatable, Sendable {

    struct Model: Equatable, Sendable {
        let id: String
        let displayName: String
        let visionCapable: Bool
        let contextLength: Int
    }

    let port: Int
    /// Downloaded agent models, in catalogue order — mirrors `/v1/models`:
    /// nothing a client could pick that the server would refuse.
    let models: [Model]
    /// The user's selected agent model when downloaded, else the first
    /// downloaded model, else `nil` (nothing downloaded yet).
    let defaultModelID: String?
}

nonisolated enum IntegrationSnapshotBuilder {

    /// Context length advertised per model — the same value `/v1/models`
    /// reports as `max_context_length`.
    static let contextLength = 262_144

    static func build(
        definitions: [ModelDefinition],
        statuses: [String: ModelStatus],
        selectedAgentModelID: String,
        port: Int,
        modelDirectory: (String) -> URL?
    ) -> IntegrationSnapshot {
        let models: [IntegrationSnapshot.Model] = definitions
            .filter { $0.category == .agent }
            .compactMap { definition in
                guard case .downloaded = statuses[definition.id] else { return nil }
                let visionCapable = modelDirectory(definition.id)
                    .map(ModelVisionCapability.isVisionCapable(directory:)) ?? false
                return IntegrationSnapshot.Model(
                    id: definition.id,
                    displayName: definition.displayName,
                    visionCapable: visionCapable,
                    contextLength: contextLength
                )
            }
        let defaultModelID = models.contains { $0.id == selectedAgentModelID }
            ? selectedAgentModelID
            : models.first?.id
        return IntegrationSnapshot(
            port: port,
            models: models,
            defaultModelID: defaultModelID
        )
    }

    /// Snapshot of the live app state, taken at request time so a re-run of
    /// the Setup One-liner always reflects current downloads and port.
    @MainActor
    static func current(
        downloads: ModelDownloadManager,
        settings: SettingsManager
    ) -> IntegrationSnapshot {
        build(
            definitions: ModelDefinition.all,
            statuses: downloads.statuses,
            selectedAgentModelID: settings.selectedAgentModelID,
            port: settings.serverPort,
            modelDirectory: { downloads.modelPath(for: $0) }
        )
    }
}
