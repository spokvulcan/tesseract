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
        /// **Audio-capable** (CONTEXT.md): the model takes audio input —
        /// `input_audio` content parts serve natively.
        let audioCapable: Bool
        let contextLength: Int

        init(
            id: String,
            displayName: String,
            visionCapable: Bool,
            audioCapable: Bool = false,
            contextLength: Int
        ) {
            self.id = id
            self.displayName = displayName
            self.visionCapable = visionCapable
            self.audioCapable = audioCapable
            self.contextLength = contextLength
        }
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
        let models: [IntegrationSnapshot.Model] =
            ModelCatalog.downloaded(in: .agent, definitions: definitions, statuses: statuses)
            .map { definition in
                let directory = modelDirectory(definition.id)
                let visionCapable =
                    directory.map(ModelCatalog.isVisionCapable(directory:)) ?? false
                let audioCapable =
                    directory.map(ModelCatalog.isAudioCapable(directory:)) ?? false
                return IntegrationSnapshot.Model(
                    id: definition.id,
                    displayName: definition.displayName,
                    visionCapable: visionCapable,
                    audioCapable: audioCapable,
                    contextLength: contextLength
                )
            }
        let defaultModelID =
            models.contains { $0.id == selectedAgentModelID }
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
