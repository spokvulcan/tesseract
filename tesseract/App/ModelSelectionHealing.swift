//
//  ModelSelectionHealing.swift
//  tesseract
//
//  The availability-follows-selection rule as one pure decider (#406):
//  a model selection changes only because of what is (or is not) on disk,
//  and an available selection is never overridden. The dictation-model heal
//  is decided value-in/value-out; `AppBindings` performs the settings write
//  and the logging. Nothing else ever switches a selected model.
//

import Foundation

nonisolated enum ModelSelectionHealing {

    /// Selection follows availability only when the selected model is
    /// missing: if the selected speech-to-text model is not on disk but
    /// another variant is, return the downloaded variant's id to flip to.
    /// Covers a fresh install that downloads only the compact variant and
    /// deletion of the selected variant — dictation is never silently dead
    /// while a speech model exists on disk. An available selection is never
    /// overridden; nothing downloaded means nothing to heal to.
    static func healedSpeechToTextSelection(
        selectedID: String,
        definitions: [ModelDefinition],
        statuses: [String: ModelStatus]
    ) -> String? {
        if ModelCatalog.isDownloaded(selectedID, statuses: statuses) { return nil }
        return ModelCatalog.downloaded(
            in: .speechToText, definitions: definitions, statuses: statuses
        ).first?.id
    }
}
