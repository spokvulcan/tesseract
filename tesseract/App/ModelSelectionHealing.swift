//
//  ModelSelectionHealing.swift
//  tesseract
//
//  The availability-follows-selection rules as one pure decider (#406):
//  a model selection changes only because of what is (or is not) on disk,
//  and an available selection is never overridden. Two rules live here —
//  the dictation-model heal and the Companion agent-default adoption
//  (ADR-0040 §9) — decided value-in/value-out; `AppBindings` performs the
//  settings writes and the logging.
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

    /// The Companion-model default rule (ADR-0040 §9): an enabled Companion
    /// adopts its own model as the interactive agent default — but an
    /// undownloaded model must never become the default, an empty companion
    /// model decides nothing, and a selection already matching stands.
    static func adoptedCompanionAgentDefault(
        companionEnabled: Bool,
        companionModelID: String,
        selectedAgentModelID: String,
        isDownloaded: (String) -> Bool
    ) -> String? {
        guard companionEnabled, !companionModelID.isEmpty,
            companionModelID != selectedAgentModelID,
            isDownloaded(companionModelID)
        else { return nil }
        return companionModelID
    }
}
