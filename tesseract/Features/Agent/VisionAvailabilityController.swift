//
//  VisionAvailabilityController.swift
//  tesseract
//
//  The **Vision Availability** leaf: whether image input is available for the
//  selected agent model, and the one-tap remedy when it isn't — carved out of
//  `AgentComposerView`, where the probe cache, the availability transitions,
//  and the switch-hint projection lived as view orchestration nothing could
//  test. A publisher-agnostic leaf sibling to ``ComposerDraftController``:
//  it owns its own state, subscribes to no agent events, and the view keeps
//  only the refresh triggers.
//
//  The verdict itself stays the pure `ImageInputAvailability` projection
//  (ADR-0013); this leaf owns the *lifecycle around it* — when to re-probe,
//  what an availability flip does to the Composer Draft, and how the user
//  gets from "can't attach" to "can".
//

import Foundation
import Observation

@Observable @MainActor
final class VisionAvailabilityController {

    // MARK: - Observable State

    /// Whether the *selected* agent model can serve images. Probed off the
    /// view body (disk read via `ModelIdentity`) and cached here, refreshed
    /// only when the selection or its download status changes — never per
    /// keystroke.
    private(set) var selectedModelIsVisionCapable = false

    /// The availability verdict (ADR-0013): image affordances show only when
    /// the model is vision-capable *and* the global vision opt-in is on.
    private(set) var imageInputAvailable = false

    // MARK: - Dependencies

    @ObservationIgnored private let settings: SettingsManager
    /// The Composer Draft carries the availability facts this leaf derives:
    /// the mirror the full-window drop reads, the switch hint, and the
    /// pending images an availability loss would silently drop.
    @ObservationIgnored private let draft: ComposerDraftController
    @ObservationIgnored private let isVisionCapable: @MainActor (String) -> Bool
    @ObservationIgnored private let downloadedAgentModels: @MainActor () -> [ModelDefinition]

    // MARK: - Init

    init(
        settings: SettingsManager,
        draft: ComposerDraftController,
        isVisionCapable: @MainActor @escaping (String) -> Bool,
        downloadedAgentModels: @MainActor @escaping () -> [ModelDefinition]
    ) {
        self.settings = settings
        self.draft = draft
        self.isVisionCapable = isVisionCapable
        self.downloadedAgentModels = downloadedAgentModels
    }

    // MARK: - Refresh

    /// Re-probe the selected model, recompute the verdict, and reconcile the
    /// Composer Draft. Called on appear and whenever an input changes (model
    /// selection, its download status, the vision setting).
    func refresh() {
        selectedModelIsVisionCapable = isVisionCapable(settings.selectedAgentModelID)
        let available = ImageInputAvailability.showImageAffordance(
            isVisionCapable: selectedModelIsVisionCapable,
            useVisionWhenAvailable: settings.useVisionWhenAvailable
        )
        let changed = available != imageInputAvailable
        imageInputAvailable = available
        // Mirror unconditionally so the full-window drop (hosted above the
        // composer) can decide attach-vs-hint from the first appear (#117).
        draft.imageInputAvailable = available
        guard changed else { return }
        if available {
            // Vision input just became available (model switched / opt-in) —
            // the hint is moot.
            draft.showImageSwitchHint = false
        } else {
            // Clear any queued images when image input becomes unavailable
            // (model switched to text-only, or vision opted out) — the LLM
            // container would silently drop them.
            draft.pendingImages = []
        }
    }

    // MARK: - Image Switch Hint (slice #115)

    /// How the user can make image input available from the current state.
    enum Remedy: Equatable {
        /// The selected model is vision-capable but the global opt-in is off.
        case turnOnSetting
        /// A different downloaded model can see images — offer to switch to it.
        case switchModel(id: String, name: String)
        /// No vision-capable model is downloaded — nothing to switch to.
        case noVisionModel

        var message: String {
            switch self {
            case .turnOnSetting:
                "Vision is turned off. Turn it on to attach images."
            case .switchModel(_, let name):
                "This model can’t see images. Switch to \(name) to attach images."
            case .noVisionModel:
                "This model can’t see images. Download a vision model from Settings → Models."
            }
        }

        var actionTitle: String? {
            switch self {
            case .turnOnSetting: "Turn On"
            case .switchModel: "Switch"
            case .noVisionModel: nil
            }
        }
    }

    var remedy: Remedy {
        if selectedModelIsVisionCapable && !settings.useVisionWhenAvailable {
            return .turnOnSetting
        }
        if let model = firstDownloadedVisionModel() {
            return .switchModel(id: model.id, name: model.displayName)
        }
        return .noVisionModel
    }

    /// Apply the one-tap remedy: turn vision on, or switch to a vision model.
    /// The setting writes re-enter this leaf through the view's refresh
    /// triggers; the hint comes down either way.
    func applyRemedy() {
        switch remedy {
        case .turnOnSetting:
            settings.useVisionWhenAvailable = true
        case .switchModel(let id, _):
            settings.selectedAgentModelID = id
            if !settings.useVisionWhenAvailable {
                settings.useVisionWhenAvailable = true
            }
        case .noVisionModel:
            break
        }
        draft.showImageSwitchHint = false
    }

    /// The first downloaded agent model that can serve images, if any.
    private func firstDownloadedVisionModel() -> ModelDefinition? {
        downloadedAgentModels().first { isVisionCapable($0.id) }
    }
}
