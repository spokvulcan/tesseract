//
//  LFM3VoiceMode.swift
//  tesseract
//
//  **LFM3 Voice**: a voice session in the chat where LFM3-Audio is the ear,
//  the mind and the voice. It hears the owner's take, answers with the agent's
//  tools, and speaks the answer as it writes it, while the chat shows every
//  turn. The loop, its half-duplex turn order and the overlay are the voice
//  session's (ADR-0082). It is offered while an LFM3-Audio checkpoint is the
//  selected model, and loads its own copy of that checkpoint for as long as a
//  session lasts.
//

import Foundation
import Observation

@Observable @MainActor
final class LFM3VoiceMode {

    /// The catalog's LFM3-Audio checkpoints, the ones that hear and speak,
    /// named as `AgentGenerateParameters.forModel` names them.
    static func hearsAndSpeaks(_ modelID: String) -> Bool {
        modelID.hasPrefix("lfm3-audio")
    }

    /// The voice session loop it runs.
    @ObservationIgnored let session: CompanionVoiceSessionController
    @ObservationIgnored private let engine: LFM3VoiceEngine
    @ObservationIgnored private let turns: LFM3VoiceTurns
    /// The selected model's folder when it is a downloaded LFM3-Audio checkpoint.
    @ObservationIgnored private let checkpoint: @MainActor () -> URL?
    /// Ends Jarvis's voice session, which shares the overlay.
    @ObservationIgnored private let leaveVoiceSession: @MainActor () -> Void
    @ObservationIgnored private let onError: @MainActor (String) -> Void
    @ObservationIgnored private var unloading: Task<Void, Never>?

    init(
        session: CompanionVoiceSessionController,
        engine: LFM3VoiceEngine,
        turns: LFM3VoiceTurns,
        checkpoint: @escaping @MainActor () -> URL?,
        leaveVoiceSession: @escaping @MainActor () -> Void,
        onError: @escaping @MainActor (String) -> Void
    ) {
        self.session = session
        self.engine = engine
        self.turns = turns
        self.checkpoint = checkpoint
        self.leaveVoiceSession = leaveVoiceSession
        self.onError = onError
        session.onSessionEnded = { [weak self] in self?.unloadWhenSettled() }
    }

    var isActive: Bool { session.isActive }

    func toggle() {
        if isActive { session.exit(reason: "toggled-off") } else { enter() }
    }

    private func enter() {
        guard let directory = checkpoint() else {
            onError("LFM3 Voice needs LFM3-Audio: download it and pick it as the model.")
            return
        }
        unloading?.cancel()
        unloading = nil
        leaveVoiceSession()
        session.enter(via: "lfm3-voice")
        // The load runs while the owner speaks the first take; the take's
        // transcription waits for it.
        Task { [weak self, engine] in
            do {
                try await engine.load(from: directory)
            } catch {
                Log.agent.error("LFM3 Voice failed to load: \(error.localizedDescription)")
                self?.onError("LFM3 Voice couldn't load LFM3-Audio: \(error.localizedDescription)")
                self?.session.exit(reason: "lfm3-load-failed")
            }
        }
    }

    /// The session ended: once its last turn settles (its answer still lands
    /// in the chat), the model leaves memory, unless a new session began.
    private func unloadWhenSettled() {
        unloading = Task { [turns, engine] in
            await turns.settle()
            guard !Task.isCancelled else { return }
            await engine.unload()
        }
    }
}
