//
//  DependencyContainer+LFM3Voice.swift
//  tesseract
//
//  LFM3 Voice's graph: its own LFM3-Audio engine, the turns that answer with
//  the agent's tools and land in the chat, and a voice session loop whose
//  ear and voice are LFM3-Audio's.
//

import Foundation

extension DependencyContainer {

    func makeLFM3Voice() -> LFM3VoiceMode {
        let engine = LFM3VoiceEngine()
        let playback = LFM3VoicePlayback()
        let turns = LFM3VoiceTurns(
            engine: engine,
            playback: playback,
            tools: { [weak self] in
                guard let self else { return [] }
                let live = ActiveToolSet.resolve(
                    from: newToolRegistry.allTools,
                    gating: ToolGating(webAccessEnabled: settingsManager.webAccessEnabled))
                return LFM3VoiceTurns.toolNames.compactMap { name in
                    live.first { $0.name == name }
                }
            },
            chatMessages: { [weak self] in self?.agent.state.messages ?? [] },
            commit: { [weak self] messages in
                self?.chatSession.appendCommitted(messages) ?? false
            },
            onError: { [weak self] message in self?.chatSession.error = message }
        )
        let session = CompanionVoiceSessionController(
            capture: voiceCapture.makeSession(recognizer: LFM3VoiceTranscriber(engine: engine)),
            meterLevel: { [dictationFeed] in dictationFeed.level },
            meterSpectrum: { [dictationFeed] in dictationFeed.spectrum },
            inputDead: { [audioCaptureEngine] in audioCaptureEngine.isInputDead },
            sendMessage: { [turns] text in turns.send(text) },
            stageToComposer: { [composerDraft] text in
                composerDraft.restore(text: text, images: [])
            },
            // The answer speaks itself as it is written: the session's speech
            // only waits for the turn's audio to play out.
            speak: { [playback] _, onDone in playback.speak(onDone: onDone) },
            speakMore: { _ in },
            finishSpeaking: { [playback] in playback.finishSpeaking() },
            stopSpeaking: { [playback] in playback.stop() },
            cancelReply: { [turns] in turns.cancel() },
            speechState: { [weak self, playback] in
                playback.isActive ? .playing : (self?.speechCoordinator.state ?? .idle)
            },
            currentConversationID: { [weak self] in
                self?.agentConversationStore.currentConversation?.id
            },
            overlay: companionVoicePrototype,
            recorder: companionTrace,
            settings: settingsManager
        )
        turns.replyText = { [weak session] piece in session?.replyText(piece) }
        turns.replyCompleted = { [weak session] text in _ = session?.replyCompleted(text) }
        return LFM3VoiceMode(
            session: session,
            engine: engine,
            turns: turns,
            checkpoint: { [weak self] in
                guard let self else { return nil }
                let id = settingsManager.selectedAgentModelID
                guard LFM3VoiceMode.hearsAndSpeaks(id) else { return nil }
                return modelDownloadManager.usableDirectory(for: id)
            },
            leaveVoiceSession: { [weak self] in
                self?.builtVoiceSession?.exit(reason: "lfm3-voice")
            },
            onError: { [weak self] message in self?.chatSession.error = message }
        )
    }

    /// The voice session that is live, Jarvis's or LFM3 Voice's: they share
    /// the overlay, so never both.
    var liveVoiceSession: CompanionVoiceSessionController? {
        [builtVoiceSession, builtLFM3Voice?.session].lazy.compactMap { $0 }.first(where: \.isActive)
    }
}
