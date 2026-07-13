//
//  SpeechContentView.swift
//  tesseract
//

import SwiftUI

/// Speech page surface constants (design language §2: one type size and
/// one spacing rhythm per surface; hierarchy comes from weight and color).
enum SpeechPageStyle {
    static let bodySize: CGFloat = 15
    static let rhythm: CGFloat = 12
}

struct SpeechContentView: View {
    @Environment(SpeechCoordinator.self) private var speechCoordinator
    @Environment(SpeechEnginePresenter.self) private var speechEngine
    @Environment(SettingsManager.self) private var settings

    @AppStorage("ttsParametersPanelVisible") private var isParametersPanelVisible: Bool = true
    @State private var inputText: String = ""

    var body: some View {
        @Bindable var settings = settings
        SpeechComposerView(
            text: $inputText,
            voiceDescription: $settings.ttsVoiceDescription,
            language: $settings.ttsLanguage
        )
        .padding(.horizontal, Theme.Spacing.xxl)
        .padding(.top, SpeechPageStyle.rhythm)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .safeAreaInset(edge: .bottom) {
            SpeechTransportBar(
                state: speechCoordinator.state,
                isModelLoading: speechEngine.isLoading,
                modelLoadingStatus: speechEngine.loadingStatus,
                hasText: !inputText.isEmpty,
                hotkeyHint: settings.ttsHotkey.displayString,
                onSpeak: { speechCoordinator.speakText(inputText) },
                onStop: { speechCoordinator.stop() },
                onPause: { speechCoordinator.pause() },
                onResume: { speechCoordinator.resume() }
            )
            .padding(.horizontal, Theme.Spacing.xxl)
            .padding(.vertical, SpeechPageStyle.rhythm)
        }
        .inspector(isPresented: $isParametersPanelVisible) {
            TTSParametersInspector()
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    isParametersPanelVisible.toggle()
                } label: {
                    Label("Parameters", systemImage: "slider.horizontal.3")
                }
                .help("Show or hide the generation parameters")
            }
        }
        .navigationTitle("Speech")
    }
}
