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
    @EnvironmentObject private var downloadManager: ModelDownloadManager

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
        .safeAreaInset(edge: .top, spacing: 0) {
            voiceEngineNotice
        }
        .safeAreaInset(edge: .bottom) {
            SpeechTransportBar(
                state: speechCoordinator.state,
                isModelLoading: speechEngine.isLoading,
                modelLoadingStatus: speechEngine.loadingStatus,
                hasText: !inputText.isEmpty,
                hotkeyHint: settings.ttsHotkey.displayString,
                onSpeak: { speechCoordinator.speakText(inputText, userInitiated: true) },
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
        .onAppear {
            downloadManager.refreshStatus(for: ModelDefinition.defaultTextToSpeechModelID)
        }
    }

    /// Speech can't start until the Voice Engine is on disk, and the engine
    /// never downloads it, so say so before Speak is pressed.
    @ViewBuilder
    private var voiceEngineNotice: some View {
        let id = ModelDefinition.defaultTextToSpeechModelID
        switch downloadManager.status(for: id) {
        case .notDownloaded:
            let size = ModelDefinition.withID(id)?.sizeDescription ?? ""
            noticeRow {
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.secondary)
                Text("Speech needs the Voice Engine (\(size)), which isn't downloaded yet.")
                    .foregroundStyle(.secondary)
                Spacer(minLength: Theme.Spacing.sm)
                Button("Open Models") {
                    (NSApp.delegate as? AppDelegate)?.navigateToModels()
                }
                .buttonStyle(.borderless)
            }
        case .downloading(let progress):
            noticeRow {
                ProgressView(value: progress)
                    .controlSize(.small)
                    .frame(width: 80)
                Text("The Voice Engine is downloading: \(Int(progress * 100))%")
                    .foregroundStyle(.secondary)
                Spacer(minLength: Theme.Spacing.sm)
            }
        case .downloaded, .verifying, .error:
            // `.error` is on the Models page; Speak re-checks the disk.
            EmptyView()
        }
    }

    private func noticeRow<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: Theme.Spacing.sm, content: content)
            .font(.callout)
            .padding(.horizontal, Theme.Spacing.xxl)
            .padding(.top, SpeechPageStyle.rhythm)
    }
}
