//
//  SpeechContentView.swift
//  tesseract
//
//  The Speech page: the **Reader** (ADR-0076). One surface of text that is
//  an editor at rest and the player while reading, one floating control bar,
//  and one display popover. Voices open in a sheet.
//

import AppKit
import SwiftUI

struct SpeechContentView: View {
    @Environment(SpeechReader.self) private var reader
    @Environment(SettingsManager.self) private var settings
    @Environment(\.controlActiveState) private var controlActiveState
    @State private var showsVoices = false
    @State private var showsDisplay = false

    /// Room under the text for the control bar, so the last line can scroll
    /// clear of it.
    private let barClearance: CGFloat = 96

    var body: some View {
        ReaderTextView(
            reader: reader, font: font, highlightStyle: settings.readerHighlight,
            isReading: reader.isReading, bottomInset: barClearance
        )
        .overlay {
            if reader.isEmpty { emptyState }
        }
        .overlay(alignment: .bottom) {
            ZStack(alignment: .bottom) {
                // Fades the text out under the bar instead of showing it
                // through the glass.
                LinearGradient(
                    colors: [.clear, Color(nsColor: .windowBackgroundColor).opacity(0.94)],
                    startPoint: .top, endPoint: .bottom
                )
                .frame(height: barClearance + 20)
                .allowsHitTesting(false)
                ReaderControlBar { showsVoices = true }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 18)
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) { SpeechEngineNotice() }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showsDisplay.toggle()
                } label: {
                    Label("Display", systemImage: "textformat.size")
                }
                .help("Text size, highlighting and the overlay")
                .popover(isPresented: $showsDisplay, arrowEdge: .bottom) {
                    ReaderDisplaySettings()
                }
            }
        }
        .sheet(isPresented: $showsVoices) { VoicesSheet() }
        .navigationTitle("Speech")
        .onChange(of: controlActiveState, initial: true) { _, state in
            reader.isInFront = state == .key
        }
        .onDisappear { reader.isInFront = false }
    }

    private var font: NSFont {
        let size = CGFloat(settings.readerTextSize)
        let base = NSFont.systemFont(ofSize: size)
        guard settings.readerTypeface == .serif,
            let serif = base.fontDescriptor.withDesign(.serif)
        else { return base }
        return NSFont(descriptor: serif, size: size) ?? base
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Text("Paste or type something to hear it read.")
                .font(.title3)
            HStack(spacing: 4) {
                Text("In any other app, select text and press")
                Text(settings.ttsHotkey.displayString)
                    .font(.callout.weight(.medium))
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(RoundedRectangle(cornerRadius: 4).strokeBorder(.tertiary))
            }
            .foregroundStyle(.secondary)
        }
        .foregroundStyle(.secondary)
        .allowsHitTesting(false)
    }
}

/// Speech can't start until the Voice Engine is on disk, and the engine
/// never downloads it, so say so before Play is pressed.
private struct SpeechEngineNotice: View {
    @EnvironmentObject private var downloadManager: ModelDownloadManager

    var body: some View {
        let id = ModelDefinition.defaultTextToSpeechModelID
        Group {
            switch downloadManager.status(for: id) {
            case .notDownloaded:
                row {
                    Image(systemName: "arrow.down.circle")
                    Text(
                        "Speech needs the Voice Engine (\(ModelDefinition.withID(id)?.sizeDescription ?? "")), which isn't downloaded yet."
                    )
                    Spacer(minLength: 8)
                    Button("Open Models") { (NSApp.delegate as? AppDelegate)?.navigateToModels() }
                        .buttonStyle(.borderless)
                }
            case .downloading(let progress):
                row {
                    ProgressView(value: progress).controlSize(.small).frame(width: 80)
                    Text("The Voice Engine is downloading: \(Int(progress * 100))%")
                    Spacer(minLength: 8)
                }
            case .downloaded, .verifying, .error:
                // `.error` is on the Models page; Play re-checks the disk.
                EmptyView()
            }
        }
        .onAppear { downloadManager.refreshStatus(for: id) }
    }

    private func row<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 8, content: content)
            .font(.callout)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .padding(.horizontal, 24)
            .padding(.vertical, 10)
    }
}
