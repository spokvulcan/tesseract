//
//  LibraryScreen.swift
//  tesseract-ios
//
//  The **Library**: every text you added, newest first, with how far each
//  has been read. Paste adds the clipboard's text and opens it.
//

import SwiftUI
import TesseractSpeech

struct LibraryScreen: View {
    @Environment(ReaderLibrary.self) private var library
    @Environment(PhoneReading.self) private var reading
    @Environment(PhoneIntake.self) private var intake
    @Environment(PhoneVoice.self) private var voice
    @State private var showsSettings = false
    @State private var showsFiles = false

    var body: some View {
        @Bindable var intake = intake
        NavigationStack(path: $intake.path) {
            List {
                if !voice.isReading {
                    NeuralVoiceBanner()
                }
                ForEach(library.entries) { entry in
                    NavigationLink(value: entry.id) {
                        LibraryRow(entry: entry, progress: library.progress[entry.id] ?? 0)
                    }
                }
                .onDelete { offsets in
                    for index in offsets {
                        let id = library.entries[index].id
                        reading.close(id)
                        library.remove(id)
                    }
                }
            }
            .listStyle(.plain)
            .overlay {
                if library.entries.isEmpty {
                    ContentUnavailableView(
                        "Nothing to read yet", systemImage: "text.page",
                        description: Text(
                            "Copy some text and tap Paste, share a page from Safari, or open a file."
                        ))
                }
            }
            .navigationTitle("Library")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Settings", systemImage: "gearshape") { showsSettings = true }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    PasteButton(payloadType: String.self) { strings in
                        let text = strings.joined(separator: "\n\n")
                        guard text.contains(where: { !$0.separatesWords }) else { return }
                        Task { @MainActor in
                            let entry = library.add(text)
                            intake.path = [entry.id]
                        }
                    }
                    .labelStyle(.iconOnly)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Open a file", systemImage: "folder") { showsFiles = true }
                }
            }
            .fileImporter(isPresented: $showsFiles, allowedContentTypes: TextIntake.fileTypes) {
                result in
                if case .success(let url) = result { intake.open(file: url) }
            }
            .alert(
                "Can't open this file",
                isPresented: Binding(
                    get: { intake.failure != nil }, set: { if !$0 { intake.failure = nil } })
            ) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(intake.failure ?? "")
            }
            .navigationDestination(for: UUID.self) { id in
                ReaderScreen(id: id)
            }
            .safeAreaInset(edge: .bottom) {
                if let now = reading.nowReading, intake.path.last != now.id {
                    NowReadingBar(
                        title: library.entry(now.id)?.title ?? "", reader: now.reader
                    ) { intake.path.append(now.id) }
                }
            }
            .onAppear { library.refreshProgress() }
            .sheet(isPresented: $showsSettings) { PhoneSettingsScreen() }
        }
    }
}

private struct LibraryRow: View {
    let entry: ReaderLibrary.Entry
    let progress: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.title)
                .font(.headline)
                .lineLimit(2)
            HStack(spacing: 8) {
                Text(entry.added, format: .dateTime.day().month(.abbreviated))
                Text(Self.length(entry.length))
                if progress > 0 {
                    Text(progress >= 0.995 ? "Read" : "\(Int(progress * 100))%")
                }
            }
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
        .accessibilityElement(children: .combine)
    }

    /// Reading time at an average pace of about 1,000 characters a minute.
    static func length(_ characters: Int) -> String {
        let minutes = max(1, Int((Double(characters) / 1_000).rounded()))
        return minutes < 60 ? "\(minutes) min" : "\(minutes / 60) h \(minutes % 60) min"
    }
}

/// What is being read, under the list: its title and play/pause.
private struct NowReadingBar: View {
    let title: String
    let reader: SpeechReader
    let onOpen: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Button(action: onOpen) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Reading")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            Button {
                reader.togglePause()
            } label: {
                Image(systemName: reader.isPaused ? "play.fill" : "pause.fill")
                    .font(.title3)
                    .frame(width: 44, height: 44)
            }
            .accessibilityLabel(reader.isPaused ? "Resume" : "Pause")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .glassEffect(.regular, in: .capsule)
        .padding(.horizontal, 12)
        .padding(.bottom, 4)
    }
}

/// Until the neural voice reads: what is happening with it, and the download
/// with its size the first time.
private struct NeuralVoiceBanner: View {
    @Environment(PhoneVoice.self) private var voice

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Neural voice", systemImage: "waveform")
                .font(.headline)
            Text(voice.summary)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if case .downloading(let received, let total) = voice.state, total > 0 {
                ProgressView(value: Double(received), total: Double(total))
            }
            if case .notDownloaded = voice.state {
                Button("Download \(ModelDefinition.phoneVoice.sizeDescription)") {
                    voice.download()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(.vertical, 6)
        .listRowSeparator(.hidden)
    }
}
