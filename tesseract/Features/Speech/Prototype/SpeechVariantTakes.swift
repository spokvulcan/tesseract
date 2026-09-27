//
//  SpeechVariantTakes.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  D · Takes — making an audio file, not just listening (ElevenLabs Studio,
//  Murf blocks, WellSaid takes, Descript's text-as-audio). The script is a
//  column of blocks; each block has its own take, rendered offline (faster
//  than real time, nothing plays). Edit a block and only that block goes
//  stale. Retake a block and keep every take to choose from; lock the good
//  ones, mute one without deleting it, drop in pauses. A timeline ribbon
//  shows the assembled piece; export it as one file with captions.
//  Signature: the status bar beside every block — grey, rendering, done,
//  stale — the whole production state at a glance.
//

import AppKit
import SwiftUI
import TesseractSpeech

// MARK: - Project

@Observable @MainActor
final class TakesProject {
    static let shared = TakesProject()

    enum Split: String, CaseIterable, Identifiable {
        case paragraphs, sentences
        var id: String { rawValue }
        var label: String { self == .paragraphs ? "Paragraphs" : "Sentences" }
    }

    struct Block: Identifiable, Equatable {
        let id: UUID
        var text: String
        /// Non-nil: a pause of this many seconds instead of speech.
        var pause: Double?
        var chosenTakeID: UUID?
        var isLocked = false
        var isMuted = false

        init(text: String, pause: Double? = nil) {
            id = UUID()
            self.text = text
            self.pause = pause
        }
    }

    enum BlockState: Equatable {
        case pause, muted, empty, rendering(Double), ready, stale(String)
    }

    var blocks: [Block] = []
    var split: Split = .paragraphs
    /// Silence between consecutive speech blocks, seconds.
    var gap: Double = 0.35
    var trimSilence = true
    var normalize = true
    private(set) var isRenderingAll = false
    @ObservationIgnored private var renderAllTask: Task<Void, Never>?
    @ObservationIgnored private let assembledID = UUID()

    private var lab: SpeechLab { SpeechLab.shared }

    // MARK: Building blocks

    var joinedText: String {
        blocks.filter { $0.pause == nil }.map(\.text).joined(separator: "\n\n")
    }

    var hasTakes: Bool { blocks.contains { !lab.takes(forBlock: $0.id).isEmpty } }

    func load(from draft: String) {
        let document = ReadingDocument(text: draft)
        switch split {
        case .paragraphs:
            blocks = document.paragraphs.map {
                Block(text: String(draft[$0.range]).trimmingCharacters(in: .whitespacesAndNewlines))
            }
        case .sentences:
            blocks = document.sentences.map {
                Block(text: String(draft[$0.range]).trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
        blocks.removeAll { $0.text.isEmpty }
    }

    func syncDraft() {
        let text = joinedText
        if Self.normalized(text) != Self.normalized(lab.draft) { lab.draft = text }
    }

    static func normalized(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    // MARK: Takes

    func takes(for block: Block) -> [SpeechTake] { lab.takes(forBlock: block.id) }

    func chosenTake(for block: Block) -> SpeechTake? {
        let takes = takes(for: block)
        return takes.first { $0.id == block.chosenTakeID } ?? takes.first
    }

    func state(of block: Block) -> BlockState {
        if block.pause != nil { return .pause }
        if block.isMuted { return .muted }
        if let render = lab.render, render.blockID == block.id {
            return .rendering(render.progress)
        }
        if lab.liveTake?.blockID == block.id { return .rendering(0) }
        guard let take = chosenTake(for: block) else { return .empty }
        if Self.normalized(take.text) != Self.normalized(block.text) {
            return .stale("Text changed")
        }
        if take.voiceDescription != lab.currentVoiceDescription {
            return .stale("Rendered as \(take.voiceName)")
        }
        return .ready
    }

    var readyCount: Int { blocks.filter { state(of: $0) == .ready }.count }
    var speechCount: Int { blocks.filter { $0.pause == nil && !$0.isMuted }.count }

    func render(_ block: Block, newTake: Bool = false) async {
        syncDraft()
        let seed = newTake ? Int.random(in: 0...99_999) : nil
        if let take = await lab.renderOffline(block.text, blockID: block.id, seed: seed),
            let index = blocks.firstIndex(where: { $0.id == block.id })
        {
            blocks[index].chosenTakeID = take.id
        }
    }

    func renderAll() {
        renderAllTask?.cancel()
        isRenderingAll = true
        renderAllTask = Task {
            defer { isRenderingAll = false }
            for block in blocks {
                guard !Task.isCancelled else { return }
                guard !block.isLocked else { continue }
                switch state(of: block) {
                case .empty, .stale: await render(block)
                default: continue
                }
            }
        }
    }

    func cancelRenderAll() {
        renderAllTask?.cancel()
        renderAllTask = nil
        isRenderingAll = false
    }

    func choose(_ take: SpeechTake, for block: Block) {
        guard let index = blocks.firstIndex(where: { $0.id == block.id }) else { return }
        blocks[index].chosenTakeID = take.id
    }

    // MARK: Editing

    func insertPause(after block: Block, seconds: Double = 0.75) {
        guard let index = blocks.firstIndex(where: { $0.id == block.id }) else { return }
        blocks.insert(Block(text: "", pause: seconds), at: index + 1)
    }

    func addBlock() {
        blocks.append(Block(text: ""))
    }

    func delete(_ block: Block) {
        blocks.removeAll { $0.id == block.id }
    }

    func move(_ block: Block, by delta: Int) {
        guard let index = blocks.firstIndex(where: { $0.id == block.id }) else { return }
        let target = index + delta
        guard blocks.indices.contains(target) else { return }
        blocks.swapAt(index, target)
    }

    // MARK: Assembly

    struct Span: Equatable {
        let blockID: UUID
        let start: TimeInterval
        let end: TimeInterval
        let kind: Kind
        enum Kind: Equatable { case speech, pause, missing }
    }

    /// The whole piece: chosen takes in order, trimmed and levelled if asked,
    /// with gaps and pauses. Blocks without a take are left out (and shown
    /// as missing on the ribbon).
    func assemble() -> (take: SpeechTake, spans: [Span])? {
        var samples: [Float] = []
        var spans: [Span] = []
        var segments: [ReadAlongMap.Segment] = []
        var sampleRate = 24_000
        var previousWasSpeech = false
        for block in blocks {
            let start = Double(samples.count) / Double(sampleRate)
            if let pause = block.pause {
                samples += TakeEditing.silence(seconds: pause, sampleRate: sampleRate)
                spans.append(
                    Span(blockID: block.id, start: start, end: start + pause, kind: .pause))
                previousWasSpeech = false
                continue
            }
            guard !block.isMuted else { continue }
            guard let take = chosenTake(for: block) else {
                spans.append(Span(blockID: block.id, start: start, end: start, kind: .missing))
                continue
            }
            sampleRate = take.audio.sampleRate
            if previousWasSpeech {
                samples += TakeEditing.silence(seconds: gap, sampleRate: sampleRate)
            }
            var clip = take.audio.samples
            var leadTrim = 0.0
            if trimSilence {
                let trimmed = TakeEditing.trimmingSilence(clip, sampleRate: sampleRate)
                if let first = clip.firstIndex(where: { abs($0) > 0.01 }) {
                    leadTrim =
                        Double(max(0, first - Int(Double(sampleRate) * 0.12))) / Double(sampleRate)
                }
                clip = trimmed
            }
            if normalize { clip = TakeEditing.normalized(clip) }
            clip = TakeEditing.faded(clip, sampleRate: sampleRate)
            let clipStart = Double(samples.count) / Double(sampleRate)
            let wordBase = segments.last.map { $0.firstWord + $0.wordCount } ?? 0
            for segment in take.segments {
                segments.append(
                    ReadAlongMap.Segment(
                        timeline: segment.timeline, base: clipStart + segment.base - leadTrim,
                        end: segment.end.map { clipStart + $0 - leadTrim },
                        firstWord: wordBase + segment.firstWord))
            }
            samples += clip
            spans.append(
                Span(
                    blockID: block.id, start: clipStart,
                    end: Double(samples.count) / Double(sampleRate), kind: .speech))
            previousWasSpeech = true
        }
        guard !samples.isEmpty else { return nil }
        let text = blocks.filter { $0.pause == nil && !$0.isMuted }.map(\.text).joined(
            separator: " ")
        let take = SpeechTake(
            id: assembledID, createdAt: .now, text: text, source: .render, blockID: nil,
            voiceDescription: lab.currentVoiceDescription, language: lab.currentLanguage,
            parameters: lab.settings?.ttsParameters ?? .init(), seed: lab.settings?.ttsSeed ?? 0,
            audio: TakeAudio(samples: samples, sampleRate: sampleRate), segments: segments,
            wordOffset: 0, isComplete: true)
        return (take, spans)
    }

    func playAll(from time: TimeInterval = 0) {
        guard let assembled = assemble() else { return }
        lastAssembly = assembled.spans
        lab.player.play(assembled.take, from: time)
    }

    /// The ribbon's spans from the last assembly.
    private(set) var lastAssembly: [Span] = []

    var isPlayingAll: Bool { lab.player.takeID == assembledID }

    func export(_ format: TakeExportFormat) {
        guard let assembled = assemble() else { return }
        TakeExport.save(
            samples: assembled.take.audio.samples, sampleRate: assembled.take.audio.sampleRate,
            suggestedName: TakeExport.fileStem(for: assembled.take.title), format: format)
    }

    func exportCaptions() {
        guard let assembled = assemble() else { return }
        TakeExport.saveCaptions(assembled.take)
    }

    /// Estimated length of the piece: real takes where there are any,
    /// a reading-speed estimate elsewhere.
    var estimatedDuration: TimeInterval {
        blocks.reduce(0) { total, block in
            if let pause = block.pause { return total + pause }
            if block.isMuted { return total }
            if let take = chosenTake(for: block) { return total + take.duration + gap }
            return total + Double(SpeechLab.estimatedSeconds(block.text)) + gap
        }
    }
}

// MARK: - Page

struct TakesVariant: View {
    @State private var showsReloadBanner = false

    var body: some View {
        let project = TakesProject.shared
        let lab = SpeechLab.shared
        VStack(spacing: 0) {
            SpeechEngineNotice()
            if showsReloadBanner {
                HStack(spacing: 10) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                    Text("The text changed in another view.")
                        .lineLimit(1)
                    Spacer()
                    Button("Rebuild Blocks") {
                        project.load(from: lab.draft)
                        showsReloadBanner = false
                    }
                    Button("Keep These") { showsReloadBanner = false }
                }
                .font(.callout)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 20)
                .padding(.vertical, 8)
                .background(.fill.quinary)
            }
            TakesSummary()
                .padding(.horizontal, 24)
                .padding(.top, 14)
                .padding(.bottom, 6)
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(Bindable(project).blocks) { $block in
                        BlockRow(block: $block)
                    }
                    Button {
                        project.addBlock()
                    } label: {
                        Label("Add Block", systemImage: "plus")
                    }
                    .buttonStyle(.borderless)
                    .padding(.vertical, 8)
                }
                .frame(maxWidth: 860)
                .padding(.horizontal, 24)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity)
            }
            TimelineRibbon()
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(.bar)
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Picker("Split", selection: Bindable(project).split) {
                    ForEach(TakesProject.Split.allCases) { split in Text(split.label).tag(split) }
                }
                .pickerStyle(.menu)
                .fixedSize()
                .help("Split the text into blocks by paragraph or by sentence")
                .onChange(of: project.split) { _, _ in project.load(from: lab.draft) }

                if project.isRenderingAll {
                    Button {
                        project.cancelRenderAll()
                    } label: {
                        Label("Stop Rendering", systemImage: "stop.circle")
                    }
                } else {
                    Button {
                        project.renderAll()
                    } label: {
                        Label("Render All", systemImage: "waveform.badge.plus")
                    }
                    .help("Render every block that has no take or a stale one, without playing it")
                }

                Menu {
                    Button("Export as WAV…") { project.export(.wav) }
                    Button("Export as M4A…") { project.export(.m4a) }
                    Button("Export Captions (SRT)…") { project.exportCaptions() }
                    Divider()
                    Toggle("Trim Silence at Block Edges", isOn: Bindable(project).trimSilence)
                    Toggle("Level Loudness", isOn: Bindable(project).normalize)
                    Picker("Gap Between Blocks", selection: Bindable(project).gap) {
                        Text("None").tag(0.0)
                        Text("0.2 s").tag(0.2)
                        Text("0.35 s").tag(0.35)
                        Text("0.6 s").tag(0.6)
                        Text("1 s").tag(1.0)
                    }
                } label: {
                    Label("Export", systemImage: "square.and.arrow.up")
                }
                .help("Export the assembled audio")

                OverlayToolbarButton()
            }
        }
        .onAppear {
            if project.blocks.isEmpty {
                project.load(from: lab.draft)
            } else if TakesProject.normalized(project.joinedText)
                != TakesProject.normalized(lab.draft)
            {
                if project.hasTakes {
                    showsReloadBanner = true
                } else {
                    project.load(from: lab.draft)
                }
            }
        }
        .onDisappear { project.syncDraft() }
    }
}

private struct TakesSummary: View {
    var body: some View {
        let project = TakesProject.shared
        HStack(spacing: 12) {
            Text(
                "\(project.blocks.count) blocks · \(project.readyCount) of \(project.speechCount) rendered · \(SpeechLabFormat.time(project.estimatedDuration))"
            )
            .foregroundStyle(.secondary)
            .monospacedDigit()
            .lineLimit(1)
            Spacer()
            SpeechStatusLine()
                .lineLimit(1)
        }
        .font(.callout)
        .frame(maxWidth: 860)
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Block row

private struct BlockRow: View {
    @Binding var block: TakesProject.Block
    @State private var hovering = false

    var body: some View {
        let project = TakesProject.shared
        let state = project.state(of: block)
        if let pause = block.pause {
            pauseRow(pause)
        } else {
            HStack(alignment: .top, spacing: 12) {
                StatusBar(state: state)
                    .frame(width: 4)
                VStack(alignment: .leading, spacing: 10) {
                    TextField(
                        "Block", text: $block.text, prompt: Text("Write this block…"),
                        axis: .vertical
                    )
                    .labelsHidden()
                    .textFieldStyle(.plain)
                    .font(.system(size: 15))
                    .lineSpacing(3)
                    .strikethrough(block.isMuted, color: .secondary)
                    .foregroundStyle(block.isMuted ? .secondary : .primary)
                    .disabled(block.isLocked)
                    HStack(spacing: 10) {
                        takeArea(state: state)
                        Spacer(minLength: 8)
                        controls(state: state)
                    }
                    .font(.callout)
                }
            }
            .padding(.vertical, 12)
            .padding(.horizontal, 14)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(hovering ? AnyShapeStyle(.fill.quaternary) : AnyShapeStyle(.fill.quinary))
            )
            .onHover { hovering = $0 }
            .contextMenu { menuItems }
        }
    }

    private func pauseRow(_ seconds: Double) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "pause.circle")
                .foregroundStyle(.secondary)
            Text("Pause")
                .foregroundStyle(.secondary)
            Picker(
                "Pause",
                selection: Binding(get: { block.pause ?? seconds }, set: { block.pause = $0 })
            ) {
                ForEach([0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0], id: \.self) { value in
                    Text(String(format: "%g s", value)).tag(value)
                }
            }
            .labelsHidden()
            .fixedSize()
            Spacer()
            Button {
                TakesProject.shared.delete(block)
            } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.borderless)
            .help("Remove the pause")
        }
        .font(.callout)
        .padding(.horizontal, 30)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                .foregroundStyle(.quaternary)
        )
    }

    @ViewBuilder
    private func takeArea(state: TakesProject.BlockState) -> some View {
        let project = TakesProject.shared
        let lab = SpeechLab.shared
        switch state {
        case .rendering(let progress):
            ProgressView(value: max(progress, 0.03))
                .frame(width: 120)
            Text("Rendering…").foregroundStyle(.secondary)
        case .empty:
            Text("Not rendered").foregroundStyle(.tertiary)
        case .muted:
            Text("Muted: left out of the export").foregroundStyle(.tertiary)
        case .ready, .stale:
            if let take = project.chosenTake(for: block) {
                Button {
                    lab.player.toggle(take)
                } label: {
                    Image(
                        systemName: lab.player.isCurrent(take) && lab.player.isPlaying
                            ? "pause.circle.fill" : "play.circle.fill"
                    )
                    .font(.system(size: 20))
                    .foregroundStyle(.tint)
                }
                .buttonStyle(.plain)
                TakeScrubber(take: take, height: 20)
                    .frame(minWidth: 40, maxWidth: min(max(take.duration * 16, 60), 280))
                Text(SpeechLabFormat.time(take.duration))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                if case .stale(let reason) = state {
                    Text(reason)
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.orange.opacity(0.18)))
                        .foregroundStyle(.orange)
                        .lineLimit(1)
                }
            }
        case .pause:
            EmptyView()
        }
    }

    @ViewBuilder
    private func controls(state: TakesProject.BlockState) -> some View {
        let project = TakesProject.shared
        let takes = project.takes(for: block)
        if takes.count > 1 {
            Menu {
                ForEach(Array(takes.enumerated()), id: \.element.id) { index, take in
                    Button {
                        project.choose(take, for: block)
                        SpeechLab.shared.player.play(take)
                    } label: {
                        let chosen = project.chosenTake(for: block)?.id == take.id
                        Text(
                            "Take \(takes.count - index) · \(SpeechLabFormat.time(take.duration))\(chosen ? "  ✓" : "")"
                        )
                    }
                }
            } label: {
                Text(
                    "Take \(takes.count - (takes.firstIndex { $0.id == project.chosenTake(for: block)?.id } ?? 0)) of \(takes.count)"
                )
                .monospacedDigit()
            }
            .fixedSize()
            .help("Every take of this block; pick the one to keep")
        }
        if case .rendering = state {
            EmptyView()
        } else if case .empty = state {
            Button("Render") { Task { await project.render(block) } }
                .disabled(block.isLocked || block.isMuted || block.text.isEmpty)
        } else if state != .muted {
            Menu {
                Button("Re-render") { Task { await project.render(block) } }
                Button("Try Another Take") { Task { await project.render(block, newTake: true) } }
                Button("Listen Live") { SpeechLab.shared.speak(block.text, blockID: block.id) }
            } label: {
                Image(systemName: "arrow.triangle.2.circlepath")
            }
            .menuIndicator(.hidden)
            .fixedSize()
            .disabled(block.isLocked)
            .help("Re-render or try another take of this block")
        }
        Button {
            block.isLocked.toggle()
        } label: {
            Image(systemName: block.isLocked ? "lock.fill" : "lock.open")
                .foregroundStyle(block.isLocked ? Color.accentColor : .secondary)
        }
        .buttonStyle(.borderless)
        .help(
            block.isLocked
                ? "Unlock: allow edits and re-renders"
                : "Lock this block: Render All leaves it alone")
        Menu {
            menuItems
        } label: {
            Image(systemName: "ellipsis")
        }
        .menuIndicator(.hidden)
        .buttonStyle(.borderless)
        .fixedSize()
    }

    @ViewBuilder
    private var menuItems: some View {
        let project = TakesProject.shared
        Button(block.isMuted ? "Unmute" : "Mute (Keep, Leave Out of Export)") {
            block.isMuted.toggle()
        }
        Menu("Insert Pause After") {
            ForEach([0.25, 0.5, 0.75, 1.0, 1.25, 2.0], id: \.self) { value in
                Button(String(format: "%g s", value)) {
                    project.insertPause(after: block, seconds: value)
                }
            }
        }
        Divider()
        Button("Move Up") { project.move(block, by: -1) }
        Button("Move Down") { project.move(block, by: 1) }
        Divider()
        if let take = project.chosenTake(for: block) {
            Button("Export This Block…") { TakeExport.save(take, format: .wav) }
        }
        Button("Delete Block", role: .destructive) { project.delete(block) }
    }
}

/// ElevenLabs Studio's paragraph status bar, one step richer.
private struct StatusBar: View {
    let state: TakesProject.BlockState

    var body: some View {
        switch state {
        case .rendering:
            TimelineView(.animation(minimumInterval: 1.0 / 30)) { context in
                let phase = context.date.timeIntervalSinceReferenceDate
                Capsule()
                    .fill(Color.accentColor.opacity(0.45 + 0.4 * abs(sin(phase * 3))))
            }
        case .ready: Capsule().fill(Color.green.opacity(0.8))
        case .stale: Capsule().fill(Color.orange.opacity(0.85))
        case .muted: Capsule().fill(.quaternary)
        case .empty, .pause: Capsule().fill(.tertiary)
        }
    }
}

// MARK: - Timeline

private struct TimelineRibbon: View {
    var body: some View {
        let project = TakesProject.shared
        let lab = SpeechLab.shared
        HStack(spacing: 14) {
            Button {
                if project.isPlayingAll, lab.player.isPlaying {
                    lab.player.pause()
                } else if project.isPlayingAll, let assembled = project.assemble() {
                    lab.player.play(assembled.take)
                } else {
                    project.playAll()
                }
            } label: {
                Image(
                    systemName: project.isPlayingAll && lab.player.isPlaying
                        ? "pause.fill" : "play.fill"
                )
                .font(.system(size: 15, weight: .semibold))
                .frame(width: 26, height: 26)
            }
            .buttonStyle(.glassProminent)
            .buttonBorderShape(.circle)
            .controlSize(.large)
            .disabled(
                project.readyCount == 0
                    && !project.blocks.contains { project.chosenTake(for: $0) != nil }
            )
            .help("Play the assembled piece")

            GeometryReader { proxy in
                ribbon(width: proxy.size.width)
            }
            .frame(height: 34)

            TimelineView(.animation(minimumInterval: 0.25, paused: !project.isPlayingAll)) { _ in
                Text(
                    project.isPlayingAll
                        ? "\(SpeechLabFormat.time(lab.player.currentTime)) / \(SpeechLabFormat.time(lab.player.duration))"
                        : SpeechLabFormat.time(project.estimatedDuration)
                )
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .font(.callout)
            }
            .fixedSize()
        }
    }

    /// Blocks laid end to end by duration: rendered, stale, missing and pauses.
    private func ribbon(width: CGFloat) -> some View {
        let project = TakesProject.shared
        let lab = SpeechLab.shared
        let items: [(id: UUID, seconds: Double, state: TakesProject.BlockState, peaks: [Float])] =
            project.blocks.compactMap { block in
                let state = project.state(of: block)
                if state == .muted { return nil }
                if let pause = block.pause { return (block.id, pause, state, []) }
                if let take = project.chosenTake(for: block) {
                    return (block.id, take.duration, state, take.audio.peaks)
                }
                return (block.id, Double(SpeechLab.estimatedSeconds(block.text)), state, [])
            }
        let total = max(items.reduce(0) { $0 + $1.seconds }, 0.1)
        let spacing: CGFloat = 3
        let available = max(width - spacing * CGFloat(max(items.count - 1, 0)), 1)
        return ZStack(alignment: .leading) {
            HStack(spacing: spacing) {
                ForEach(items, id: \.id) { item in
                    let itemWidth = max(available * item.seconds / total, 4)
                    ZStack {
                        switch item.state {
                        case .ready, .stale:
                            RoundedRectangle(cornerRadius: 5)
                                .fill(
                                    item.state == .ready
                                        ? Color.accentColor.opacity(0.18)
                                        : Color.orange.opacity(0.18))
                            WaveformBars(
                                peaks: item.peaks, tint: .accentColor,
                                rest: item.state == .ready
                                    ? Color.accentColor.opacity(0.75) : Color.orange.opacity(0.8),
                                barWidth: 1.5, spacing: 1
                            )
                            .padding(.vertical, 4)
                        case .pause:
                            RoundedRectangle(cornerRadius: 5).fill(.quaternary)
                        case .rendering:
                            RoundedRectangle(cornerRadius: 5).fill(Color.accentColor.opacity(0.35))
                        default:
                            RoundedRectangle(cornerRadius: 5)
                                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .frame(width: itemWidth)
                    .help(item.state == .empty ? "Not rendered yet" : "")
                }
            }
            if project.isPlayingAll {
                TimelineView(.animation(minimumInterval: 1.0 / 30)) { _ in
                    let fraction =
                        lab.player.duration > 0 ? lab.player.currentTime / lab.player.duration : 0
                    Rectangle()
                        .fill(Color.primary)
                        .frame(width: 2)
                        .offset(x: width * fraction)
                }
            }
        }
        .frame(width: width, alignment: .leading)
        .contentShape(Rectangle())
        .onTapGesture { location in
            let fraction = min(max(location.x / max(width, 1), 0), 1)
            project.playAll(from: fraction * (project.assemble()?.take.duration ?? 0))
        }
    }
}
