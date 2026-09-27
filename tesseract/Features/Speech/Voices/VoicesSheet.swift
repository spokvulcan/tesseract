//
//  VoicesSheet.swift
//  tesseract
//
//  Choosing and designing voices. The grid is every voice the page offers;
//  with a VoiceDesign checkpoint it also opens the designer: trait chips
//  write a description, the voice is born reading a reference line, each
//  "Try Another Take" joins a strip of takes to compare, and any of them can
//  be kept. A CustomVoice checkpoint lists its own speakers and nothing more.
//

import SwiftUI
import TesseractSpeech

struct VoicesSheet: View {
    @Environment(VoiceLibrary.self) private var library
    @Environment(\.dismiss) private var dismiss
    @State private var path: [VoiceDesignRoute] = []

    var body: some View {
        NavigationStack(path: $path) {
            VoiceGrid { design in path.append(VoiceDesignRoute(design: design)) }
                .navigationTitle("Voices")
                .navigationDestination(for: VoiceDesignRoute.self) { route in
                    VoiceDesigner(design: route.design) { path.removeAll() }
                }
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
        .frame(minWidth: 640, idealWidth: 720, minHeight: 520, idealHeight: 620)
        .onAppear { library.refresh() }
    }
}

private struct VoiceDesignRoute: Hashable {
    let id = UUID()
    let design: VoiceDesign

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

// MARK: - Grid

private struct VoiceGrid: View {
    @Environment(VoiceLibrary.self) private var library
    @Environment(SettingsManager.self) private var settings
    let onDesign: (VoiceDesign) -> Void

    private let columns = [GridItem(.adaptive(minimum: 180, maximum: 260), spacing: 12)]

    var body: some View {
        @Bindable var settings = settings
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HStack {
                    Text("A voice is a sentence that describes it. Pick one, or design your own.")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Picker("Language", selection: $settings.ttsLanguage) {
                        ForEach(TTSLanguage.allCases) { language in
                            Text("\(language.flag) \(language.displayName)").tag(language.rawValue)
                        }
                    }
                    .labelsHidden()
                    .fixedSize()
                }
                if library.source.supportsVoiceDesign {
                    section("Your Voices") {
                        DesignVoiceCard { onDesign(VoiceDesign(language: settings.ttsLanguage)) }
                        ForEach(library.yourVoices) { voice in
                            VoiceCard(voice: voice, onDesign: onDesign)
                        }
                    }
                }
                section(library.source.supportsVoiceDesign ? "Built In" : "Voices") {
                    ForEach(library.builtIn) { voice in
                        VoiceCard(voice: voice, onDesign: onDesign)
                    }
                }
            }
            .padding(24)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content)
        -> some View
    {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.headline)
            LazyVGrid(columns: columns, spacing: 12, content: content)
        }
    }
}

private struct DesignVoiceCard: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: "plus")
                    .font(.system(size: 20, weight: .medium))
                Text("Design a Voice")
                    .font(.system(size: 13, weight: .semibold))
                Text("Describe who you want to hear")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, minHeight: 118)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
                    .foregroundStyle(.tertiary)
            )
            .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
    }
}

private struct VoiceCard: View {
    @Environment(VoiceLibrary.self) private var library
    @Environment(SpeechCoordinator.self) private var coordinator
    @Environment(SettingsManager.self) private var settings
    let voice: VoiceOption
    let onDesign: (VoiceDesign) -> Void

    @State private var isRenaming = false
    @State private var newName = ""

    var body: some View {
        let isCurrent = library.currentDescription == voice.description
        VStack(alignment: .leading, spacing: 10) {
            Voiceprint(seed: voice.description, isCurrent: isCurrent)
                .frame(height: 26)
            VStack(alignment: .leading, spacing: 2) {
                Text(voice.name)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                Text(voice.detail.isEmpty ? " " : voice.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            HStack {
                if isCurrent {
                    Label("In use", systemImage: "checkmark.circle.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.tint)
                }
                Spacer()
                Button {
                    library.select(voice)
                    coordinator.speakText(
                        VoiceDesign.referenceLine(for: settings.ttsLanguage), userInitiated: true)
                } label: {
                    Image(systemName: "play.fill")
                        .font(.system(size: 10, weight: .bold))
                        .frame(width: 24, height: 24)
                        .background(Circle().fill(.quaternary))
                }
                .buttonStyle(.plain)
                .help("Hear \(voice.name)")
                .accessibilityLabel("Hear \(voice.name)")
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(
                    isCurrent
                        ? AnyShapeStyle(Color.accentColor.opacity(0.1))
                        : AnyShapeStyle(.fill.quaternary))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(isCurrent ? Color.accentColor.opacity(0.7) : .clear, lineWidth: 1.5)
        )
        .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .onTapGesture { library.select(voice) }
        .help(voice.description)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isCurrent ? [.isButton, .isSelected] : .isButton)
        .contextMenu {
            Button("Use This Voice") { library.select(voice) }
            if library.source.supportsVoiceDesign {
                Button("Edit a Copy…") {
                    onDesign(
                        VoiceDesign(description: voice.description, language: settings.ttsLanguage))
                }
            }
            if voice.isEditable {
                Button("Rename…") {
                    newName = voice.name
                    isRenaming = true
                }
                Divider()
                Button("Delete", role: .destructive) { library.delete(voice) }
            }
        }
        .alert("Rename Voice", isPresented: $isRenaming) {
            TextField("Name", text: $newName)
            Button("Rename") { library.rename(voice, to: newName) }
            Button("Cancel", role: .cancel) {}
        }
    }
}

/// A voice's glyph: bars drawn from a hash of its description, distinct per
/// voice and the same every time.
private struct Voiceprint: View {
    let seed: String
    let isCurrent: Bool

    var body: some View {
        let bars = Self.bars(for: seed)
        Canvas { context, size in
            let step = size.width / CGFloat(bars.count)
            for (index, value) in bars.enumerated() {
                let height = max(3, CGFloat(value) * size.height)
                let rect = CGRect(
                    x: CGFloat(index) * step, y: (size.height - height) / 2, width: step * 0.6,
                    height: height)
                context.fill(
                    Path(roundedRect: rect, cornerRadius: step * 0.3),
                    with: .color(isCurrent ? .accentColor : .secondary.opacity(0.45)))
            }
        }
        .accessibilityHidden(true)
    }

    static func bars(for seed: String) -> [Float] {
        var state: UInt64 = 1_469_598_103_934_665_603
        for byte in seed.utf8 { state = (state ^ UInt64(byte)) &* 1_099_511_628_211 }
        var values: [Float] = []
        var phase = Double(state % 628) / 100
        for index in 0..<40 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let noise = Double(state >> 33) / Double(1 << 31)
            phase += 0.35 + noise * 0.3
            let envelope = sin(Double(index) / 39 * .pi)
            values.append(
                Float(0.2 + 0.8 * envelope * (0.55 + 0.45 * sin(phase)) * (0.7 + 0.3 * noise)))
        }
        return values
    }
}

// MARK: - Designer

private struct VoiceDesigner: View {
    @Environment(VoiceLibrary.self) private var library
    @Environment(SpeechCoordinator.self) private var coordinator
    @Environment(SettingsManager.self) private var settings
    @State var design: VoiceDesign
    let onDone: () -> Void

    @State private var takes: [VoiceTake] = []
    @State private var player = VoiceTakePlayer()
    @State private var name = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                traits
                descriptionField
                referenceField
                audition
                if !takes.isEmpty { takeStrip }
                saveRow
            }
            .padding(24)
        }
        .navigationTitle("Design a Voice")
        .onChange(of: settings.ttsLanguage) { _, language in design.setLanguage(language) }
        .onChange(of: coordinator.latestTake?.id) { _, _ in
            guard let take = coordinator.latestTake,
                take.voice.voiceDescription
                    == design.description.trimmingCharacters(in: .whitespacesAndNewlines)
            else { return }
            takes.insert(take, at: 0)
        }
        .onDisappear { player.stop() }
    }

    private var traits: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Traits").font(.headline)
                Spacer()
                Button {
                    design.shuffle()
                } label: {
                    Label("Shuffle", systemImage: "shuffle")
                }
                .help("Random traits")
            }
            ForEach(design.traits) { trait in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Text(trait.title)
                        .foregroundStyle(.secondary)
                        .frame(width: 56, alignment: .leading)
                    HStack(spacing: 6) {
                        ForEach(trait.options, id: \.self) { option in
                            TraitChip(
                                title: option.label, isOn: design.choice(for: trait) == option
                            ) {
                                design.toggle(option, for: trait)
                            }
                        }
                    }
                }
            }
        }
    }

    private var descriptionField: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Description").font(.headline)
            TextField(
                "Description",
                text: Binding(get: { design.description }, set: { design.setDescription($0) }),
                prompt: Text("Who is speaking, how they sound, how they speak."),
                axis: .vertical
            )
            .labelsHidden()
            .lineLimit(2...5)
            .textFieldStyle(.roundedBorder)
            Text(
                "Name concrete traits: gender, age, pitch, timbre, pace. Describe how the voice always sounds, not a performance: it reads everything this way. Write it in English or Chinese, and don't name real people."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    private var referenceField: some View {
        @Bindable var settings = settings
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Reference Line").font(.headline)
                Spacer()
                Picker("Language", selection: $settings.ttsLanguage) {
                    ForEach(TTSLanguage.allCases) { language in
                        Text("\(language.flag) \(language.displayName)").tag(language.rawValue)
                    }
                }
                .labelsHidden()
                .fixedSize()
                .help("The language this voice speaks")
            }
            TextField(
                "Reference line", text: $design.referenceLine,
                prompt: Text("The first thing this voice says"), axis: .vertical
            )
            .labelsHidden()
            .lineLimit(2...3)
            .textFieldStyle(.roundedBorder)
            Text(
                "The voice is born reading this line, and everything it reads later continues it: two plain, complete sentences in its language."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    private var audition: some View {
        let isDesigned = library.currentDescription == design.description
        return HStack(spacing: 12) {
            Button {
                settings.ttsVoiceDescription = design.description
                coordinator.tryAnotherTake(sampleFrom: design.referenceLine)
            } label: {
                Label(
                    takes.isEmpty || !isDesigned ? "Audition" : "Try Another Take",
                    systemImage: "waveform")
            }
            .buttonStyle(.borderedProminent)
            .disabled(design.isEmpty || coordinator.state.isActive)
            if coordinator.state.isActive {
                ProgressView().controlSize(.small)
                Text("Rendering…").foregroundStyle(.secondary)
            }
        }
    }

    private var takeStrip: some View {
        let kept = coordinator.pinnedVoice(
            description: design.description, language: settings.ttsLanguage)
        return VStack(alignment: .leading, spacing: 8) {
            Text("Takes").font(.headline)
            ForEach(Array(takes.enumerated()), id: \.element.id) { index, take in
                HStack(spacing: 10) {
                    Button {
                        coordinator.stop()
                        player.toggle(take)
                    } label: {
                        Image(
                            systemName: player.playingID == take.id
                                ? "stop.circle.fill" : "play.circle.fill"
                        )
                        .font(.system(size: 20))
                        .foregroundStyle(.tint)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(player.playingID == take.id ? "Stop take" : "Play take")
                    Text("Take \(takes.count - index)")
                        .monospacedDigit()
                    Text(String(format: "%.1f s", take.duration))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    Spacer()
                    if kept?.codeFrames == take.voice.codeFrames {
                        Label("In use", systemImage: "checkmark.circle.fill")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.tint)
                    } else {
                        Button("Use") {
                            player.stop()
                            Task {
                                await coordinator.keep(take.voice)
                                library.refresh()
                            }
                        }
                        .help("Make this take the voice from now on")
                    }
                }
            }
        }
    }

    private var saveRow: some View {
        HStack(spacing: 10) {
            TextField(
                "Name", text: $name,
                prompt: Text(design.isEmpty ? "Name this voice" : design.suggestedName)
            )
            .textFieldStyle(.roundedBorder)
            .frame(maxWidth: 240)
            Button("Save Voice") {
                let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
                library.save(
                    name: trimmed.isEmpty ? design.suggestedName : trimmed,
                    description: design.description)
                settings.ttsVoiceDescription = design.description
                library.refresh()
                onDone()
            }
            .disabled(design.isEmpty)
            Spacer()
        }
    }
}

private struct TraitChip: View {
    let title: String
    let isOn: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: isOn ? .semibold : .regular))
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(
                    Capsule().fill(
                        isOn
                            ? AnyShapeStyle(Color.accentColor.opacity(0.22))
                            : AnyShapeStyle(.fill.tertiary))
                )
                .overlay(Capsule().strokeBorder(isOn ? Color.accentColor : .clear, lineWidth: 1))
                .foregroundStyle(isOn ? Color.primary : Color.secondary)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}
