//
//  SpeechVariantVoiceLab.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Speech page redesign, 2026-09-27). Never merge.
//
//  C · Voice Lab — voices first (OpenAI.fm's voice grid and vibes,
//  ElevenLabs and Hume voice design, WellSaid's take history). The engine's
//  real strength is that a voice is a sentence you write, so the page is
//  built around casting: a grid of voices, a designer whose trait chips
//  write the description, a reference line the voice is born reading, and
//  a strip of takes you can bring back. The script is a dock at the bottom.
//  Signature: every voice gets a voiceprint — generated from its
//  description, replaced by its real waveform once you have heard it.
//

import SwiftUI

struct VoiceLabVariant: View {
    @State private var design = VoiceDesign()
    @State private var isDesigning = false

    var body: some View {
        let lab = SpeechLab.shared
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                VoiceLabHeader(isDesigning: $isDesigning, design: $design)
                VoiceGrid(isDesigning: $isDesigning, design: $design)
                if isDesigning {
                    VoiceDesigner(design: $design, isDesigning: $isDesigning)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                } else {
                    CurrentVoicePanel(isDesigning: $isDesigning, design: $design)
                }
            }
            .frame(maxWidth: 900, alignment: .leading)
            .padding(.horizontal, 28)
            .padding(.vertical, 24)
            .frame(maxWidth: .infinity)
            .animation(.smooth(duration: 0.3), value: isDesigning)
        }
        .safeAreaInset(edge: .top, spacing: 0) { SpeechEngineNotice() }
        .safeAreaInset(edge: .bottom, spacing: 0) { ScriptDock() }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                OverlayToolbarButton()
            }
        }
        .onAppear { design = VoiceDesign(description: lab.currentVoiceDescription) }
    }
}

// MARK: - Header

private struct VoiceLabHeader: View {
    @Binding var isDesigning: Bool
    @Binding var design: VoiceDesign

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Voices")
                    .font(.system(size: 22, weight: .semibold))
                Text("A voice here is a sentence you write. Pick one, or describe a new one.")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            LanguagePicker()
                .labelsHidden()
                .fixedSize()
            Button {
                design = VoiceDesign()
                isDesigning = true
            } label: {
                Label("New Voice", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
        }
    }
}

// MARK: - Grid

private struct VoiceGrid: View {
    @Binding var isDesigning: Bool
    @Binding var design: VoiceDesign

    var body: some View {
        let lab = SpeechLab.shared
        VStack(alignment: .leading, spacing: 18) {
            if !lab.yourVoices.isEmpty {
                section("Your voices", voices: lab.yourVoices)
            }
            section("Built in", voices: SpeechLab.presets)
        }
    }

    private func section(_ title: String, voices: [LabVoice]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.system(size: 11, weight: .semibold))
                .kerning(0.6)
                .foregroundStyle(.tertiary)
            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 176, maximum: 260), spacing: 12)],
                spacing: 12
            ) {
                ForEach(voices) { voice in
                    VoiceCard(voice: voice) {
                        design = VoiceDesign(description: voice.description)
                        isDesigning = true
                    }
                }
            }
        }
    }
}

private struct VoiceCard: View {
    let voice: LabVoice
    let onEdit: () -> Void
    @State private var hovering = false

    var body: some View {
        let lab = SpeechLab.shared
        let isCurrent = lab.currentVoiceDescription == voice.description
        let heard = lab.takes.first { $0.voiceDescription == voice.description }
        VStack(alignment: .leading, spacing: 10) {
            Voiceprint(seed: voice.description, peaks: heard?.audio.peaks, isCurrent: isCurrent)
                .frame(height: 30)
            VStack(alignment: .leading, spacing: 3) {
                Text(voice.name)
                    .font(.system(size: 14, weight: .semibold))
                    .lineLimit(1)
                Text(voice.blurb)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .lineLimit(2, reservesSpace: true)
            }
            HStack(spacing: 8) {
                if isCurrent {
                    Label("In use", systemImage: "checkmark.circle.fill")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.tint)
                } else {
                    Text(heard == nil ? "Not heard yet" : "Heard")
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                if hovering {
                    Button("Edit", action: onEdit)
                        .buttonStyle(.borderless)
                        .font(.system(size: 11))
                }
                Button {
                    lab.select(voice)
                    lab.speak(VoiceDesign.defaultReferenceLine)
                } label: {
                    Image(systemName: "play.fill")
                        .font(.system(size: 10, weight: .bold))
                        .frame(width: 24, height: 24)
                        .background(
                            Circle().fill(
                                isCurrent ? Color.accentColor : Color.secondary.opacity(0.25))
                        )
                        .foregroundStyle(isCurrent ? .white : .primary)
                }
                .buttonStyle(.plain)
                .help("Hear this voice")
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(
                    isCurrent
                        ? AnyShapeStyle(Color.accentColor.opacity(0.1))
                        : AnyShapeStyle(.fill.quaternary))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(isCurrent ? Color.accentColor.opacity(0.8) : .clear, lineWidth: 1.5)
        )
        .contentShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .onTapGesture { lab.select(voice) }
        .onHover { hovering = $0 }
        .contextMenu {
            Button("Use This Voice") { lab.select(voice) }
            Button("Edit a Copy…", action: onEdit)
            Button("Copy Description") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(voice.description, forType: .string)
            }
        }
        .help(voice.description)
    }
}

/// A voice's glyph: its real waveform once heard, until then bars drawn
/// from a hash of its description — distinct per voice, stable across runs.
private struct Voiceprint: View {
    let seed: String
    let peaks: [Float]?
    let isCurrent: Bool

    var body: some View {
        WaveformBars(
            peaks: peaks.map { Array($0.prefix(80)) } ?? Self.synthetic(seed),
            progress: isCurrent ? 1 : 0,
            tint: .accentColor,
            rest: .secondary.opacity(0.45),
            barWidth: 3, spacing: 2, minimumBar: 3)
    }

    static func synthetic(_ seed: String) -> [Float] {
        var state: UInt64 = 1469598103934665603
        for byte in seed.utf8 { state = (state ^ UInt64(byte)) &* 1099511628211 }
        var values: [Float] = []
        var phase = Double(state % 628) / 100
        for i in 0..<48 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let noise = Double(state >> 33) / Double(1 << 31)
            phase += 0.35 + noise * 0.3
            let envelope = sin(Double(i) / 47 * .pi)
            values.append(
                Float(0.2 + 0.8 * envelope * (0.55 + 0.45 * sin(phase)) * (0.7 + 0.3 * noise)))
        }
        return values
    }
}

// MARK: - Current voice

private struct CurrentVoicePanel: View {
    @Binding var isDesigning: Bool
    @Binding var design: VoiceDesign

    var body: some View {
        let lab = SpeechLab.shared
        let description = lab.currentVoiceDescription
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("IN USE")
                        .font(.system(size: 11, weight: .semibold))
                        .kerning(0.6)
                        .foregroundStyle(.tertiary)
                    Text(lab.currentVoiceName)
                        .font(.system(size: 18, weight: .semibold))
                }
                Spacer()
                Button("Edit a Copy…") {
                    design = VoiceDesign(description: description)
                    isDesigning = true
                }
                Button("Try Another Take") {
                    lab.tryAnotherTake(sample: VoiceDesign.defaultReferenceLine)
                }
                .help(
                    "Render this voice again. The take you hear last is kept; earlier ones stay below."
                )
            }
            Text(
                description.isEmpty
                    ? "The model's default voice, with no description." : description
            )
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
            VoiceTakeStrip(description: description)
        }
        .padding(18)
        .background(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(.fill.quinary))
    }
}

// MARK: - Designer

struct VoiceDesign: Equatable {
    var gender: String?
    var age: String?
    var energy: String?
    var texture: String?
    var pace: String?
    var accent: String?
    var text: String = ""
    var name: String = ""
    var referenceLine: String = VoiceDesign.defaultReferenceLine
    /// True once the text was edited by hand — chips no longer rewrite it.
    var isCustom = false

    init() {}

    init(description: String) {
        text = description
        isCustom = !description.isEmpty
        name = description.isEmpty ? "" : SpeechLab.shortName(description) + " copy"
    }

    static let defaultReferenceLine =
        "Hello there. This is how I sound when I read to you: a story, an article, a long email. Every line keeps this same voice."

    static let traits:
        [(title: String, key: WritableKeyPath<VoiceDesign, String?>, options: [String])] = [
            ("Voice", \.gender, ["female", "male", "androgynous"]),
            ("Age", \.age, ["young", "middle-aged", "older"]),
            ("Mood", \.energy, ["calm", "warm", "bright", "intense"]),
            ("Texture", \.texture, ["smooth", "breathy", "raspy", "crisp"]),
            ("Pace", \.pace, ["slow", "natural", "brisk"]),
            ("Accent", \.accent, ["American", "British", "Australian", "Irish"]),
        ]

    /// "A warm, breathy older female voice with a British accent and a slow, unhurried pace."
    var composed: String {
        let adjectives = [energy, texture].compactMap { $0 }
        let who = [age, gender].compactMap { $0 }.joined(separator: " ")
        var sentence = "A "
        if !adjectives.isEmpty { sentence += adjectives.joined(separator: ", ") + " " }
        sentence += who.isEmpty ? "voice" : "\(who) voice"
        var tails: [String] = []
        if let accent { tails.append("a\(accent == "American" ? "n" : "") \(accent) accent") }
        if let pace {
            tails.append(
                pace == "slow"
                    ? "a slow, unhurried pace"
                    : pace == "brisk" ? "a brisk, lively pace" : "a natural pace")
        }
        if !tails.isEmpty { sentence += " with " + tails.joined(separator: " and ") }
        return sentence + "."
    }

    var description: String { isCustom ? text : (hasTraits ? composed : text) }

    var hasTraits: Bool { [gender, age, energy, texture, pace, accent].contains { $0 != nil } }

    mutating func shuffle() {
        for trait in Self.traits {
            self[keyPath: trait.key] =
                Bool.random() || trait.title == "Voice" ? trait.options.randomElement() : nil
        }
        isCustom = false
        text = composed
    }
}

private struct VoiceDesigner: View {
    @Binding var design: VoiceDesign
    @Binding var isDesigning: Bool

    var body: some View {
        let lab = SpeechLab.shared
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("DESIGN A VOICE")
                    .font(.system(size: 11, weight: .semibold))
                    .kerning(0.6)
                    .foregroundStyle(.tertiary)
                Spacer()
                Button {
                    design.shuffle()
                } label: {
                    Label("Shuffle", systemImage: "shuffle")
                }
                .help("Random traits")
                Button("Close") { isDesigning = false }
            }

            VStack(alignment: .leading, spacing: 8) {
                ForEach(VoiceDesign.traits, id: \.title) { trait in
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(trait.title)
                            .foregroundStyle(.secondary)
                            .frame(width: 58, alignment: .leading)
                        FlowLayout(spacing: 6) {
                            ForEach(trait.options, id: \.self) { option in
                                TraitChip(
                                    title: option.prefix(1).uppercased() + option.dropFirst(),
                                    isOn: design[keyPath: trait.key] == option
                                ) {
                                    design[keyPath: trait.key] =
                                        design[keyPath: trait.key] == option ? nil : option
                                    design.isCustom = false
                                    design.text = design.composed
                                }
                            }
                        }
                    }
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Description")
                    .foregroundStyle(.secondary)
                TextField(
                    "Description",
                    text: Binding(
                        get: { design.description },
                        set: {
                            design.text = $0
                            design.isCustom = true
                        }),
                    prompt: Text("Who is speaking: timbre, age, mood, pace. One or two sentences."),
                    axis: .vertical
                )
                .labelsHidden()
                .lineLimit(2...4)
                .textFieldStyle(.roundedBorder)
                Text("Keep it to one or two sentences: long descriptions make a voice drift.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Reference line")
                    .foregroundStyle(.secondary)
                TextField(
                    "Reference line", text: $design.referenceLine,
                    prompt: Text("The first thing this voice says"), axis: .vertical
                )
                .labelsHidden()
                .lineLimit(2...3)
                .textFieldStyle(.roundedBorder)
                Text(
                    "The voice is born reading this line, and every later passage continues it. Match the line to the voice."
                )
                .font(.caption)
                .foregroundStyle(.tertiary)
            }

            HStack(spacing: 10) {
                Button {
                    lab.setVoiceDescription(design.description)
                    lab.tryAnotherTake(sample: design.referenceLine)
                } label: {
                    Label(
                        lab.currentVoiceDescription == design.description
                            ? "Try Another Take" : "Audition",
                        systemImage: "waveform")
                }
                .buttonStyle(.borderedProminent)
                .disabled(design.description.trimmingCharacters(in: .whitespaces).isEmpty)
                TextField("Name", text: $design.name, prompt: Text("Name this voice"))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 180)
                Button("Save Voice") {
                    lab.setVoiceDescription(design.description)
                    lab.nameVoice(design.description, name: design.name)
                    isDesigning = false
                }
                .disabled(design.description.trimmingCharacters(in: .whitespaces).isEmpty)
                Spacer()
            }

            if lab.currentVoiceDescription == design.description {
                VoiceTakeStrip(description: design.description)
            }
        }
        .padding(18)
        .background(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(.fill.quinary))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.accentColor.opacity(0.35), lineWidth: 1)
        )
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
    }
}

/// Chips that wrap onto new lines.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        var maxX: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width {
                x = 0
                y += lineHeight + spacing
                lineHeight = 0
            }
            x += size.width + spacing
            maxX = max(maxX, x - spacing)
            lineHeight = max(lineHeight, size.height)
        }
        return CGSize(width: min(maxX, width), height: y + lineHeight)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        var x = bounds.minX
        var y = bounds.minY
        var lineHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                x = bounds.minX
                y += lineHeight + spacing
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

// MARK: - Script dock

private struct ScriptDock: View {
    var body: some View {
        @Bindable var lab = SpeechLab.shared
        let status = lab.status
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: "text.quote")
                .foregroundStyle(.secondary)
            TextField(
                "Script", text: $lab.draft, prompt: Text("Type a line to hear it in this voice"),
                axis: .vertical
            )
            .labelsHidden()
            .textFieldStyle(.plain)
            .lineLimit(1...3)
            SpeechStatusLine()
                .lineLimit(1)
            if status.isActive {
                Button(role: .destructive) {
                    lab.stop()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(.glassProminent)
                .tint(.red)
                .keyboardShortcut(.escape, modifiers: [])
            } else {
                Button {
                    lab.speak(lab.draft)
                } label: {
                    Label("Read", systemImage: "play.fill")
                }
                .buttonStyle(.glassProminent)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(lab.draftIsEmpty)
            }
        }
        .controlSize(.large)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .glassEffect(.regular, in: .rect(cornerRadius: 20))
        .frame(maxWidth: 900)
        .padding(.horizontal, 24)
        .padding(.bottom, 14)
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Overlay button (shared by the non-inspector variants)

struct OverlayToolbarButton: View {
    @State private var isPresented = false

    var body: some View {
        Button {
            isPresented.toggle()
        } label: {
            Label("Overlay", systemImage: SpeechLab.shared.overlay.style.symbol)
        }
        .help("How speech shows on screen outside the app")
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            Form {
                Section {
                    OverlaySettingsControls()
                } header: {
                    Text("Overlay")
                } footer: {
                    Text(
                        "Shows the words as they are read, on top of every app. Changes preview on screen."
                    )
                }
            }
            .formStyle(.grouped)
            .frame(width: 340)
            .frame(minHeight: 300)
        }
    }
}
