//
//  VoiceDesign.swift
//  tesseract
//
//  A voice being designed: trait choices that write its description, or a
//  description written by hand, plus the reference line it is born reading.
//  Pure; the Voices sheet edits it.
//
//  The traits and their words are the ones Qwen's VoiceDesign examples and
//  its benchmark use (docs/research/2026-09-27-qwen3-tts-voice-design.md):
//  who is speaking first (gender, age, pitch), then the timbre, a steady
//  baseline mood, the pace, and the accent. The description sits in front of
//  everything the voice reads, so it describes how the voice always sounds,
//  never a performance.
//

import Foundation

nonisolated struct VoiceDesign: Equatable, Sendable {
    /// One row of choices.
    struct Trait: Identifiable, Sendable {
        let id: String
        let title: String
        let options: [Option]
        /// The only language the trait applies to, when it has one (accent).
        var language: String?
    }

    /// A chip, and the words it puts in the description.
    struct Option: Hashable, Sendable {
        let label: String
        let phrase: String
    }

    static let traits: [Trait] = [
        Trait(
            id: "gender", title: "Voice",
            options: [
                Option(label: "Female", phrase: "Female"), Option(label: "Male", phrase: "Male"),
            ]),
        Trait(
            id: "age", title: "Age",
            options: [
                Option(label: "Teen", phrase: "teenage"),
                Option(label: "Young adult", phrase: "young adult"),
                Option(label: "Middle-aged", phrase: "middle-aged"),
                Option(label: "Senior", phrase: "senior"),
            ]),
        Trait(
            id: "pitch", title: "Pitch",
            options: [
                Option(label: "Low", phrase: "low pitch"),
                Option(label: "Medium", phrase: "medium pitch"),
                Option(label: "High", phrase: "high pitch"),
            ]),
        Trait(
            id: "texture", title: "Timbre",
            options: [
                Option(label: "Resonant", phrase: "deep, resonant"),
                Option(label: "Bright", phrase: "bright, clear"),
                Option(label: "Mellow", phrase: "warm, mellow"),
                Option(label: "Husky", phrase: "slightly husky"),
                Option(label: "Breathy", phrase: "soft, breathy"),
                Option(label: "Crisp", phrase: "crisp"),
            ]),
        Trait(
            id: "mood", title: "Mood",
            options: [
                Option(label: "Calm", phrase: "calm and steady"),
                Option(label: "Gentle", phrase: "gentle and soft-spoken"),
                // Qwen: ask for the mood and rule out what it can bring.
                Option(label: "Cheerful", phrase: "cheerful, without laughing"),
                Option(label: "Serious", phrase: "serious and composed"),
                Option(label: "Lively", phrase: "lively and engaged"),
            ]),
        Trait(
            id: "pace", title: "Pace",
            options: [
                Option(label: "Slow", phrase: "a slow, unhurried pace"),
                Option(label: "Moderate", phrase: "a moderate, even pace"),
                Option(label: "Brisk", phrase: "a brisk pace"),
            ]),
        Trait(
            id: "accent", title: "Accent",
            options: [
                Option(label: "American", phrase: "General American accent"),
                Option(label: "British", phrase: "British accent"),
            ],
            language: "English"),
    ]

    /// Chosen option per trait id.
    private(set) var choices: [String: Option] = [:]
    /// The description as last edited by hand, when it was.
    private var handWritten: String?
    /// The language the voice speaks (a `TTSLanguage` raw value): it picks
    /// the reference line and which traits apply.
    private(set) var language: String
    var referenceLine: String

    init(language: String = "English") {
        self.language = language
        self.referenceLine = Self.referenceLine(for: language)
    }

    /// Starts from an existing description, kept as written.
    init(description: String, language: String = "English") {
        self.init(language: language)
        handWritten = description.isEmpty ? nil : description
    }

    /// The traits that apply to the voice's language.
    var traits: [Trait] {
        Self.traits.filter { $0.language == nil || $0.language == language }
    }

    func choice(for trait: Trait) -> Option? { choices[trait.id] }

    /// Picks `option` for `trait`, or clears it if it was picked. Choosing a
    /// trait rewrites the description from the traits.
    mutating func toggle(_ option: Option, for trait: Trait) {
        choices[trait.id] = choices[trait.id] == option ? nil : option
        handWritten = nil
    }

    /// Editing the text by hand takes over from the traits.
    mutating func setDescription(_ text: String) {
        handWritten = text
        choices = [:]
    }

    /// Changes the voice's language: the reference line follows unless it
    /// was edited, and traits of other languages are dropped.
    mutating func setLanguage(_ language: String) {
        guard language != self.language else { return }
        if referenceLine == Self.referenceLine(for: self.language) {
            referenceLine = Self.referenceLine(for: language)
        }
        self.language = language
        for trait in Self.traits where trait.language.map({ $0 != language }) ?? false {
            choices[trait.id] = nil
        }
    }

    /// A new voice at random: always who is speaking and the timbre, which
    /// make voices distinct, sometimes a mood, a pace and an accent.
    mutating func shuffle() {
        choices = [:]
        let always: Set = ["gender", "age", "pitch", "texture"]
        for trait in traits where always.contains(trait.id) || Bool.random() {
            choices[trait.id] = trait.options.randomElement()
        }
        handWritten = nil
    }

    var isEmpty: Bool { description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    /// What the engine is given.
    var description: String { handWritten ?? composed }

    /// Qwen's compact shape, who first and then how they sound: "Female,
    /// middle-aged, low pitch. A slightly husky voice, calm and steady,
    /// speaking fluently at a slow, unhurried pace with clear articulation.
    /// General American accent."
    var composed: String {
        guard !choices.isEmpty else { return "" }
        var sentences: [String] = []
        let who = ["gender", "age", "pitch"].compactMap { choices[$0]?.phrase }
        if !who.isEmpty {
            let line = who.joined(separator: ", ")
            sentences.append(line.prefix(1).uppercased() + line.dropFirst() + ".")
        }
        var voice = choices["texture"].map { "A \($0.phrase) voice" } ?? "A voice"
        if let mood = choices["mood"] { voice += ", \(mood.phrase)," }
        voice += " speaking fluently"
        if let pace = choices["pace"] { voice += " at \(pace.phrase)" }
        sentences.append(voice + " with clear articulation.")
        if let accent = choices["accent"] { sentences.append(accent.phrase + ".") }
        return sentences.joined(separator: " ")
    }

    /// A name from the traits: "Husky older man", "Bright young woman".
    var suggestedName: String {
        guard handWritten == nil, !choices.isEmpty else { return Self.shortName(for: description) }
        let quality = (choices["texture"] ?? choices["mood"])?.label
        let age: String? =
            switch choices["age"]?.label {
            case "Teen"?: "teenage"
            case "Young adult"?: "young"
            case "Senior"?: "older"
            case let label?: label.lowercased()
            case nil: nil
            }
        let person =
            switch choices["gender"]?.label {
            case "Female"?: "woman"
            case "Male"?: "man"
            default: "voice"
            }
        let words = [quality, age, person].compactMap { $0 }.joined(separator: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }

    /// A few words to name a description by. A designed one's opening
    /// ("Female, middle-aged, low pitch.") names it; otherwise the first
    /// words: "A warm, raspy older…" → "Warm, raspy older".
    static func shortName(for description: String) -> String {
        if let opening = description.split(separator: ".", maxSplits: 1).first,
            opening.count <= 40, opening.split(separator: ",").count >= 2,
            !opening.lowercased().hasPrefix("a ")
        {
            return opening.trimmingCharacters(in: .whitespaces)
        }
        var words = description.split(whereSeparator: \.isWhitespace).map(String.init)
        if let first = words.first, ["a", "an", "the"].contains(first.lowercased()) {
            words.removeFirst()
        }
        let head = words.prefix(3).joined(separator: " ")
            .trimmingCharacters(in: CharacterSet(charactersIn: ",.;:"))
        guard let first = head.first else { return "Custom voice" }
        return first.uppercased() + head.dropFirst()
    }

    /// What a description names that tells it from its neighbours, for a
    /// voice card: "Female · British accent · slow". Empty when it names
    /// none of it.
    static func summary(of description: String) -> String {
        let words = description.lowercased().split { !$0.isLetter && $0 != "-" }.map(String.init)
        let named = Set(words)
        var facts: [String] = []
        if named.contains("female") || named.contains("woman") {
            facts.append("female")
        } else if named.contains("male") || named.contains("man") {
            facts.append("male")
        }
        if let index = words.firstIndex(of: "accent"), index > 0 {
            facts.append(words[index - 1].capitalized + " accent")
        }
        if named.contains("slow") || named.contains("unhurried") {
            facts.append("slow")
        } else if named.contains("brisk") || named.contains("quick") || named.contains("fast") {
            facts.append("brisk")
        }
        let line = facts.joined(separator: " · ")
        return line.prefix(1).uppercased() + line.dropFirst()
    }

    // MARK: - Reference lines

    /// What a new voice first reads, per language: two plain, complete
    /// sentences, about ten seconds at an even pace. No lists, numbers or
    /// headings, and it ends on a finished sentence: every later passage
    /// continues this take (ADR-0072).
    static func referenceLine(for language: String) -> String {
        referenceLines[language] ?? referenceLines["English"] ?? ""
    }

    private static let referenceLines: [String: String] = [
        "English":
            "This is how I sound when I read aloud to you. From the first page to the last, every line you give me will keep this same voice.",
        "Chinese": "这就是我为你朗读时的声音。从第一页到最后一页，你交给我的每一行文字，都会保持同样的声音。",
        "Japanese": "これが、あなたに本を読み聞かせるときの私の声です。最初のページから最後のページまで、どの行も同じ声のままお届けします。",
        "Korean": "이것이 제가 소리 내어 읽어 드릴 때의 목소리입니다. 첫 페이지부터 마지막 페이지까지, 모든 문장을 같은 목소리로 읽어 드릴게요.",
        "German":
            "So klinge ich, wenn ich dir vorlese. Von der ersten bis zur letzten Seite behält jede Zeile, die du mir gibst, dieselbe Stimme.",
        "French":
            "Voici ma voix quand je vous lis un texte à voix haute. De la première à la dernière page, chaque ligne que vous me confiez gardera cette même voix.",
        "Russian":
            "Вот так звучит мой голос, когда я читаю вам вслух. С первой страницы до последней каждая строка, которую вы мне дадите, прозвучит этим же голосом.",
        "Portuguese":
            "É assim que eu soo quando leio em voz alta para você. Da primeira à última página, cada linha que você me der terá esta mesma voz.",
        "Spanish":
            "Así suena mi voz cuando te leo en voz alta. De la primera a la última página, cada línea que me des conservará esta misma voz.",
        "Italian":
            "Ecco come suona la mia voce quando ti leggo ad alta voce. Dalla prima all'ultima pagina, ogni riga che mi darai avrà questa stessa voce.",
    ]
}
