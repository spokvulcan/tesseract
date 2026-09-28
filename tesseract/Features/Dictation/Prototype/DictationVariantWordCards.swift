//
//  DictationVariantWordCards.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  B · Word Cards. Mouse first. When the words land, a card shows them for a
//  few seconds (longer while the pointer is on it). Click the word that's
//  wrong, click the right one: fixed where it landed, and learned. The page
//  is a reading column of what you said, every word clickable the same way,
//  with what the app learned along the top.
//

import SwiftUI

@MainActor
enum WordCardsFlow {
    static func showCard(_ lab: DictationLab, take: LabTake?, linger: Duration) {
        guard let take else {
            lab.showToast(LabToast(title: "Nothing to fix yet", detail: "Dictate something first"))
            return
        }
        let lines = max(1, min(4, take.text.count / 70 + 1))
        lab.panels.showCard(
            WordCard(lab: lab, take: take),
            size: CGSize(width: 640, height: CGFloat(96 + lines * 24)),
            duration: linger)
    }
}

/// The overlay card: the take's words, each one a button.
struct WordCard: View {
    let lab: DictationLab
    let take: LabTake
    @State private var picked: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ClickableWords(text: take.text, applied: take.applied, picked: $picked)
            if let picked {
                Alternatives(
                    lab: lab, word: word(at: picked),
                    choose: { choice in apply(choice, at: picked) },
                    type: {
                        lab.panels.closeCard()
                        FixBarFlow.open(lab, focusWord: picked)
                    },
                    keep: { self.picked = nil })
            } else {
                Text("Click a word that's wrong")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(16)
        .frame(width: 620, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 18))
        .padding(10)
        .onHover { inside in
            if inside { lab.panels.holdCard() } else { lab.panels.extendCard(by: .seconds(3)) }
        }
    }

    private func word(at index: Int) -> String {
        let words = take.text.split(separator: " ").map(String.init)
        return index < words.count ? words[index] : ""
    }

    private func apply(_ choice: String, at index: Int) {
        var words = take.text.split(separator: " ").map(String.init)
        guard index < words.count else { return }
        let trailing = String(
            words[index].reversed().prefix(while: { $0.isPunctuation }).reversed())
        words[index] = choice + trailing
        let corrected = words.joined(separator: " ")
        lab.panels.closeCard()
        Task { await lab.fix(take.id, to: corrected, source: .overlay, allowedKeys: 0) }
    }
}

/// Words that wrap like text; hover lifts one, a click picks it.
struct ClickableWords: View {
    let text: String
    var applied: [LabApplied] = []
    @Binding var picked: Int?
    var size: CGFloat = 15

    var body: some View {
        let words = text.split(separator: " ").map(String.init)
        let learned = Set(applied.map(\.term))
        WordFlow(spacing: 0, lineSpacing: 4) {
            ForEach(Array(words.enumerated()), id: \.offset) { index, word in
                WordButton(
                    word: word,
                    isLearned: learned.contains(
                        word.trimmingCharacters(in: .punctuationCharacters)),
                    isPicked: picked == index, size: size
                ) {
                    picked = picked == index ? nil : index
                }
            }
        }
    }
}

struct WordButton: View {
    let word: String
    let isLearned: Bool
    let isPicked: Bool
    let size: CGFloat
    let action: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            Text(word)
                .font(.system(size: size, weight: isLearned ? .medium : .regular))
                .underline(isLearned, color: .accentColor)
                .padding(.horizontal, 2)
                .padding(.vertical, 1)
                .background(
                    RoundedRectangle(cornerRadius: 5)
                        .fill(
                            isPicked
                                ? Color.accentColor.opacity(0.25)
                                : (hover ? Color.primary.opacity(0.08) : .clear))
                )
        }
        .overlayAffordance()
        .onHover { hover = $0 }
    }
}

/// The choices for a picked word: known words that sound like it, typing,
/// or leaving it.
struct Alternatives: View {
    let lab: DictationLab
    let word: String
    let choose: (String) -> Void
    let type: () -> Void
    let keep: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Text(word.trimmingCharacters(in: .punctuationCharacters))
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.secondary)
                .strikethrough(true, color: .secondary.opacity(0.6))
            Image(systemName: "arrow.right")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.tertiary)
            ForEach(lab.suggestions(for: word), id: \.self) { option in
                Button(option) { choose(option) }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .focusable(false)
            }
            Button("Type…", action: type)
                .buttonStyle(.bordered)
                .controlSize(.small)
                .focusable(false)
            Button("Keep", action: keep)
                .overlayAffordance()
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Page

struct WordCardsPage: View {
    let lab: DictationLab
    @Environment(TranscriptionHistory.self) private var history
    @State private var picked: (entry: UUID, index: Int)?
    @State private var typing: UUID?
    @State private var draft = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                learnedStrip
                ForEach(history.entries.prefix(60)) { entry in
                    paragraph(entry)
                }
                if history.entries.isEmpty {
                    Text(
                        "Dictate anywhere. Click any word that came out wrong, here or in the card that appears after you speak."
                    )
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: LabStyle.column, alignment: .leading)
            .padding(.horizontal, 32)
            .padding(.vertical, 24)
            .frame(maxWidth: .infinity)
        }
        .safeAreaInset(edge: .bottom) {
            HStack {
                LabStatusLine(extra: "click a word to fix it")
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            .background(.bar)
        }
    }

    @ViewBuilder
    private var learnedStrip: some View {
        let pairs = lab.lexicon.terms.flatMap { term in
            term.heardAs.isEmpty ? [("", term.term)] : term.heardAs.map { ($0, term.term) }
        }
        if !pairs.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Learned")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.secondary)
                WordFlow(spacing: 16, lineSpacing: 8) {
                    ForEach(Array(pairs.enumerated()), id: \.offset) { _, pair in
                        if pair.0.isEmpty {
                            LearnedWord(pair.1, size: 15)
                        } else {
                            HeardMeant(heard: pair.0, meant: pair.1, size: 15)
                        }
                    }
                }
            }
            .padding(.bottom, 8)
        }
    }

    @ViewBuilder
    private func paragraph(_ entry: TranscriptionEntry) -> some View {
        let take = lab.takes.first { $0.pairID != nil && $0.pairID == entry.pairID }
        let text = take?.text ?? entry.text
        VStack(alignment: .leading, spacing: 6) {
            Text(TranscriptionHistory.formattedTime(for: entry.timestamp))
                .font(.system(size: 15))
                .foregroundStyle(.tertiary)
                .monospacedDigit()
            if typing == entry.id {
                TextField("", text: $draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 15))
                    .onSubmit {
                        commit(entry, take: take, original: text, corrected: draft)
                        typing = nil
                    }
                    .onKeyPress(.escape) {
                        typing = nil
                        return .handled
                    }
            } else {
                ClickableWords(
                    text: text, applied: take?.applied ?? [],
                    picked: Binding(
                        get: { picked?.entry == entry.id ? picked?.index : nil },
                        set: { index in picked = index.map { (entry.id, $0) } }))
                if let p = picked, p.entry == entry.id {
                    pageAlternatives(entry: entry, take: take, text: text, index: p.index)
                }
            }
        }
    }

    private func pageAlternatives(
        entry: TranscriptionEntry, take: LabTake?, text: String, index: Int
    ) -> some View {
        let words = text.split(separator: " ").map(String.init)
        let word = index < words.count ? words[index] : ""
        return Alternatives(
            lab: lab, word: word,
            choose: { choice in
                var w = words
                let trailing = String(
                    w[index].reversed().prefix(while: { $0.isPunctuation }).reversed())
                w[index] = choice + trailing
                commit(entry, take: take, original: text, corrected: w.joined(separator: " "))
                picked = nil
            },
            type: {
                draft = text
                typing = entry.id
                picked = nil
            },
            keep: { picked = nil })
    }

    private func commit(
        _ entry: TranscriptionEntry, take: LabTake?, original: String, corrected: String
    ) {
        let corrected = corrected.replacingOccurrences(of: "\n", with: " ")
        if let take {
            Task {
                await lab.fix(
                    take.id, to: corrected, source: .page, putBack: take.id == lab.lastTake?.id)
            }
        } else {
            lab.learn(before: original, after: corrected, source: .page, appName: "an earlier take")
        }
    }
}
