//
//  DictationVariantSayItAgain.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  E · Say It Again. Hands stay off the keyboard. Right after a take, hold
//  ⌃⌥Space and say the fix: the right word ("Claude"), its letters
//  ("C L A U D E"), "cloud to Claude", or the whole sentence again. The
//  app finds the words it replaces, fixes them where they landed, and
//  learns. The page is the last take, large, with what the last fix did.
//

import SwiftUI

@MainActor
enum SayItAgainFlow {
    static func showHint(_ lab: DictationLab, take: LabTake) {
        lab.panels.showCard(
            SayHintCard(take: take), size: CGSize(width: 500, height: 64),
            duration: take.applied.isEmpty ? .seconds(3) : .seconds(5))
    }

    static func showListening(_ lab: DictationLab) {
        guard let voice = lab.voice, let feed = lab.feed else { return }
        lab.panels.showCard(
            ListeningCard(voice: voice, feed: feed), size: CGSize(width: 420, height: 64),
            duration: nil)
    }
}

struct SayHintCard: View {
    let take: LabTake
    var body: some View {
        HStack(spacing: 10) {
            if take.applied.isEmpty {
                Image(systemName: "checkmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(.green)
                Text("Inserted")
                    .font(.system(size: 12, weight: .semibold))
            } else {
                ForEach(Array(take.applied.prefix(2).enumerated()), id: \.offset) { _, item in
                    HeardMeant(heard: item.heard, meant: item.term, size: 12)
                }
            }
            Spacer(minLength: 8)
            Image(systemName: "mic")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
            Text("Wrong word? Hold \(LabStyle.fixShortcut) and say it")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .frame(height: 40)
        .glassEffect(.regular, in: .capsule)
        .padding(8)
    }
}

struct ListeningCard: View {
    let voice: LabVoiceFix
    let feed: DictationFeed

    var body: some View {
        HStack(spacing: 10) {
            switch voice.state {
            case .listening:
                AudioBarsView(feed: feed)
                    .frame(width: 60, height: 18)
                Text("Say the right words")
                    .font(.system(size: 12, weight: .semibold))
            case .resolving:
                ProgressView().controlSize(.small)
                Text("Fixing…")
                    .font(.system(size: 12, weight: .semibold))
            case .failed(let message):
                Image(systemName: "questionmark.circle")
                    .foregroundStyle(.orange)
                Text(message)
                    .font(.system(size: 12, weight: .semibold))
                    .lineLimit(2)
            case .idle:
                EmptyView()
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .frame(height: 40)
        .glassEffect(.regular, in: .capsule)
        .padding(8)
    }
}

// MARK: - Page

struct SayItAgainPage: View {
    let lab: DictationLab
    @Environment(TranscriptionHistory.self) private var history

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                lastTakeCard
                VStack(alignment: .leading, spacing: 10) {
                    Text("To fix the last take, hold \(LabStyle.fixShortcut) and say")
                        .font(.system(size: 15, weight: .semibold))
                    ForEach(Self.phrases, id: \.0) { example, effect in
                        HStack(alignment: .firstTextBaseline, spacing: 12) {
                            Text("“\(example)”")
                                .font(.system(size: 15, weight: .medium))
                                .frame(width: 200, alignment: .leading)
                            Text(effect)
                                .font(.system(size: 15))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                if !lab.lexicon.terms.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Words it learned")
                            .font(.system(size: 15, weight: .semibold))
                        WordFlow(spacing: 14, lineSpacing: 8) {
                            ForEach(lab.lexicon.terms) { term in
                                LearnedWord(term.term, size: 15)
                                    .contextMenu {
                                        Button("Forget \(term.term)", role: .destructive) {
                                            lab.forget(term.id)
                                        }
                                    }
                            }
                        }
                    }
                }
            }
            .frame(maxWidth: LabStyle.column, alignment: .leading)
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: .infinity)
        }
        .safeAreaInset(edge: .bottom) {
            HStack {
                LabStatusLine(extra: "hold \(LabStyle.fixShortcut) and say the fix")
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            .background(.bar)
        }
    }

    static let phrases: [(String, String)] = [
        ("Claude", "replaces the word that sounds like it"),
        ("C L A U D E", "spells a name letter by letter"),
        ("cloud to Claude", "says exactly what to change"),
        ("the whole sentence again", "replaces the take"),
    ]

    @ViewBuilder
    private var lastTakeCard: some View {
        let lesson = lab.lessons.first
        VStack(alignment: .leading, spacing: 12) {
            if let take = lab.lastTake {
                Text("Last take, in \(take.appName)")
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
                Text(labAttributed(take.text, applied: take.applied, size: 22))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                if let lesson, lesson.learned, !lesson.undone {
                    HStack(spacing: 10) {
                        HeardMeant(heard: lesson.hunk.before, meant: lesson.hunk.after, size: 15)
                        Text("learned").font(.system(size: 15)).foregroundStyle(.secondary)
                        Button("Undo") { lab.undo([lesson.id]) }
                            .buttonStyle(.link)
                    }
                }
            } else if let entry = history.entries.first {
                Text("Last take")
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
                Text(entry.text)
                    .font(.system(size: 22))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Say something with the dictation shortcut. It shows up here, large.")
                    .font(.system(size: 22))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: .rect(cornerRadius: 16))
    }
}
