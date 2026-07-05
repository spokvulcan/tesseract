//
//  AgentChapter.swift
//  tesseract
//
//  Chapter 2 — the headline. A scripted agent exchange plays out token by
//  token (the LLM is the download queue's long tail, so this chapter stays
//  scripted on a first run; the styling mirrors the real chat surface).
//

import SwiftUI

struct AgentChapter: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var revealedCharacters: Int = 0
    @State private var showsToolChip = false
    @State private var playback: Task<Void, Never>?

    private static let userLine = "What was that idea I had about the reading list?"
    private static let toolLine = "memory.search — \u{201C}reading list\u{201D}"
    private static let assistantLine =
        "Found it — three weeks ago you wanted to sort the list by how long each "
        + "book has waited, oldest first. Want me to draft that view now?"

    var body: some View {
        ChapterScaffold(
            kicker: "Chapter 2 · The Agent",
            title: "An agent that can actually know you",
            subtitle: "It remembers your goals, searches your notes, and uses tools — "
                + "and because it runs here, trusting it costs nothing."
        ) {
            StagePanel {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Spacer(minLength: 60)
                        Text(Self.userLine)
                            .font(.system(size: 12.5))
                            .foregroundStyle(.primary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .fill(.tint.opacity(0.22))
                            )
                    }

                    if showsToolChip {
                        HStack(spacing: 6) {
                            Image(systemName: "wrench.and.screwdriver.fill")
                                .font(.system(size: 9))
                            Text(Self.toolLine)
                                .font(.system(size: 10.5, design: .monospaced))
                        }
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(
                            Capsule(style: .continuous).fill(.quaternary.opacity(0.6))
                        )
                        .transition(.scale(scale: 0.9).combined(with: .opacity))
                    }

                    if revealedCharacters > 0 {
                        HStack {
                            Text(String(Self.assistantLine.prefix(revealedCharacters)))
                                .font(.system(size: 12.5))
                                .lineSpacing(3)
                                .foregroundStyle(.primary)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .frame(maxWidth: 380, alignment: .leading)
                                .background(
                                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                                        .fill(.quaternary.opacity(0.5))
                                )
                            Spacer(minLength: 40)
                        }
                        .transition(.opacity)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .animation(.spring(response: 0.4, dampingFraction: 0.85), value: showsToolChip)
            }
        }
        .onAppear { startPlayback() }
        .onDisappear {
            playback?.cancel()
            playback = nil
        }
    }

    private func startPlayback() {
        playback?.cancel()
        if reduceMotion {
            showsToolChip = true
            revealedCharacters = Self.assistantLine.count
            return
        }
        showsToolChip = false
        revealedCharacters = 0
        playback = Task {
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            showsToolChip = true
            try? await Task.sleep(for: .milliseconds(700))
            for count in 0...Self.assistantLine.count {
                guard !Task.isCancelled else { return }
                revealedCharacters = count
                try? await Task.sleep(for: .milliseconds(9))
            }
        }
    }
}
