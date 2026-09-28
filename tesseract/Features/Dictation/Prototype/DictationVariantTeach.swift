//
//  DictationVariantTeach.swift
//  tesseract
//
//  PROTOTYPE — THROWAWAY (Dictation page redesign, 2026-09-28). Never merge.
//
//  C · Teach. Works on any text, dictated now or last week, or typed:
//  select the wrong words in any app, press ⌃⌥Space, type what it should
//  be, Return. The selection is replaced in place and the app learns it;
//  ⌘Return only adds the word to what the app knows. With nothing selected
//  the shortcut fixes the last dictation. The page is the dictionary.
//

import SwiftUI

@MainActor
enum TeachFlow {
    static func open(_ lab: DictationLab) async {
        let appName = NSWorkspace.shared.frontmostApplication?.localizedName ?? "the app"
        guard let selection = await lab.replacer?.readSelection(),
            !selection.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            selection.text.count <= 120
        else {
            FixBarFlow.open(lab)
            return
        }
        lab.panels.closeCard()
        lab.panels.showKey(
            TeachField(lab: lab, selected: selection.text, appName: appName) {
                lab.panels.closeKey()
            },
            size: CGSize(width: 480, height: 150), anchor: selection.bounds
        ) {}
    }

    /// After a take: only speaks up when something learned changed it.
    static func showApplied(_ lab: DictationLab, take: LabTake) {
        guard !take.applied.isEmpty else { return }
        lab.panels.showCard(
            AppliedCard(take: take), size: CGSize(width: 460, height: 64), duration: .seconds(4))
    }
}

struct AppliedCard: View {
    let take: LabTake
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "sparkle")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Color.accentColor)
            ForEach(Array(take.applied.prefix(2).enumerated()), id: \.offset) { _, item in
                HeardMeant(heard: item.heard, meant: item.term, size: 12)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .frame(height: 40)
        .glassEffect(.regular, in: .capsule)
        .padding(8)
    }
}

/// The popover-like field that sits under a selection.
struct TeachField: View {
    let lab: DictationLab
    let selected: String
    let appName: String
    let onDone: () -> Void
    @State private var text = ""
    @State private var selection: TextSelection?
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Text(selected)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Image(systemName: "arrow.right")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.tertiary)
                Text("should be")
                    .font(.system(size: 12))
                    .foregroundStyle(.tertiary)
            }
            TextField("", text: $text, selection: $selection)
                .textFieldStyle(.plain)
                .font(.system(size: 17, weight: .medium))
                .focused($focused)
                .onSubmit(teach)
                .onKeyPress(.escape) {
                    onDone()
                    return .handled
                }
            HStack(spacing: 12) {
                ForEach(Array(lab.suggestions(for: selected).enumerated()), id: \.offset) {
                    index, option in
                    Button {
                        text = option
                    } label: {
                        HStack(spacing: 4) {
                            Text(option).font(.system(size: 12, weight: .medium))
                            Text("⌘\(index + 1)").font(.system(size: 11)).foregroundStyle(.tertiary)
                        }
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
                }
                Spacer()
                Button("Just add the word") { addOnly() }
                    .buttonStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .keyboardShortcut(.return, modifiers: .command)
                Text("↩ replace and learn")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(16)
        .frame(width: 460, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 16))
        .padding(10)
        // Escape reaches a text field as cancelOperation, not a key press.
        .onExitCommand(perform: onDone)
        .onAppear {
            text = selected
            selection = TextSelection(range: text.startIndex..<text.endIndex)
            focused = true
        }
    }

    private func teach() {
        let corrected = text.trimmingCharacters(in: .whitespacesAndNewlines)
        onDone()
        guard corrected != selected else { return }
        Task {
            try? await Task.sleep(for: .milliseconds(120))
            await lab.teach(selection: selected, to: corrected, appName: appName)
        }
    }

    private func addOnly() {
        let word = text.trimmingCharacters(in: .whitespacesAndNewlines)
        onDone()
        lab.add(term: word, source: .typed)
        lab.showToast(LabToast(title: "Added \(word)", detail: "Whisper will lean toward it"))
    }
}

// MARK: - Page

struct TeachPage: View {
    let lab: DictationLab
    @Environment(TranscriptionHistory.self) private var history
    @State private var query = ""
    @State private var newTerm = ""
    @State private var showsTakes = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                TextField("Search your words", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 260)
                Spacer()
                TextField("Add a word or name", text: $newTerm)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 220)
                    .onSubmit {
                        lab.add(term: newTerm)
                        newTerm = ""
                    }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            Divider()
            table
            if !lab.memorySuggestions.isEmpty { suggestions }
            Divider()
            DisclosureGroup("Recent takes", isExpanded: $showsTakes) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(history.entries.prefix(30)) { entry in
                            Text(entry.text)
                                .font(.system(size: 13))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(.vertical, 8)
                }
                .frame(maxHeight: 180)
            }
            .font(.system(size: 13))
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            HStack {
                LabStatusLine(extra: "select any wrong text, \(LabStyle.fixShortcut) to teach")
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 10)
        }
    }

    private var rows: [LabTerm] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return lab.lexicon.terms }
        return lab.lexicon.terms.filter {
            $0.term.lowercased().contains(q) || $0.heardAs.contains(where: { $0.contains(q) })
        }
    }

    @ViewBuilder
    private var table: some View {
        if lab.lexicon.terms.isEmpty {
            ContentUnavailableView {
                Label("No words yet", systemImage: "character.book.closed")
            } description: {
                Text(
                    "Select a misheard word in any app and press \(LabStyle.fixShortcut). Type what it should be; from then on the app writes it right."
                )
            }
            .frame(maxHeight: .infinity)
        } else {
            Table(rows) {
                TableColumn("Word") { term in
                    LearnedWord(term.term, size: 13)
                }
                .width(min: 120, ideal: 160)
                TableColumn("Heard as") { term in
                    Text(term.heardAs.isEmpty ? "—" : term.heardAs.joined(separator: ", "))
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .width(min: 160, ideal: 240)
                TableColumn("From") { term in
                    Text(term.source.rawValue).foregroundStyle(.secondary)
                }
                .width(90)
                TableColumn("Fixed") { term in
                    Text(term.applied == 0 ? "—" : "\(term.applied)×")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                .width(50)
            }
            .contextMenu(forSelectionType: LabTerm.ID.self) { ids in
                Button("Forget", role: .destructive) {
                    for id in ids { lab.forget(id) }
                }
            }
        }
    }

    private var suggestions: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text("From your memory")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(lab.memorySuggestions, id: \.self) { word in
                        Button {
                            lab.add(term: word, source: .memory)
                        } label: {
                            Label(word, systemImage: "plus").font(.system(size: 12))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .contextMenu {
                            Button("Not a word I use") { lab.dismissSuggestion(word) }
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }
}
