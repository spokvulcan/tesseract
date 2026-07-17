//
//  CompanionInstructionsView.swift
//  tesseract
//
//  The owner's window on the entity's standing instructions (ADR-0040 §12):
//  read the version in force, edit it (an owner-authored revision — appended,
//  never rewritten), and walk the history of who changed what, when, and why.
//

import SwiftUI

struct CompanionInstructionsView: View {

    let store: MemoryStore
    let recorder: CompanionFlightRecorder

    @State private var history: [CompanionInstructionsVersion] = []
    @State private var draft: String = ""
    @State private var loadedVersion: Int?
    @State private var saveNote: String = ""

    private var current: CompanionInstructionsVersion? { history.first }
    private var isDirty: Bool { draft != (current?.text ?? "") }

    var body: some View {
        HSplitView {
            editor
                .frame(minWidth: 420)
            historyList
                .frame(minWidth: 240, maxWidth: 340)
        }
        .frame(minWidth: 700, minHeight: 420)
        .task { await reload() }
    }

    private var editor: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                if let current {
                    Text("v\(current.version) · \(current.author)")
                        .font(.headline)
                    Text(current.createdAt.formatted(date: .abbreviated, time: .shortened))
                        .foregroundStyle(.secondary)
                } else {
                    Text("Not seeded yet — the first Companion turn installs v1.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            TextEditor(text: $draft)
                .font(.body.monospaced())
                .scrollContentBackground(.hidden)
                .padding(6)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
            HStack {
                TextField("Why this edit? (goes in the history)", text: $saveNote)
                    .textFieldStyle(.roundedBorder)
                Button("Save as New Version") { Task { await save() } }
                    .keyboardShortcut("s", modifiers: [.command])
                    .disabled(
                        !isDirty
                            || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            Text(
                "Edits append a new owner-authored version — the Companion sees it on his "
                    + "next turn, and his own revisions land here for you to read."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding()
    }

    private var historyList: some View {
        List(history) { version in
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("v\(version.version)")
                        .font(.subheadline.bold())
                    Text(version.author)
                        .font(.caption)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(badgeColor(version.author).opacity(0.2), in: Capsule())
                    Spacer()
                    Text(version.createdAt.formatted(date: .numeric, time: .shortened))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let note = version.note, !note.isEmpty {
                    Text(note)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }
            }
            .padding(.vertical, 2)
            .contentShape(Rectangle())
            .onTapGesture { load(version) }
            .listRowBackground(
                loadedVersion == version.version
                    ? Color.accentColor.opacity(0.12) : Color.clear)
        }
        .listStyle(.inset)
    }

    private func badgeColor(_ author: String) -> Color {
        switch author {
        case "entity": .blue
        case "owner": .green
        default: .gray
        }
    }

    /// Loading an old version into the editor is how "restore" works: save it
    /// and it becomes the newest version — history stays intact.
    private func load(_ version: CompanionInstructionsVersion) {
        draft = version.text
        loadedVersion = version.version
        if let current, version.version != current.version {
            saveNote = "restored from v\(version.version)"
        }
    }

    private func reload() async {
        history = (try? await store.instructionsHistory()) ?? []
        if let current {
            draft = current.text
            loadedVersion = current.version
        }
    }

    private func save() async {
        let text = draft
        let note = saveNote.isEmpty ? "owner edit" : saveNote
        guard
            let version = try? await store.appendInstructions(
                text: text, author: "owner", note: note)
        else { return }
        recorder.record(
            "instructions.owner-edited",
            snapshot: ["version": String(version), "chars": String(text.count)],
            note: note)
        saveNote = ""
        await reload()
    }
}
