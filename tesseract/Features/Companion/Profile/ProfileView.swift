//
//  ProfileView.swift
//  tesseract
//
//  The Profile page: everything Jarvis knows about the owner, and nothing
//  else. Every fact can be read, edited and deleted; proposals wait here
//  (and in Today) until the owner decides.
//

import SwiftUI

struct ProfileView: View {
    @Environment(ProfileStore.self) private var profile
    @State private var newFact = ""

    var body: some View {
        Form {
            Section {
                ForEach(profile.facts) { fact in
                    FactRow(fact: fact)
                }
                HStack {
                    TextField("Add something Jarvis should know", text: $newFact)
                        .onSubmit(add)
                    Button("Add", action: add)
                        .disabled(newFact.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            } header: {
                Text("What Jarvis knows about you")
            } footer: {
                Text(
                    "This is all of it. Jarvis reads these facts at the start of each day's thread, and nowhere else. He saves one only when you ask him to remember something or keep one of his proposals."
                )
            }

            if !profile.openProposals.isEmpty {
                Section("Should I remember this?") {
                    ForEach(profile.openProposals) { proposal in
                        ProposalRow(proposal: proposal)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 520, minHeight: 420)
    }

    private func add() {
        profile.add(newFact, source: .owner)
        newFact = ""
    }
}

private struct FactRow: View {
    @Environment(ProfileStore.self) private var profile
    let fact: ProfileFact
    @State private var text = ""
    @State private var editing = false

    var body: some View {
        HStack {
            if editing {
                TextField("Fact", text: $text).onSubmit(save)
                Button("Save", action: save)
                Button("Cancel") { editing = false }
            } else {
                Text(fact.text)
                Spacer()
                Button("Edit") {
                    text = fact.text
                    editing = true
                }
                Button("Delete", role: .destructive) { profile.delete(fact.id) }
            }
        }
    }

    private func save() {
        profile.update(fact.id, text: text, area: fact.area)
        editing = false
    }
}

/// One "Should I remember this?" proposal: Remember, Edit, Not true.
struct ProposalRow: View {
    @Environment(ProfileStore.self) private var profile
    let proposal: FactProposal
    @State private var text = ""
    @State private var editing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if editing {
                TextField("Fact", text: $text).onSubmit(keepEdited)
            } else {
                Text(proposal.text)
            }
            if !proposal.reason.isEmpty {
                Text(proposal.reason).foregroundStyle(.secondary).fixedSize(
                    horizontal: false, vertical: true)
            }
            HStack(spacing: 12) {
                if editing {
                    Button("Remember", action: keepEdited)
                    Button("Cancel") { editing = false }
                } else {
                    Button("Remember") { profile.decide(proposal.id, keep: true) }
                    Button("Edit") {
                        text = proposal.text
                        editing = true
                    }
                    Button("Not true") { profile.decide(proposal.id, keep: false) }
                }
            }
            .buttonStyle(.plain)
            .foregroundStyle(Color.accentColor)
            .focusable(false)
        }
    }

    private func keepEdited() {
        profile.decide(proposal.id, keep: true, editedText: text)
        editing = false
    }
}
