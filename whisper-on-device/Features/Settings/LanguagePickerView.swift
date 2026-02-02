//
//  LanguagePickerView.swift
//  whisper-on-device
//

import SwiftUI

struct LanguagePickerView: View {
    @Binding var selectedLanguage: String
    @State private var searchText = ""

    private var filteredLanguages: [SupportedLanguage] {
        if searchText.isEmpty {
            return SupportedLanguage.all
        }
        let search = searchText.lowercased()
        return SupportedLanguage.all.filter { language in
            language.name.lowercased().contains(search) ||
            language.code.lowercased().contains(search) ||
            (language.nativeName?.lowercased().contains(search) ?? false)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with current selection
            HStack {
                Text("Transcription Language")
                    .font(.headline)
                Spacer()
                if let current = SupportedLanguage.language(forCode: selectedLanguage) {
                    Text(current.displayName)
                        .foregroundStyle(.secondary)
                }
            }

            // Search field - use ZStack to ensure full clickable area
            ZStack {
                RoundedRectangle(cornerRadius: 6)
                    .fill(.quinary)

                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.tertiary)
                        .font(.system(size: 12))
                        .allowsHitTesting(false)

                    TextField("Search languages...", text: $searchText)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))

                    if !searchText.isEmpty {
                        Button {
                            searchText = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.tertiary)
                                .font(.system(size: 12))
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
            }
            .frame(height: 28)

            // Language list using ScrollView for reliable scrolling
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(filteredLanguages) { language in
                        LanguageRow(
                            language: language,
                            isSelected: language.code == selectedLanguage
                        )
                        .contentShape(Rectangle())
                        .onTapGesture {
                            // Dispatch to avoid "Publishing changes from within view updates"
                            Task { @MainActor in
                                selectedLanguage = language.code
                            }
                        }
                    }
                }
                .padding(8)
            }
            .frame(height: 280)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(.quaternary, lineWidth: 1)
            )

            // Helper text
            if filteredLanguages.isEmpty {
                Text("No languages found matching \"\(searchText)\"")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.top, 8)
            }
        }
    }
}

struct LanguageRow: View {
    let language: SupportedLanguage
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 10) {
            Text(language.flag)
                .font(.title3)

            VStack(alignment: .leading, spacing: 1) {
                Text(language.name)
                    .font(.system(size: 13, weight: isSelected ? .medium : .regular))
                if let nativeName = language.nativeName, nativeName != language.name {
                    Text(nativeName)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            if isSelected {
                Image(systemName: "checkmark")
                    .foregroundStyle(.tint)
                    .font(.system(size: 12, weight: .semibold))
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

// Compact picker for onboarding
struct CompactLanguagePickerView: View {
    @Binding var selectedLanguage: String
    @State private var searchText = ""

    private var filteredLanguages: [SupportedLanguage] {
        if searchText.isEmpty {
            return SupportedLanguage.all
        }
        let search = searchText.lowercased()
        return SupportedLanguage.all.filter { language in
            language.name.lowercased().contains(search) ||
            language.code.lowercased().contains(search) ||
            (language.nativeName?.lowercased().contains(search) ?? false)
        }
    }

    var body: some View {
        VStack(spacing: 12) {
            // Search field
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search languages...", text: $searchText)
                    .textFieldStyle(.plain)
                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(10)
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)

            // Language grid
            ScrollView {
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible())
                ], spacing: 8) {
                    ForEach(filteredLanguages) { language in
                        CompactLanguageCell(
                            language: language,
                            isSelected: language.code == selectedLanguage
                        )
                        .onTapGesture {
                            selectedLanguage = language.code
                        }
                    }
                }
                .padding(4)
            }
            .frame(height: 200)
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)
        }
    }
}

struct CompactLanguageCell: View {
    let language: SupportedLanguage
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 8) {
            Text(language.flag)
                .font(.title3)

            Text(language.name)
                .font(.callout)
                .lineLimit(1)

            Spacer(minLength: 0)

            if isSelected {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.tint)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(isSelected ? Color.accentColor.opacity(0.15) : Color(nsColor: .windowBackgroundColor))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
        )
    }
}

#Preview("Language Picker") {
    LanguagePickerView(selectedLanguage: .constant("en"))
        .frame(width: 300)
        .padding()
}

#Preview("Compact Picker") {
    CompactLanguagePickerView(selectedLanguage: .constant("en"))
        .frame(width: 400)
        .padding()
}
