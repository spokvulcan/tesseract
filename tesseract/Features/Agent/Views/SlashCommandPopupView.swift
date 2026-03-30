import SwiftUI

struct SlashCommandPopupView: View {
    let commands: [SlashCommand]
    let selectedIndex: Int
    let onSelect: (SlashCommand) -> Void

    @State private var hoveredId: String?

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(commands.enumerated()), id: \.element.id) { index, command in
                        commandRow(command, isSelected: index == selectedIndex, isHovered: hoveredId == command.id)
                            .id(command.id)
                            .onHover { hovering in
                                hoveredId = hovering ? command.id : nil
                            }
                            .onTapGesture { onSelect(command) }
                    }
                }
                .padding(.vertical, 4)
            }
            .onChange(of: selectedIndex) { _, newIndex in
                if commands.indices.contains(newIndex) {
                    withAnimation(.easeOut(duration: 0.1)) {
                        proxy.scrollTo(commands[newIndex].id, anchor: .center)
                    }
                }
            }
        }
        .frame(maxHeight: 240)
        .frame(maxWidth: 360)
        .fixedSize(horizontal: false, vertical: true)
        .glassEffect(.regular.interactive(), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .shadow(color: .black.opacity(0.15), radius: 16, x: 0, y: -4)
    }

    @ViewBuilder
    private func commandRow(_ command: SlashCommand, isSelected: Bool, isHovered: Bool) -> some View {
        HStack(spacing: 8) {
            Image(systemName: iconForSource(command.source))
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text("/\(command.name)")
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                    .foregroundStyle(.primary)
                Text(command.description)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            if let hint = command.argumentHint {
                Text(hint)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .contentShape(Rectangle())
        .background {
            if isSelected {
                RoundedRectangle(cornerRadius: 6)
                    .fill(.selection)
                    .padding(.horizontal, 4)
            } else if isHovered {
                RoundedRectangle(cornerRadius: 6)
                    .fill(.quaternary)
                    .padding(.horizontal, 4)
            }
        }
    }

    private func iconForSource(_ source: SlashCommandSource) -> String {
        switch source {
        case .builtIn: return "terminal"
        case .skill: return "book"
        case .extension: return "puzzlepiece.extension"
        }
    }
}
