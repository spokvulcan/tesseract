import SwiftUI

struct SlashCommandPopupView: View {
    let commands: [SlashCommand]
    let selectedIndex: Int
    let onSelect: (SlashCommand) -> Void

    @State private var hoveredId: String?
    @State private var pendingScrollTask: Task<Void, Never>?

    private static let rowHeight: CGFloat = 38

    private var popupHeight: CGFloat {
        let contentHeight = CGFloat(max(commands.count, 1)) * Self.rowHeight + 8
        return min(contentHeight, 240)
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
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
                    scheduleScroll(proxy: proxy, id: commands[newIndex].id)
                }
            }
            .onDisappear {
                pendingScrollTask?.cancel()
                pendingScrollTask = nil
            }
        }
        .frame(width: 360, height: popupHeight)
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
        .frame(maxWidth: .infinity, minHeight: Self.rowHeight, alignment: .leading)
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

    private func scheduleScroll(proxy: ScrollViewProxy, id: String) {
        pendingScrollTask?.cancel()
        pendingScrollTask = Task { @MainActor in
            await Task.yield()
            guard !Task.isCancelled else { return }
            proxy.scrollTo(id, anchor: .center)
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
