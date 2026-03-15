import SwiftUI
import os

struct AgentSystemPromptView: View {
    @Environment(AgentCoordinator.self) private var coordinator
    @State private var isExpanded = false
    @State private var selectedTab: PromptTab = .assembled

    private enum PromptTab: String, CaseIterable {
        case assembled = "Assembled"
        case rawChatML = "Raw ChatML"
    }

    private var hasRawPrompt: Bool { coordinator.rawChatMLPrompt != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header button
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    isExpanded.toggle()
                }
            }) {
                HStack(spacing: 4) {
                    Text("System Prompt")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)

                    if let count = coordinator.systemPromptTokenCount {
                        Text("(\(count) tokens)")
                            .font(.system(size: 11))
                            .foregroundStyle(.tertiary)
                    }

                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isExpanded ? 0 : -90))
                }
                .padding(.vertical, 4)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            // Expanded content
            if isExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    Picker("", selection: $selectedTab) {
                        ForEach(PromptTab.allCases, id: \.self) { tab in
                            Text(tab.rawValue).tag(tab)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()

                    ScrollView {
                        promptContent
                    }
                    .frame(maxHeight: 300)
                }
                .padding(.horizontal, 14)
                .padding(.bottom, 10)
                .padding(.top, 4)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .task(id: hasRawPrompt) {
            Log.agent.info("SystemPromptView .task fired — hasRawPrompt=\(hasRawPrompt)")
            if !hasRawPrompt {
                coordinator.fetchRawSystemPrompt()
            }
        }
        .onChange(of: coordinator.isGenerating) {
            Log.agent.info("SystemPromptView .onChange(isGenerating) — isGenerating=\(coordinator.isGenerating), hasRawPrompt=\(hasRawPrompt)")
            if !coordinator.isGenerating, !hasRawPrompt {
                coordinator.fetchRawSystemPrompt()
            }
        }
    }

    @ViewBuilder
    private var promptContent: some View {
        switch selectedTab {
        case .assembled:
            monoText(coordinator.assembledSystemPrompt)
        case .rawChatML:
            if let raw = coordinator.rawChatMLPrompt {
                monoText(raw)
            } else {
                Text("Load a model to view the raw ChatML prompt")
                    .font(.system(size: 12))
                    .foregroundStyle(.tertiary)
                    .italic()
            }
        }
    }

    private func monoText(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 12, design: .monospaced))
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}
