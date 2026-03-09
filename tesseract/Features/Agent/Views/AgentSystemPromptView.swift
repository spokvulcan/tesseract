import SwiftUI
import os

struct AgentSystemPromptView: View {
    @EnvironmentObject private var coordinator: AgentCoordinator
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
                HStack(spacing: 8) {
                    Image(systemName: "terminal")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .frame(width: 16)

                    Text("System Prompt")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)

                    if let count = coordinator.systemPromptTokenCount {
                        Text("\(count) tokens")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.quaternary.opacity(0.5))
                            .clipShape(Capsule())
                    }

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                .padding(.vertical, 6)
                .padding(.horizontal, 14)
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
