import SwiftUI
import Textual
import MLXLMCommon

struct AgentStepView: View {
    let step: AssistantTurnView.Step
    let isLast: Bool

    @AppStorage("agentUseMarkdown") private var useMarkdown = true
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Timeline line and icon
            ZStack(alignment: .top) {
                if !isLast {
                    Rectangle()
                        .fill(Color(white: 0.2))
                        .frame(width: 1)
                        .padding(.top, 16) // Start just below the icon (which is 16pt tall)
                }
                
                iconView
                    .frame(width: 16, height: 16)
            }
            .frame(width: 20)
            
            // Content
            contentView
                .padding(.bottom, isLast ? 0 : 8)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .fixedSize(horizontal: false, vertical: true)
    }
    
    @ViewBuilder
    private var iconView: some View {
        switch step.type {
        case .thinking:
            Image(systemName: "brain")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        case .text:
            Image(systemName: "text.bubble")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        case .toolCall(let call, let result):
            Image(systemName: ToolDisplayHelpers.iconForTool(call.function.name))
                .font(.system(size: 11))
                .foregroundStyle(result?.isError == true ? .red : .secondary)
        }
    }
    
    @ViewBuilder
    private var contentView: some View {
        switch step.type {
        case .thinking(let content):
            AgentThinkingMinimalView(content: content)
                .frame(maxWidth: .infinity, alignment: .leading)
        case .text(let content):
            if useMarkdown {
                StructuredText(markdown: content)
                    .textual.structuredTextStyle(.gitHub)
                    .textual.textSelection(.enabled)
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Text(content)
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .toolCall(let call, let result):
            AgentToolCallMinimalView(toolCall: call, toolResult: result)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
