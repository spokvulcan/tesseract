import Foundation
import MLXLMCommon

struct GetCurrentTimeTool: AgentTool {
    let name = "get_current_time"
    let description = "Get the current date and time. Only call once per response."
    let parameters: [ToolParameter] = []

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let now = Date()
        let iso = now.formatted(.iso8601)
        let human = now.formatted(
            .dateTime.year().month(.wide).day()
                .hour().minute().second()
                .timeZone()
        )
        return "\(iso) (\(human))"
    }
}
