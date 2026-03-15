import Foundation
import MLXLMCommon

/// Shared helpers for displaying tool calls across agent views.
enum ToolDisplayHelpers {

    /// Maps a tool name to an SF Symbol icon name.
    static func iconForTool(_ name: String) -> String {
        switch name.lowercased() {
        case let n where n.contains("read"): return "doc.text"
        case let n where n.contains("write"): return "square.and.pencil"
        case let n where n.contains("edit"): return "pencil.line"
        case "ls": return "folder"
        case let n where n.contains("list"): return "folder"
        case let n where n.contains("memory"): return "brain"
        case let n where n.contains("task") || n.contains("goal"): return "checklist"
        case let n where n.contains("search"): return "magnifyingglass"
        default: return "wrench.adjustable"
        }
    }

    /// Human-readable title for a tool call, extracting filenames from arguments.
    static func titleForTool(_ name: String, arguments: [String: JSONValue]?) -> String {
        guard let args = arguments else { return name }

        switch name.lowercased() {
        case "read_file":
            if case .string(let path)? = args["path"] {
                return "Reading \((path as NSString).lastPathComponent)"
            }
            return "Reading file"
        case "write_file":
            if case .string(let path)? = args["path"] {
                return "Writing to \((path as NSString).lastPathComponent)"
            }
            return "Writing file"
        case "edit_file":
            if case .string(let path)? = args["path"] {
                return "Editing \((path as NSString).lastPathComponent)"
            }
            return "Editing file"
        case "ls", "list", "list_files", "list_directory":
            if case .string(let path)? = args["path"] {
                return "Listing files in \((path as NSString).lastPathComponent)"
            }
            return "Listing files"
        case "search_files":
            if case .string(let query)? = args["query"] {
                return "Searching for \"\(query)\""
            }
            return "Searching files"
        case "memory_save":
            return "Saving memory"
        case "task_create":
            return "Creating task"
        case "task_complete":
            return "Completing task"
        default:
            return name
        }
    }

    private static let argumentEncoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }()

    /// Pretty-prints tool call arguments as JSON.
    static func formatArguments(_ arguments: [String: JSONValue]) -> String {
        do {
            let data = try argumentEncoder.encode(arguments)
            if let jsonString = String(data: data, encoding: .utf8) {
                return jsonString == "{}" ? "No arguments" : jsonString
            }
        } catch {
            return "\(arguments)"
        }

        return "No arguments"
    }

    /// All display properties for a tool call, computed once.
    static func displayProps(for info: ToolCallInfo) -> (title: String, icon: String, argsFormatted: String) {
        let args = info.parsedArguments
        return (
            title: titleForTool(info.name, arguments: args),
            icon: iconForTool(info.name),
            argsFormatted: formatArguments(args)
        )
    }
}
