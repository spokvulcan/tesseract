import Foundation

enum PromptCacheFormatting {
    static let bytesFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .memory
        formatter.includesUnit = true
        return formatter
    }()

    static let percentFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .percent
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = 1
        return formatter
    }()

    static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()

    static func bytes(_ value: Int) -> String {
        bytesFormatter.string(fromByteCount: Int64(value))
    }

    static func percent(_ value: Double) -> String {
        percentFormatter.string(from: NSNumber(value: value)) ?? "0%"
    }

    /// Human-scale duration from a millisecond count: milliseconds under a
    /// second, seconds under a minute, minutes beyond — "20499 ms" reads as
    /// "20.5 s", nobody converts in their head (owner call, 2026-07-11).
    static func milliseconds(_ value: Double) -> String {
        if value <= 0 { return "-" }
        if value < 10 { return String(format: "%.1f ms", value) }
        if value < 1_000 { return String(format: "%.0f ms", value) }
        let seconds = value / 1_000
        if seconds < 10 { return String(format: "%.2f s", seconds) }
        if seconds < 60 { return String(format: "%.1f s", seconds) }
        return String(format: "%dm %02ds", Int(seconds) / 60, Int(seconds) % 60)
    }

    static func compactNumber(_ value: Int) -> String {
        if value >= 1_000_000 { return String(format: "%.1fM", Double(value) / 1_000_000) }
        if value >= 1_000 { return String(format: "%.1fK", Double(value) / 1_000) }
        return "\(value)"
    }

    static func age(_ seconds: Double) -> String {
        if seconds < 1 { return "<1s" }
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        if seconds < 3_600 { return String(format: "%.0fm", seconds / 60) }
        return String(format: "%.1fh", seconds / 3_600)
    }
}
