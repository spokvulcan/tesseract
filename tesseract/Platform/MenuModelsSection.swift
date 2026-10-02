//
//  MenuModelsSection.swift
//  tesseract
//
//  The menu bar's Models section: each model Tesseract can hold, whether it
//  is loaded, and what it is doing right now — with how much memory the app
//  uses. Models run side by side (ADR-0081): memory, not the GPU, is the
//  shared limit, so this is where the owner sees what takes it. One glance,
//  no submenu: a SwiftUI view in the menu, refreshed while the menu is open.
//

import SwiftUI

/// One model's line in the menu.
nonisolated struct ModelActivityRow: Equatable, Identifiable, Sendable {
    enum State: Equatable, Sendable {
        case notLoaded
        case loading
        /// In memory, nothing running.
        case loaded
        /// Running right now: "Generating", "Speaking", "Listening"…
        case working(String)
    }

    let id: String
    /// What it is for: "Language model", "Voice", "Dictation"…
    let role: String
    /// Which model: "Qwen3.8 27B".
    let name: String
    let state: State
    /// Its weights' size in memory, when loaded and known.
    let bytes: Int64?

    init(id: String, role: String, name: String, state: State, bytes: Int64? = nil) {
        self.id = id
        self.role = role
        self.name = name
        self.state = state
        self.bytes = bytes
    }

    /// The right-hand text: what it is doing, and its size when loaded.
    var detail: String {
        let size = bytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .memory) }
        switch state {
        case .notLoaded: return "Not loaded"
        case .loading: return "Loading…"
        case .loaded: return size.map { "Idle · \($0)" } ?? "Idle"
        case .working(let what): return size.map { "\(what) · \($0)" } ?? what
        }
    }
}

/// Everything the Models section shows.
nonisolated struct ModelActivity: Equatable, Sendable {
    var rows: [ModelActivityRow]
    /// The app's memory footprint (what Activity Monitor shows).
    var footprintBytes: Int?
    /// The system's swap in use.
    var swapBytes: UInt64?

    static let empty = ModelActivity(rows: [], footprintBytes: nil, swapBytes: nil)

    /// "Tesseract uses 21.3 GB · swap 8.0 GB".
    var memoryLine: String {
        var parts: [String] = []
        if let footprintBytes {
            parts.append(
                "Tesseract uses "
                    + ByteCountFormatter.string(
                        fromByteCount: Int64(footprintBytes), countStyle: .memory))
        }
        if let swapBytes, swapBytes > 0 {
            parts.append(
                "swap "
                    + ByteCountFormatter.string(
                        fromByteCount: Int64(clamping: swapBytes), countStyle: .memory))
        }
        return parts.joined(separator: " · ")
    }
}

@Observable @MainActor
final class MenuModelsModel {
    var activity = ModelActivity.empty
}

/// The section's view: a dot per model (green working, gray loaded, hollow
/// not loaded), its role and name, and what it is doing.
struct MenuModelsView: View {
    let model: MenuModelsModel

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ForEach(model.activity.rows) { row in
                HStack(alignment: .center, spacing: 8) {
                    dot(row.state)
                    VStack(alignment: .leading, spacing: 0) {
                        Text(row.role)
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                        Text(row.name)
                            .font(.system(size: 13))
                            .lineLimit(1)
                    }
                    Spacer(minLength: 12)
                    Text(row.detail)
                        .font(.system(size: 12))
                        .monospacedDigit()
                        .foregroundStyle(isWorking(row.state) ? .primary : .secondary)
                        .lineLimit(1)
                }
            }
            if !model.activity.memoryLine.isEmpty {
                Text(model.activity.memoryLine)
                    .font(.system(size: 11))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
            }
        }
        .padding(.leading, 14)
        .padding(.trailing, 14)
        .padding(.vertical, 4)
        .frame(width: 320, alignment: .leading)
    }

    @ViewBuilder
    private func dot(_ state: ModelActivityRow.State) -> some View {
        switch state {
        case .notLoaded:
            Circle().strokeBorder(.secondary, lineWidth: 1).frame(width: 8, height: 8)
        case .loading:
            Circle().fill(.orange).frame(width: 8, height: 8)
        case .loaded:
            Circle().fill(.secondary).frame(width: 8, height: 8)
        case .working:
            Circle().fill(.green).frame(width: 8, height: 8)
        }
    }

    private func isWorking(_ state: ModelActivityRow.State) -> Bool {
        if case .working = state { return true }
        return false
    }
}
