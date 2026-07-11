//
//  ServerCacheView.swift
//  tesseract
//

import SwiftUI

/// The Cache page (map #269, direction locked in #272, prototype accepted
/// in #273): one page, two modes behind a remembered toolbar switch.
/// Overview answers "what is the cache buying me" — headline, tiles, and
/// three durable-source Swift Charts; Explorer keeps the full-power
/// radix-tree instrument. The ⌘` event console drawer is available in
/// both modes.
struct ServerCacheView: View {
    @Environment(AgentEngine.self) private var agentEngine
    @Environment(PromptCacheTelemetryStore.self) private var telemetry

    @AppStorage("server.cache.mode") private var modeRaw = CacheMode.overview.rawValue
    @AppStorage("server.cache.window") private var windowRaw = CacheWindow.day.rawValue
    @AppStorage("server.cache.events.open") private var isEventsOpen = false

    @State private var corpus = CacheCorpusStore()
    @State private var isInspectorPresented = false

    private enum CacheMode: String, CaseIterable, Identifiable {
        case overview
        case explorer

        var id: String { rawValue }

        var label: String {
            switch self {
            case .overview: "Overview"
            case .explorer: "Explorer"
            }
        }
    }

    private var mode: CacheMode {
        CacheMode(rawValue: modeRaw) ?? .overview
    }

    private var window: CacheWindow {
        CacheWindow(rawValue: windowRaw) ?? .day
    }

    var body: some View {
        @Bindable var telemetry = telemetry

        VStack(spacing: 0) {
            switch mode {
            case .overview:
                CacheOverviewView(
                    snapshot: telemetry.snapshot,
                    endurance: telemetry.endurance,
                    corpus: corpus,
                    window: window
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            case .explorer:
                PromptCacheTreeCanvasView(
                    tree: telemetry.selectedTree,
                    selectedNodeID: telemetry.selectedNodeID,
                    onSelectNode: telemetry.selectNode
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                CacheSelectionHUD(
                    tree: telemetry.selectedTree,
                    node: telemetry.selectedNode
                )
            }

            if isEventsOpen {
                PromptCacheEventsDrawer(onClose: { isEventsOpen = false })
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.32, dampingFraction: 0.86), value: isEventsOpen)
        .navigationTitle("Cache")
        .searchable(
            text: $telemetry.searchText,
            placement: .toolbar,
            prompt: "Offset, hash, checkpoint, storage"
        )
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Picker("Mode", selection: $modeRaw) {
                    ForEach(CacheMode.allCases) { mode in
                        Text(mode.label).tag(mode.rawValue)
                    }
                }
                .pickerStyle(.segmented)
                .help("Overview: what the cache is buying. Explorer: the radix-tree instrument.")

                switch mode {
                case .overview:
                    Picker("Window", selection: $windowRaw) {
                        ForEach(CacheWindow.allCases) { window in
                            Text(window.rawValue).tag(window.rawValue)
                        }
                    }
                    .pickerStyle(.segmented)
                    .help("Time window for the tiles and charts")

                    Button {
                        corpus.reload()
                        Task {
                            await telemetry.refreshSnapshot(llmActor: agentEngine.llmActor)
                        }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .help("Reload the trace corpus and snapshot")

                case .explorer:
                    explorerFilterMenu

                    Button {
                        telemetry.toggleLiveUpdates()
                    } label: {
                        Image(systemName: telemetry.isLive ? "pause.fill" : "play.fill")
                    }
                    .help(telemetry.isLive ? "Pause Events" : "Resume Events")

                    Button {
                        isInspectorPresented.toggle()
                    } label: {
                        Image(systemName: "sidebar.trailing")
                    }
                    .keyboardShortcut("i", modifiers: [.command, .option])
                    .help(isInspectorPresented ? "Hide Inspector (⌥⌘I)" : "Show Inspector (⌥⌘I)")
                }

                Button {
                    isEventsOpen.toggle()
                } label: {
                    Image(systemName: "square.bottomthird.inset.filled")
                }
                .keyboardShortcut("`", modifiers: .command)
                .help(isEventsOpen ? "Hide Events (⌘`)" : "Show Events (⌘`)")

                Menu {
                    Button("Copy Telemetry JSON") {
                        telemetry.copyExportJSONToPasteboard()
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuIndicator(.hidden)
            }
        }
        .inspector(isPresented: $isInspectorPresented) {
            ScrollView {
                PromptCacheInspectorView(
                    tree: telemetry.selectedTree,
                    node: telemetry.selectedNode,
                    event: telemetry.selectedEvent
                )
            }
            .inspectorColumnWidth(min: 260, ideal: 320, max: 420)
        }
        .task {
            telemetry.startPolling(llmActor: agentEngine.llmActor)
            await telemetry.refreshSnapshot(llmActor: agentEngine.llmActor)
            corpus.reload()
        }
        .onDisappear {
            telemetry.stopPolling()
        }
    }

    // MARK: - Explorer filters

    private var explorerFilterMenu: some View {
        @Bindable var telemetry = telemetry

        return Menu {
            Picker("Partition", selection: $telemetry.selectedPartitionID) {
                Text("First available").tag(nil as String?)
                ForEach(telemetry.filteredTrees) { tree in
                    Text(tree.partitionDigest).tag(Optional(tree.id))
                }
            }
            .pickerStyle(.menu)

            Section("Checkpoints") {
                ForEach(["system", "leaf", "branchPoint"], id: \.self) { type in
                    Toggle(type, isOn: checkpointBinding(type))
                }
            }

            Section("Storage") {
                ForEach(PromptCacheStorageState.allCases, id: \.self) { state in
                    Toggle(state.displayName, isOn: storageBinding(state))
                }
            }

            Divider()

            Button("Reset Filters") {
                telemetry.resetFilters()
            }
        } label: {
            Image(systemName: "line.3.horizontal.decrease.circle")
        }
        .help("Filter partition, checkpoint types, and storage states")
    }

    private func checkpointBinding(_ type: String) -> Binding<Bool> {
        Binding {
            telemetry.visibleCheckpointTypes.contains(type)
        } set: { enabled in
            if enabled {
                telemetry.visibleCheckpointTypes.insert(type)
            } else {
                telemetry.visibleCheckpointTypes.remove(type)
            }
        }
    }

    private func storageBinding(_ state: PromptCacheStorageState) -> Binding<Bool> {
        Binding {
            telemetry.visibleStorageStates.contains(state)
        } set: { enabled in
            if enabled {
                telemetry.visibleStorageStates.insert(state)
            } else {
                telemetry.visibleStorageStates.remove(state)
            }
        }
    }
}

// MARK: - Selection HUD

/// One monospace status line under the Explorer canvas — the selected
/// node's vitals or the visible tree's summary (the locked Drawer-era
/// grammar, kept for the instrument mode).
private struct CacheSelectionHUD: View {
    let tree: PromptCacheTreeSnapshot?
    let node: PromptCacheTreeNodeSnapshot?

    var body: some View {
        HStack(spacing: Theme.Spacing.lg) {
            Text(leadingText)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .textSelection(.enabled)

            Spacer(minLength: 0)

            Text("⌘` events · ⌥⌘I inspector")
                .foregroundStyle(.quaternary)
        }
        .font(.caption.monospaced())
        .monospacedDigit()
        .padding(.horizontal, Theme.Spacing.md)
        .padding(.vertical, 5)
        .background(.background.secondary)
    }

    private var leadingText: String {
        if let node {
            var parts = ["▸ \(node.checkpointType ?? "path") @\(node.tokenOffset.formatted())"]
            if node.hasSnapshot || node.snapshotBytes > 0 {
                parts.append(PromptCacheFormatting.bytes(node.snapshotBytes))
            }
            parts.append(node.storageState.displayName)
            parts.append("hit \(PromptCacheFormatting.age(node.lastAccessAgeSeconds)) ago")
            if let utility = node.utility {
                parts.append(String(format: "utility %.2f", utility))
            }
            return parts.joined(separator: " · ")
        }
        if let tree {
            return
                "partition \(tree.partitionDigest) · \(tree.nodeCount) nodes"
                + " · \(tree.snapshotCount) snapshots"
                + " · \(PromptCacheFormatting.bytes(tree.totalSnapshotBytes))"
        }
        return "no topology — run an HTTP completion to instantiate the radix tree"
    }
}
