import SwiftUI

/// The Prompt Cache page in the Server telemetry grammar: a hero band of
/// four large numbers over the full-bleed radix-tree canvas, a one-line
/// selection HUD, and an events console in a slide-up drawer (⌘`).
/// Filters and actions live in the system toolbar — the content layer is
/// plain, with no glass and no sub-page navigation.
struct ServerPromptCacheView: View {
    @Environment(AgentEngine.self) private var agentEngine
    @Environment(PromptCacheTelemetryStore.self) private var telemetry

    @State private var isInspectorPresented = false
    @AppStorage("server.promptCache.events.open") private var isEventsOpen = false

    var body: some View {
        @Bindable var telemetry = telemetry

        VStack(spacing: 0) {
            PromptCacheHeroBand(
                snapshot: telemetry.snapshot,
                aggregate: telemetry.aggregate,
                samples: telemetry.metricSamples,
                isLive: telemetry.isLive
            )

            PromptCacheVitalsStrip(
                outcome: telemetry.lastRequestOutcome,
                endurance: telemetry.endurance,
                ssdEnabled: telemetry.snapshot?.ssd.enabled == true
            )

            Divider()

            PromptCacheTreeCanvasView(
                tree: telemetry.selectedTree,
                selectedNodeID: telemetry.selectedNodeID,
                onSelectNode: telemetry.selectNode
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            PromptCacheSelectionHUD(
                tree: telemetry.selectedTree,
                node: telemetry.selectedNode
            )

            if isEventsOpen {
                PromptCacheEventsDrawer(onClose: { isEventsOpen = false })
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.32, dampingFraction: 0.86), value: isEventsOpen)
        .navigationTitle("Prompt Cache")
        .searchable(
            text: $telemetry.searchText,
            placement: .toolbar,
            prompt: "Offset, hash, checkpoint, storage"
        )
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                filterMenu

                Button {
                    telemetry.toggleLiveUpdates()
                } label: {
                    Image(systemName: telemetry.isLive ? "pause.fill" : "play.fill")
                }
                .help(telemetry.isLive ? "Pause Events" : "Resume Events")

                Button {
                    Task { await telemetry.refreshSnapshot(llmActor: agentEngine.llmActor) }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh Snapshot")

                Button {
                    isInspectorPresented.toggle()
                } label: {
                    Image(systemName: "sidebar.trailing")
                }
                .keyboardShortcut("i", modifiers: [.command, .option])
                .help(isInspectorPresented ? "Hide Inspector (⌥⌘I)" : "Show Inspector (⌥⌘I)")

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
            PromptCacheInspectorPanel(
                tree: telemetry.selectedTree,
                node: telemetry.selectedNode,
                event: telemetry.selectedEvent
            )
        }
        .task {
            telemetry.startPolling(llmActor: agentEngine.llmActor)
            await telemetry.refreshSnapshot(llmActor: agentEngine.llmActor)
        }
        .onDisappear {
            telemetry.stopPolling()
        }
    }

    // MARK: - Toolbar pieces

    private var filterMenu: some View {
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

/// One monospace status line under the tree canvas: the selected node's
/// vitals, or the visible tree's summary — the Dashboard meta-line
/// grammar in place of the old status card + inspector ceremony.
private struct PromptCacheSelectionHUD: View {
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

// MARK: - Inspector panel

/// Trailing system inspector: the selection's deep numbers in a standard
/// column, not a transient popover. The system supplies the surface —
/// the content stays plain.
private struct PromptCacheInspectorPanel: View {
    let tree: PromptCacheTreeSnapshot?
    let node: PromptCacheTreeNodeSnapshot?
    let event: PromptCacheTelemetryEvent?

    var body: some View {
        ScrollView {
            PromptCacheInspectorView(tree: tree, node: node, event: event)
        }
        .inspectorColumnWidth(min: 260, ideal: 320, max: 420)
    }
}
