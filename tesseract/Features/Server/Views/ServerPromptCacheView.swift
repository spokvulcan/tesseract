import SwiftUI

struct ServerPromptCacheView: View {
    @Environment(AgentEngine.self) private var agentEngine
    @Environment(PromptCacheTelemetryStore.self) private var telemetry

    @State private var isInspectorPresented = false

    @SceneStorage("ServerPromptCacheView.section")
    private var sectionRawValue = PromptCacheSection.tree.rawValue

    var body: some View {
        @Bindable var telemetry = telemetry

        GeometryReader { proxy in
            content(
                telemetry: telemetry,
                width: proxy.size.width
            )
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle("Prompt Cache")
        .searchable(
            text: $telemetry.searchText,
            placement: .toolbar,
            prompt: "Offset, hash, checkpoint, storage"
        )
        .toolbar {
            ToolbarItem(placement: .principal) {
                sectionPicker
            }

            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    isInspectorPresented.toggle()
                } label: {
                    Image(systemName: "info.circle")
                }
                .help("Show Selection Details")
                .popover(isPresented: $isInspectorPresented, arrowEdge: .top) {
                    PromptCacheInspectorPopover(
                        tree: telemetry.selectedTree,
                        node: telemetry.selectedNode,
                        event: telemetry.selectedEvent
                    )
                }

                Button {
                    Task { await telemetry.refreshSnapshot(agentEngine: agentEngine) }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh Snapshot")

                Button {
                    telemetry.toggleLiveUpdates()
                } label: {
                    Image(systemName: telemetry.isLive ? "pause.fill" : "play.fill")
                }
                .help(telemetry.isLive ? "Pause Events" : "Resume Events")

                Button {
                    telemetry.copyExportJSONToPasteboard()
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .help("Copy Telemetry JSON")
            }
        }
        .task {
            telemetry.startPolling(agentEngine: agentEngine)
            await telemetry.refreshSnapshot(agentEngine: agentEngine)
        }
        .onDisappear {
            telemetry.stopPolling()
        }
    }

    private var sectionPicker: some View {
        ViewThatFits(in: .horizontal) {
            sectionPickerContent(useSymbols: false)
                .frame(width: 342)

            sectionPickerContent(useSymbols: false)
                .frame(width: 304)

            sectionPickerContent(useSymbols: true)
                .frame(width: 212)
        }
        .help("Prompt cache section")
    }

    private func sectionPickerContent(useSymbols: Bool) -> some View {
        Picker("Prompt Cache Section", selection: sectionBinding) {
            ForEach(PromptCacheSection.allCases) { section in
                if useSymbols {
                    Image(systemName: section.symbol)
                        .tag(section)
                } else {
                    Text(section.title)
                        .tag(section)
                }
            }
        }
        .pickerStyle(.segmented)
        .controlSize(.regular)
        .labelsHidden()
    }

    @ViewBuilder
    private func content(
        telemetry: PromptCacheTelemetryStore,
        width: CGFloat
    ) -> some View {
        let section = PromptCacheSection(rawValue: sectionRawValue) ?? .tree
        let padding = contentPadding(for: width)

        switch section {
        case .overview:
            PromptCacheOverviewView(
                snapshot: telemetry.snapshot,
                aggregate: telemetry.aggregate,
                samples: telemetry.metricSamples
            )
            .padding(padding)

        case .tree:
            VStack(spacing: spacing(for: width)) {
                filterBar(telemetry: telemetry, compact: width < PromptCacheLayout.compactWidth)

                PromptCacheTreeCanvasView(
                    tree: telemetry.selectedTree,
                    selectedNodeID: telemetry.selectedNodeID,
                    onSelectNode: telemetry.selectNode
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                PromptCacheSelectionStatusView(
                    tree: telemetry.selectedTree,
                    node: telemetry.selectedNode,
                    onShowDetails: { isInspectorPresented = true }
                )
            }
            .padding(padding)

        case .events:
            VStack(spacing: spacing(for: width)) {
                filterBar(telemetry: telemetry, compact: width < PromptCacheLayout.compactWidth)
                PromptCacheEventTableView(
                    store: telemetry,
                    onShowDetails: { isInspectorPresented = true }
                )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .padding(padding)
        }
    }

    private func filterBar(
        telemetry: PromptCacheTelemetryStore,
        compact: Bool
    ) -> some View {
        @Bindable var telemetry = telemetry

        return ViewThatFits(in: .horizontal) {
            HStack(spacing: Theme.Spacing.sm) {
                partitionMenu(telemetry: telemetry)
                checkpointMenu(telemetry: telemetry)
                storageMenu(telemetry: telemetry)
                resetButton(telemetry: telemetry)
                Spacer(minLength: 0)
                liveStatus(telemetry: telemetry)
            }

            HStack(spacing: Theme.Spacing.xs) {
                partitionMenu(telemetry: telemetry, title: "Part")
                checkpointMenu(telemetry: telemetry, title: "Type")
                storageMenu(telemetry: telemetry, title: "State")
                resetButton(telemetry: telemetry, iconOnly: true)
                Spacer(minLength: 0)
                if !compact {
                    liveStatus(telemetry: telemetry)
                }
            }
        }
        .controlSize(.small)
    }

    private func partitionMenu(
        telemetry: PromptCacheTelemetryStore,
        title: String = "Partition"
    ) -> some View {
        @Bindable var telemetry = telemetry

        return Menu {
            Picker("Partition", selection: $telemetry.selectedPartitionID) {
                Text("First available").tag(nil as String?)
                ForEach(telemetry.filteredTrees) { tree in
                    Text(tree.partitionDigest).tag(Optional(tree.id))
                }
            }
        } label: {
            Label(title, systemImage: "square.stack.3d.up")
        }
        .menuStyle(.button)
        .help("Select cache partition")
    }

    private func checkpointMenu(
        telemetry: PromptCacheTelemetryStore,
        title: String = "Checkpoint"
    ) -> some View {
        Menu {
            ForEach(["system", "leaf", "branchPoint"], id: \.self) { type in
                Toggle(type, isOn: checkpointBinding(type, telemetry: telemetry))
            }
            Divider()
            Button("Reset Filters") {
                telemetry.resetFilters()
            }
        } label: {
            Label(title, systemImage: "line.3.horizontal.decrease.circle")
        }
        .menuStyle(.button)
        .help("Filter checkpoint types")
    }

    private func storageMenu(
        telemetry: PromptCacheTelemetryStore,
        title: String = "Storage"
    ) -> some View {
        Menu {
            ForEach(PromptCacheStorageState.allCases, id: \.self) { state in
                Toggle(state.displayName, isOn: storageBinding(state, telemetry: telemetry))
            }
            Divider()
            Button("All Storage States") {
                telemetry.visibleStorageStates = Set(PromptCacheStorageState.allCases)
            }
        } label: {
            Label(title, systemImage: "externaldrive")
        }
        .menuStyle(.button)
        .help("Filter storage states")
    }

    private func resetButton(
        telemetry: PromptCacheTelemetryStore,
        iconOnly: Bool = false
    ) -> some View {
        Button {
            telemetry.resetFilters()
        } label: {
            if iconOnly {
                Image(systemName: "xmark.circle")
            } else {
                Label("Reset", systemImage: "xmark.circle")
            }
        }
        .help("Reset filters")
    }

    private func liveStatus(telemetry: PromptCacheTelemetryStore) -> some View {
        HStack(spacing: 5) {
            Circle()
                .fill(telemetry.isLive ? .green : .orange)
                .frame(width: 6, height: 6)
            Text(telemetry.isLive ? "Live" : "Paused")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .glassEffect(.regular, in: Capsule())
    }

    private var sectionBinding: Binding<PromptCacheSection> {
        Binding {
            PromptCacheSection(rawValue: sectionRawValue) ?? .tree
        } set: { section in
            sectionRawValue = section.rawValue
        }
    }

    private func checkpointBinding(
        _ type: String,
        telemetry: PromptCacheTelemetryStore
    ) -> Binding<Bool> {
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

    private func storageBinding(
        _ state: PromptCacheStorageState,
        telemetry: PromptCacheTelemetryStore
    ) -> Binding<Bool> {
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

    private func contentPadding(for width: CGFloat) -> CGFloat {
        width < PromptCacheLayout.compactWidth ? Theme.Spacing.sm : Theme.Spacing.md
    }

    private func spacing(for width: CGFloat) -> CGFloat {
        width < PromptCacheLayout.compactWidth ? Theme.Spacing.xs : Theme.Spacing.sm
    }
}

private enum PromptCacheSection: String, CaseIterable, Identifiable {
    case overview
    case tree
    case events

    var id: String { rawValue }

    var title: String {
        switch self {
        case .overview: "Overview"
        case .tree: "Tree"
        case .events: "Events"
        }
    }

    var symbol: String {
        switch self {
        case .overview: "chart.bar"
        case .tree: "point.3.connected.trianglepath.dotted"
        case .events: "list.bullet.rectangle"
        }
    }
}

private struct PromptCacheInspectorPopover: View {
    let tree: PromptCacheTreeSnapshot?
    let node: PromptCacheTreeNodeSnapshot?
    let event: PromptCacheTelemetryEvent?

    var body: some View {
        ScrollView {
            PromptCacheInspectorView(tree: tree, node: node, event: event)
                .padding(Theme.Spacing.sm)
        }
        .frame(minWidth: 320, idealWidth: 380, maxWidth: 460, minHeight: 260, idealHeight: 430, maxHeight: 560)
    }
}

enum PromptCacheChartKind: String, CaseIterable, Identifiable {
    case efficiency
    case memory
    case latency

    var id: String { rawValue }

    var title: String {
        switch self {
        case .efficiency: "Efficiency"
        case .memory: "Memory"
        case .latency: "Latency"
        }
    }

    var symbol: String {
        switch self {
        case .efficiency: "chart.line.uptrend.xyaxis"
        case .memory: "memorychip"
        case .latency: "timer"
        }
    }
}

enum PromptCacheLayout {
    static let compactWidth: CGFloat = 620
    static let wideWidth: CGFloat = 900
}
