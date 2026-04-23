import AppKit
import Foundation
import Observation

@MainActor
@Observable
final class PromptCacheTelemetryStore {
    static let maxEvents = 1_000
    static let maxSamples = 360
    static let flushIntervalNanoseconds: UInt64 = 125_000_000

    private(set) var snapshot: PromptCacheTelemetrySnapshot?
    private(set) var events: [PromptCacheTelemetryEvent] = []
    private(set) var aggregate = PromptCacheTelemetryAggregate()
    private(set) var metricSamples: [PromptCacheMetricSample] = []
    private(set) var lastRefreshError: String?
    private(set) var isPolling = false

    var isLive = true
    var searchText = ""
    var selectedPartitionID: String?
    var selectedNodeID: String?
    var selectedEventID: UUID?
    var visibleCheckpointTypes: Set<String> = ["system", "leaf", "branchPoint"]
    var visibleStorageStates: Set<PromptCacheStorageState> = Set(PromptCacheStorageState.allCases)

    @ObservationIgnored private var diagnosticsHandle: PrefixCacheDiagnostics.TelemetrySinkHandle?
    @ObservationIgnored private var pendingEvents: [PromptCacheTelemetryEvent] = []
    @ObservationIgnored private var flushScheduled = false
    @ObservationIgnored private var pollingTask: Task<Void, Never>?

    init(registerDiagnosticsSink: Bool = true) {
        if registerDiagnosticsSink {
            diagnosticsHandle = PrefixCacheDiagnostics.addTelemetrySink { [weak self] event in
                Task { @MainActor [weak self] in
                    self?.enqueue(event)
                }
            }
        }
    }

    deinit {
        if let diagnosticsHandle {
            PrefixCacheDiagnostics.removeTelemetrySink(diagnosticsHandle)
        }
        pollingTask?.cancel()
    }

    var filteredEvents: [PromptCacheTelemetryEvent] {
        let query = normalizedQuery
        guard !query.isEmpty else { return events.reversed() }
        return events.reversed().filter { event in
            event.eventName.localizedCaseInsensitiveContains(query)
                || event.fields.contains {
                    $0.key.localizedCaseInsensitiveContains(query)
                        || $0.value.localizedCaseInsensitiveContains(query)
                }
                || event.requestID?.uuidString.localizedCaseInsensitiveContains(query) == true
                || event.modelID?.localizedCaseInsensitiveContains(query) == true
        }
    }

    var filteredTrees: [PromptCacheTreeSnapshot] {
        guard let snapshot else { return [] }
        let query = normalizedQuery
        return snapshot.trees.map { tree in
            let nodesByID = Dictionary(uniqueKeysWithValues: tree.nodes.map { ($0.id, $0) })
            var includedNodeIDs: Set<String> = []

            for node in tree.nodes where
                checkpointAllowed(node) && storageAllowed(node) && queryMatches(node, in: tree, query: query)
            {
                var current: PromptCacheTreeNodeSnapshot? = node
                while let ancestor = current {
                    includedNodeIDs.insert(ancestor.id)
                    current = ancestor.parentID.flatMap { nodesByID[$0] }
                }
            }

            let nodes = tree.nodes.filter { includedNodeIDs.contains($0.id) }
            let nodeIDs = Set(nodes.map(\.id))
            let edges = tree.edges.filter { nodeIDs.contains($0.parentID) && nodeIDs.contains($0.childID) }
            return PromptCacheTreeSnapshot(
                id: tree.id,
                partitionDigest: tree.partitionDigest,
                partitionSummary: tree.partitionSummary,
                nodeCount: tree.nodeCount,
                totalSnapshotBytes: tree.totalSnapshotBytes,
                snapshotCount: tree.snapshotCount,
                snapshotsByType: tree.snapshotsByType,
                nodes: nodes,
                edges: edges
            )
        }
    }

    var selectedTree: PromptCacheTreeSnapshot? {
        let trees = filteredTrees
        if let selectedPartitionID,
           let tree = trees.first(where: { $0.id == selectedPartitionID }) {
            return tree
        }
        return trees.first
    }

    var selectedNode: PromptCacheTreeNodeSnapshot? {
        guard let selectedNodeID else { return nil }
        return selectedTree?.nodes.first { $0.id == selectedNodeID }
    }

    var selectedEvent: PromptCacheTelemetryEvent? {
        guard let selectedEventID else { return nil }
        return events.first { $0.id == selectedEventID }
    }

    func startPolling(agentEngine: AgentEngine) {
        guard !isPolling else { return }
        isPolling = true
        pollingTask = Task { @MainActor [weak self, weak agentEngine] in
            guard let self else { return }
            while !Task.isCancelled {
                if let agentEngine {
                    await self.refreshSnapshot(agentEngine: agentEngine)
                }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    func stopPolling() {
        isPolling = false
        pollingTask?.cancel()
        pollingTask = nil
    }

    func refreshSnapshot(agentEngine: AgentEngine) async {
        snapshot = await agentEngine.promptCacheTelemetrySnapshot()
        lastRefreshError = nil
        appendSample()
        normalizeSelection()
    }

    func selectNode(_ nodeID: String?) {
        selectedNodeID = nodeID
    }

    func toggleLiveUpdates() {
        isLive.toggle()
        if isLive {
            flushPendingEvents()
        }
    }

    func recordForTesting(_ incoming: [PromptCacheTelemetryEvent]) {
        events.append(contentsOf: incoming)
        trimEvents()
        recalculateAggregate()
        appendSample()
    }

    func copyExportJSONToPasteboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(exportJSONString(), forType: .string)
    }

    func copySelectedEventToPasteboard() {
        guard let selectedEvent,
              let data = try? JSONEncoder.promptCachePretty.encode(selectedEvent),
              let text = String(data: data, encoding: .utf8)
        else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    func exportJSONString() -> String {
        let payload = PromptCacheExportPayload(
            exportedAt: Date(),
            snapshot: snapshot,
            aggregate: aggregate,
            events: events
        )
        guard let data = try? JSONEncoder.promptCachePretty.encode(payload),
              let text = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return text
    }

    func resetFilters() {
        searchText = ""
        visibleCheckpointTypes = ["system", "leaf", "branchPoint"]
        visibleStorageStates = Set(PromptCacheStorageState.allCases)
    }

    private var normalizedQuery: String {
        searchText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func enqueue(_ event: PromptCacheTelemetryEvent) {
        pendingEvents.append(event)
        if pendingEvents.count > Self.maxEvents {
            pendingEvents.removeFirst(pendingEvents.count - Self.maxEvents)
        }
        guard isLive else { return }
        guard !flushScheduled else { return }
        flushScheduled = true
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: Self.flushIntervalNanoseconds)
            self?.flushPendingEvents()
        }
    }

    private func flushPendingEvents() {
        flushScheduled = false
        guard isLive else { return }
        let incoming = pendingEvents
        pendingEvents.removeAll(keepingCapacity: true)
        guard !incoming.isEmpty else { return }

        events.append(contentsOf: incoming)
        trimEvents()
        recalculateAggregate()
        appendSample()
    }

    private func trimEvents() {
        if events.count > Self.maxEvents {
            events.removeFirst(events.count - Self.maxEvents)
        }
    }

    private func recalculateAggregate() {
        aggregate = PromptCacheTelemetryAggregate.from(events: events)
    }

    private func appendSample() {
        let current = snapshot
        let ssd = current?.ssd ?? .disabled
        metricSamples.append(PromptCacheMetricSample(
            hitRate: aggregate.hitRate,
            tokenReuseRate: aggregate.tokenReuseRate,
            ramBytes: current?.residentSnapshotBytes ?? 0,
            ramBudgetBytes: current?.memoryBudgetBytes ?? 0,
            ssdBytes: ssd.currentBytes,
            ssdBudgetBytes: ssd.budgetBytes,
            evictionCount: aggregate.evictionCount,
            admissionCount: aggregate.admissionCount,
            averageLookupMs: aggregate.averageLookupMs,
            averageRestoreMs: aggregate.averageRestoreMs,
            averagePrefillMs: aggregate.averagePrefillMs
        ))
        if metricSamples.count > Self.maxSamples {
            metricSamples.removeFirst(metricSamples.count - Self.maxSamples)
        }
    }

    private func normalizeSelection() {
        let trees = filteredTrees
        if selectedPartitionID == nil || !trees.contains(where: { $0.id == selectedPartitionID }) {
            selectedPartitionID = trees.first?.id
            selectedNodeID = nil
        }

        guard let selectedTree else {
            selectedNodeID = nil
            return
        }
        if let selectedNodeID,
           selectedTree.nodes.contains(where: { $0.id == selectedNodeID }) {
            return
        }
        selectedNodeID = selectedTree.nodes.first { $0.parentID == nil }?.id ?? selectedTree.nodes.first?.id
    }

    private func checkpointAllowed(_ node: PromptCacheTreeNodeSnapshot) -> Bool {
        guard let checkpointType = node.checkpointType else { return true }
        return visibleCheckpointTypes.contains(checkpointType)
    }

    private func storageAllowed(_ node: PromptCacheTreeNodeSnapshot) -> Bool {
        visibleStorageStates.contains(node.storageState)
    }

    private func queryMatches(
        _ node: PromptCacheTreeNodeSnapshot,
        in tree: PromptCacheTreeSnapshot,
        query: String
    ) -> Bool {
        guard !query.isEmpty else { return true }
        return node.pathHash.localizedCaseInsensitiveContains(query)
            || node.id.localizedCaseInsensitiveContains(query)
            || tree.partitionDigest.localizedCaseInsensitiveContains(query)
            || tree.partitionSummary.localizedCaseInsensitiveContains(query)
            || String(node.tokenOffset).contains(query)
            || node.checkpointType?.localizedCaseInsensitiveContains(query) == true
            || node.storageState.displayName.localizedCaseInsensitiveContains(query)
            || node.storageRefID?.localizedCaseInsensitiveContains(query) == true
    }
}

private extension JSONEncoder {
    static var promptCachePretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}
