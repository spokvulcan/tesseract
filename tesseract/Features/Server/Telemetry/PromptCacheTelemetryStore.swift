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
    /// The SSD endurance ledger's current counters + notable events
    /// (PRD #150), re-pulled on every snapshot refresh and sample tick.
    private(set) var endurance: SSDEnduranceSnapshot = .empty

    var isLive = true
    var searchText = ""
    var selectedPartitionID: String?
    var selectedNodeID: String?
    var selectedEventID: UUID?
    var visibleCheckpointTypes = PromptCacheTelemetryStore.defaultVisibleCheckpointTypes
    var visibleStorageStates = PromptCacheTelemetryStore.defaultVisibleStorageStates

    static let defaultVisibleCheckpointTypes: Set<String> = ["system", "leaf", "branchPoint"]
    /// Empty nodes are pure topology — path skeleton with no snapshot —
    /// and they dominate the node count, so they are hidden by default.
    static let defaultVisibleStorageStates =
        Set(PromptCacheStorageState.allCases).subtracting([.empty])

    @ObservationIgnored private var diagnosticsHandle: PrefixCacheDiagnostics.TelemetrySinkHandle?
    @ObservationIgnored private var pendingEvents: [PromptCacheTelemetryEvent] = []
    @ObservationIgnored private var flushScheduled = false
    @ObservationIgnored private var pollingTask: Task<Void, Never>?
    @ObservationIgnored private let enduranceAccumulator: SSDEnduranceAccumulator?

    init(
        registerDiagnosticsSink: Bool = true,
        enduranceAccumulator: SSDEnduranceAccumulator? = nil
    ) {
        self.enduranceAccumulator = enduranceAccumulator
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
        return snapshot.trees.map { filteredTree(from: $0, query: query) }
    }

    /// Filtering *contracts* the tree: a hidden node disappears and its
    /// visible descendants re-parent to their nearest visible ancestor,
    /// with a synthesized edge spanning the hidden run. Re-including the
    /// hidden ancestors instead would neuter the filters entirely — in a
    /// radix tree almost every empty node is an ancestor of some
    /// snapshot-bearing node. The root always survives as the anchor.
    private func filteredTree(
        from tree: PromptCacheTreeSnapshot,
        query: String
    ) -> PromptCacheTreeSnapshot {
        let nodesByID = Dictionary(uniqueKeysWithValues: tree.nodes.map { ($0.id, $0) })

        var visibleIDs: Set<String> = []
        for node in tree.nodes
        where node.parentID == nil || (checkpointAllowed(node) && storageAllowed(node)) {
            visibleIDs.insert(node.id)
        }

        // The search query narrows further, keeping each match's visible
        // ancestors so the path context survives.
        if !query.isEmpty {
            var kept: Set<String> = []
            for node in tree.nodes
            where
                node.parentID == nil
                || (visibleIDs.contains(node.id) && queryMatches(node, in: tree, query: query))
            {
                var current: PromptCacheTreeNodeSnapshot? = node
                while let ancestor = current {
                    if visibleIDs.contains(ancestor.id) { kept.insert(ancestor.id) }
                    current = ancestor.parentID.flatMap { nodesByID[$0] }
                }
            }
            visibleIDs = kept
        }

        func nearestVisibleAncestorID(of node: PromptCacheTreeNodeSnapshot) -> String? {
            var current = node.parentID.flatMap { nodesByID[$0] }
            while let ancestor = current {
                if visibleIDs.contains(ancestor.id) { return ancestor.id }
                current = ancestor.parentID.flatMap { nodesByID[$0] }
            }
            return nil
        }

        var nodes: [PromptCacheTreeNodeSnapshot] = []
        var edges: [PromptCacheTreeEdgeSnapshot] = []
        for var node in tree.nodes where visibleIDs.contains(node.id) {
            let parentID = nearestVisibleAncestorID(of: node)
            node.parentID = parentID
            nodes.append(node)
            if let parentID {
                let parentOffset = nodesByID[parentID]?.tokenOffset ?? 0
                edges.append(
                    PromptCacheTreeEdgeSnapshot(
                        id: "\(parentID)->\(node.id)",
                        parentID: parentID,
                        childID: node.id,
                        tokenCount: max(node.tokenOffset - parentOffset, 0)
                    ))
            }
        }

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

    var selectedTree: PromptCacheTreeSnapshot? {
        let trees = filteredTrees
        if let selectedPartitionID,
            let tree = trees.first(where: { $0.id == selectedPartitionID })
        {
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

    /// The most recent completed request's humanized `lookup` + `ttft`
    /// pair (PRD #150's per-request outcome line). The `ttft` event
    /// trails its `lookup` by the whole generation, so it is matched by
    /// request ID and may legitimately still be absent.
    var lastRequestOutcome: (lookup: PromptCacheTelemetryEvent, ttft: PromptCacheTelemetryEvent?)? {
        guard let lookup = events.last(where: { $0.eventName == "lookup" }) else {
            return nil
        }
        guard let requestID = lookup.requestID else { return (lookup, nil) }
        let ttft = events.last {
            $0.eventName == "ttft" && $0.requestID == requestID
        }
        return (lookup, ttft)
    }

    func startPolling(llmActor: LLMActor) {
        guard !isPolling else { return }
        isPolling = true
        pollingTask = Task { @MainActor [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                await self.refreshSnapshot(llmActor: llmActor)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    func stopPolling() {
        isPolling = false
        pollingTask?.cancel()
        pollingTask = nil
    }

    func refreshSnapshot(llmActor: LLMActor) async {
        snapshot = llmActor.prefixCacheAdmin.makeTelemetrySnapshot()
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

    func replaceSnapshotForTesting(_ snapshot: PromptCacheTelemetrySnapshot) {
        self.snapshot = snapshot
        normalizeSelection()
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

    /// Drop the event buffer and the session aggregate derived from it.
    /// Lifetime counters and the topology snapshot are untouched — they
    /// belong to the cache, not to this window's session.
    func clearEvents() {
        pendingEvents.removeAll()
        events.removeAll()
        selectedEventID = nil
        recalculateAggregate()
    }

    func resetFilters() {
        searchText = ""
        visibleCheckpointTypes = Self.defaultVisibleCheckpointTypes
        visibleStorageStates = Self.defaultVisibleStorageStates
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
        if let enduranceAccumulator {
            endurance = enduranceAccumulator.snapshot()
        }
        let current = snapshot
        let ssd = current?.ssd ?? .disabled
        metricSamples.append(
            PromptCacheMetricSample(
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

    /// Drop selections that no longer resolve — never invent new ones.
    /// `nil` partition means "first available" and must survive a refresh;
    /// `nil` node means "nothing selected" (the HUD shows the tree
    /// summary) and must not snap back to the root on the next poll.
    private func normalizeSelection() {
        if let selectedPartitionID,
            !filteredTrees.contains(where: { $0.id == selectedPartitionID })
        {
            self.selectedPartitionID = nil
        }
        if let selectedNodeID,
            selectedTree?.nodes.contains(where: { $0.id == selectedNodeID }) != true
        {
            self.selectedNodeID = nil
        }
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
            || node.snapshotRefID?.localizedCaseInsensitiveContains(query) == true
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
