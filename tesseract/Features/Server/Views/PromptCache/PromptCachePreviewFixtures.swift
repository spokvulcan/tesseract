#if DEBUG
import SwiftUI

enum PromptCachePreviewFixtures {
    static let empty = PromptCacheTelemetrySnapshot.empty

    static let ramOnly = makeSnapshot(
        name: "RAM",
        ramBytes: 86 * 1024 * 1024,
        ssdBytes: 0,
        pendingBytes: 0,
        tree: makeTree(
            partition: "ram",
            branches: 8,
            snapshotEvery: 2,
            storageState: .ramOnly
        )
    )

    static let ssdWarmStart = makeSnapshot(
        name: "SSD Warm",
        ramBytes: 24 * 1024 * 1024,
        ssdBytes: 420 * 1024 * 1024,
        pendingBytes: 0,
        tree: makeTree(
            partition: "ssd",
            branches: 10,
            snapshotEvery: 1,
            storageState: .ssdOnly
        )
    )

    static let evictionHeavy = makeSnapshot(
        name: "Evictions",
        ramBytes: 238 * 1024 * 1024,
        ssdBytes: 920 * 1024 * 1024,
        pendingBytes: 18 * 1024 * 1024,
        tree: makeTree(
            partition: "evict",
            branches: 14,
            snapshotEvery: 3,
            storageState: .ramAndSSD
        )
    )

    static let largeBranchy = makeSnapshot(
        name: "Large",
        ramBytes: 148 * 1024 * 1024,
        ssdBytes: 512 * 1024 * 1024,
        pendingBytes: 4 * 1024 * 1024,
        tree: makeTree(
            partition: "large",
            branches: 40,
            snapshotEvery: 4,
            storageState: .ramOnly
        )
    )

    static let events: [PromptCacheTelemetryEvent] = [
        event("lookup", fields: [
            ("reason", "hit"),
            ("promptTokens", "4096"),
            ("skippedPrefillTokens", "3072"),
            ("checkpointType", "system"),
            ("lookupMs", "4.2"),
            ("restoreMs", "1.8"),
        ]),
        event("lookup", fields: [
            ("reason", "ssdHit"),
            ("promptTokens", "6144"),
            ("skippedPrefillTokens", "4096"),
            ("checkpointType", "leaf"),
            ("lookupMs", "11.0"),
            ("restoreMs", "9.5"),
        ]),
        event("eviction", fields: [
            ("strategy", "utility"),
            ("offset", "6144"),
            ("checkpointType", "leaf"),
            ("freedBytes", "12582912"),
            ("utility", "0.374"),
        ]),
        event("ssdAdmit", fields: [
            ("id", "preview-snap-12"),
            ("bytes", "12582912"),
            ("outcome", "accepted"),
        ]),
        event("ttft", fields: [
            ("lookupMs", "8.4"),
            ("restoreMs", "4.1"),
            ("prefillMs", "71.2"),
            ("totalPromptMs", "84.9"),
        ]),
    ]

    static var samples: [PromptCacheMetricSample] {
        var result: [PromptCacheMetricSample] = []
        for index in 0..<18 {
            let date = Date(timeIntervalSinceNow: Double(index - 18) * 10)
            let hitRate = Swift.min(0.35 + Double(index) * 0.028, 0.92)
            let tokenReuseRate = Swift.min(0.28 + Double(index) * 0.031, 0.88)
            let ramBytes = (42 + index * 7) * 1024 * 1024
            let ssdBytes = (180 + index * 24) * 1024 * 1024
            result.append(
                PromptCacheMetricSample(
                    date: date,
                    hitRate: hitRate,
                    tokenReuseRate: tokenReuseRate,
                    ramBytes: ramBytes,
                    ramBudgetBytes: 256 * 1024 * 1024,
                    ssdBytes: ssdBytes,
                    ssdBudgetBytes: 1024 * 1024 * 1024,
                    evictionCount: index / 4,
                    admissionCount: index / 2,
                    averageLookupMs: 4 + Double(index % 5),
                    averageRestoreMs: 1.5 + Double(index % 4),
                    averagePrefillMs: 42 + Double(index * 2)
                )
            )
        }
        return result
    }

    private static func makeSnapshot(
        name: String,
        ramBytes: Int,
        ssdBytes: Int,
        pendingBytes: Int,
        tree: PromptCacheTreeSnapshot
    ) -> PromptCacheTelemetrySnapshot {
        PromptCacheTelemetrySnapshot(
            capturedAt: Date(),
            memoryBudgetBytes: 256 * 1024 * 1024,
            residentSnapshotBytes: ramBytes,
            partitionCount: 1,
            totalNodeCount: tree.nodeCount,
            snapshotCount: tree.snapshotCount,
            snapshotsByType: tree.snapshotsByType,
            ssd: PromptCacheSSDSnapshot(
                enabled: ssdBytes > 0 || pendingBytes > 0,
                rootPath: "/tmp/tesseract-prefix-cache/\(name)",
                budgetBytes: 1024 * 1024 * 1024,
                currentBytes: ssdBytes,
                pendingBytes: pendingBytes,
                maxPendingBytes: 128 * 1024 * 1024,
                pendingCount: pendingBytes > 0 ? 2 : 0,
                snapshotCount: tree.snapshotCount,
                partitionCount: 1
            ),
            trees: [tree]
        )
    }

    private static func makeTree(
        partition: String,
        branches: Int,
        snapshotEvery: Int,
        storageState: PromptCacheStorageState
    ) -> PromptCacheTreeSnapshot {
        let digest = "preview-\(partition)"
        let rootID = "\(digest):root"
        var nodes = [
            node(
                id: rootID,
                parentID: nil,
                pathHash: "root",
                offset: 0,
                depth: 0,
                childCount: branches,
                state: .empty
            ),
        ]
        var edges: [PromptCacheTreeEdgeSnapshot] = []
        var snapshotsByType: [String: Int] = [:]

        for branch in 0..<branches {
            let midID = "\(digest):b\(branch)"
            let leafID = "\(digest):b\(branch)-leaf"
            let checkpoint = checkpointType(for: branch)
            let hasSnapshot = branch.isMultiple(of: snapshotEvery)
            let branchState = hasSnapshot ? storageState : .empty

            nodes.append(node(
                id: midID,
                parentID: rootID,
                pathHash: "b\(branch)",
                offset: 1024 + branch * 256,
                depth: 1,
                childCount: 1,
                checkpointType: hasSnapshot ? checkpoint : nil,
                bytes: hasSnapshot ? 8 * 1024 * 1024 : 0,
                state: branchState,
                storageRefID: branchState == .empty ? nil : "preview-ref-\(branch)"
            ))
            nodes.append(node(
                id: leafID,
                parentID: midID,
                pathHash: "b\(branch)-leaf",
                offset: 2048 + branch * 512,
                depth: 2,
                childCount: 0,
                checkpointType: "leaf",
                bytes: 12 * 1024 * 1024,
                state: storageState,
                storageRefID: "preview-leaf-\(branch)"
            ))
            edges.append(PromptCacheTreeEdgeSnapshot(
                id: "\(rootID)->\(midID)",
                parentID: rootID,
                childID: midID,
                tokenCount: 1024 + branch * 12
            ))
            edges.append(PromptCacheTreeEdgeSnapshot(
                id: "\(midID)->\(leafID)",
                parentID: midID,
                childID: leafID,
                tokenCount: 256 + branch * 8
            ))

            if hasSnapshot {
                snapshotsByType[checkpoint, default: 0] += 1
            }
            snapshotsByType["leaf", default: 0] += 1
        }

        return PromptCacheTreeSnapshot(
            id: digest,
            partitionDigest: digest,
            partitionSummary: "preview-\(partition) · denseKV/g64 · dense · nofp",
            nodeCount: nodes.count,
            totalSnapshotBytes: nodes.reduce(0) { $0 + $1.snapshotBytes },
            snapshotCount: snapshotsByType.values.reduce(0, +),
            snapshotsByType: snapshotsByType,
            nodes: nodes,
            edges: edges
        )
    }

    private static func node(
        id: String,
        parentID: String?,
        pathHash: String,
        offset: Int,
        depth: Int,
        childCount: Int,
        checkpointType: String? = nil,
        bytes: Int = 0,
        state: PromptCacheStorageState,
        storageRefID: String? = nil
    ) -> PromptCacheTreeNodeSnapshot {
        PromptCacheTreeNodeSnapshot(
            id: id,
            parentID: parentID,
            pathHash: pathHash,
            tokenOffset: offset,
            pathTokenCount: offset,
            edgeTokenCount: max(offset / max(depth, 1), 0),
            childCount: childCount,
            depth: depth,
            hasSnapshot: bytes > 0,
            checkpointType: checkpointType,
            snapshotBytes: bytes,
            storageState: state,
            storageRefID: storageRefID,
            storageBytes: storageRefID == nil ? 0 : max(bytes, 4 * 1024 * 1024),
            lastAccessAgeSeconds: Double(depth * 12 + offset % 37),
            normalizedRecency: checkpointType == "system" ? nil : 0.42,
            normalizedFlopEfficiency: checkpointType == "system" ? nil : 0.68,
            utility: checkpointType == "system" ? nil : 0.55
        )
    }

    private static func checkpointType(for index: Int) -> String {
        switch index % 3 {
        case 0: "system"
        case 1: "branchPoint"
        default: "leaf"
        }
    }

    private static func event(
        _ name: String,
        fields: [(String, String)]
    ) -> PromptCacheTelemetryEvent {
        PromptCacheTelemetryEvent(
            scope: .request,
            eventName: name,
            requestID: UUID(uuidString: "22222222-2222-2222-2222-222222222222")!,
            modelID: "preview-model",
            kvBits: 8,
            kvGroupSize: 64,
            fields: fields
        )
    }
}

private struct PromptCacheAdaptivePreview: View {
    enum Surface {
        case overview
        case tree
        case events
        case inspector
    }

    let snapshot: PromptCacheTelemetrySnapshot
    let surface: Surface
    @State private var selectedNodeID: String?
    @State private var store = PromptCacheTelemetryStore(registerDiagnosticsSink: false)

    private var tree: PromptCacheTreeSnapshot? { snapshot.trees.first }
    private var selectedNode: PromptCacheTreeNodeSnapshot? {
        guard let selectedNodeID else { return tree?.nodes.first { $0.parentID == nil } }
        return tree?.nodes.first { $0.id == selectedNodeID }
    }

    var body: some View {
        Group {
            switch surface {
            case .overview:
                PromptCacheOverviewView(
                    snapshot: snapshot,
                    aggregate: PromptCacheTelemetryAggregate.from(events: PromptCachePreviewFixtures.events),
                    samples: PromptCachePreviewFixtures.samples
                )

            case .tree:
                VStack(spacing: Theme.Spacing.xs) {
                    PromptCacheTreeCanvasView(
                        tree: tree,
                        selectedNodeID: selectedNode?.id,
                        onSelectNode: { selectedNodeID = $0 }
                    )
                    PromptCacheSelectionStatusView(tree: tree, node: selectedNode)
                }

            case .events:
                PromptCacheEventTableView(store: store)
                    .onAppear {
                        if store.events.isEmpty {
                            store.recordForTesting(PromptCachePreviewFixtures.events)
                        }
                    }

            case .inspector:
                ScrollView {
                    PromptCacheInspectorView(
                        tree: tree,
                        node: selectedNode,
                        event: PromptCachePreviewFixtures.events.first
                    )
                }
            }
        }
        .padding(Theme.Spacing.sm)
    }
}

#Preview("Prompt Cache 520x380 Overview") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.evictionHeavy,
        surface: .overview
    )
    .frame(width: 520, height: 380)
}

#Preview("Prompt Cache 520x380 Tree") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.largeBranchy,
        surface: .tree
    )
    .frame(width: 520, height: 380)
}

#Preview("Prompt Cache 640x480 Events") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.ramOnly,
        surface: .events
    )
    .frame(width: 640, height: 480)
}

#Preview("Prompt Cache 900x620 Inspector Popover") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.ssdWarmStart,
        surface: .inspector
    )
    .frame(width: 900, height: 620)
}

#Preview("Prompt Cache Wide Tree") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.largeBranchy,
        surface: .tree
    )
    .frame(width: 1200, height: 760)
}

#Preview("Prompt Cache Empty") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.empty,
        surface: .overview
    )
}

#Preview("Prompt Cache RAM") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.ramOnly,
        surface: .overview
    )
}

#Preview("Prompt Cache SSD Warm Start") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.ssdWarmStart,
        surface: .inspector
    )
}

#Preview("Prompt Cache Eviction Heavy") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.evictionHeavy,
        surface: .overview
    )
}

#Preview("Prompt Cache Large Branchy") {
    PromptCacheAdaptivePreview(
        snapshot: PromptCachePreviewFixtures.largeBranchy,
        surface: .tree
    )
}
#endif
