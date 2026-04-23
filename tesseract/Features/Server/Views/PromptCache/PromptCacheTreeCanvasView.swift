import SwiftUI

struct PromptCacheTreeCanvasView: View {
    let tree: PromptCacheTreeSnapshot?
    let selectedNodeID: String?
    let onSelectNode: (String?) -> Void

    @State private var zoom: CGFloat = 1
    @State private var pan: CGSize = CGSize(width: 36, height: 36)
    @State private var panAtDragStart: CGSize?
    @State private var lastMagnification: CGFloat = 1

    var body: some View {
        GeometryReader { proxy in
            let compact = proxy.size.width < 560 || proxy.size.height < 420
            ZStack(alignment: .topTrailing) {
                if let tree, !tree.nodes.isEmpty {
                    Canvas { context, _ in
                        draw(tree: tree, context: &context)
                    }
                    .background(.regularMaterial)
                    .gesture(dragGesture)
                    .simultaneousGesture(magnificationGesture)
                    .simultaneousGesture(tapGesture(in: tree))
                    .accessibilityLabel("Interactive radix tree with \(tree.nodes.count) nodes")

                    controls(tree: tree, size: proxy.size, compact: compact)
                        .padding(Theme.Spacing.sm)

                    if !compact {
                        minimap(tree: tree)
                            .frame(width: 128, height: 84)
                            .padding(Theme.Spacing.sm)
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                    }
                } else {
                    VStack(spacing: Theme.Spacing.sm) {
                        Image(systemName: "point.3.connected.trianglepath.dotted")
                            .font(.system(size: compact ? 24 : 34))
                            .foregroundStyle(.tertiary)
                        Text("No prefix-cache topology")
                            .font(.body)
                            .foregroundStyle(.secondary)
                        Text("Run an HTTP text completion to instantiate the radix tree.")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .lineLimit(2)
                            .multilineTextAlignment(.center)
                    }
                    .padding(Theme.Spacing.md)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.regularMaterial)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous))
        }
    }

    private var dragGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                if panAtDragStart == nil {
                    panAtDragStart = pan
                }
                let start = panAtDragStart ?? pan
                pan = CGSize(
                    width: start.width + value.translation.width,
                    height: start.height + value.translation.height
                )
            }
            .onEnded { _ in
                panAtDragStart = nil
            }
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                let delta = value / lastMagnification
                zoom = (zoom * delta).clamped(to: 0.2...3.5)
                lastMagnification = value
            }
            .onEnded { _ in
                lastMagnification = 1
            }
    }

    private func tapGesture(in tree: PromptCacheTreeSnapshot) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                let layout = PromptCacheTreeLayout(tree: tree)
                let world = inverseScreenPoint(value.location)
                let nearest = layout.positions.min { lhs, rhs in
                    lhs.value.distance(to: world) < rhs.value.distance(to: world)
                }
                guard let nearest, nearest.value.distance(to: world) < max(18 / zoom, 8) else {
                    onSelectNode(nil)
                    return
                }
                onSelectNode(nearest.key)
            }
    }

    private func controls(
        tree: PromptCacheTreeSnapshot,
        size: CGSize,
        compact: Bool
    ) -> some View {
        GlassEffectContainer(spacing: 8) {
            HStack(spacing: 8) {
                if !compact {
                    Button {
                        zoom = (zoom / 1.2).clamped(to: 0.2...3.5)
                    } label: {
                        Image(systemName: "minus.magnifyingglass")
                    }
                    .buttonStyle(.glass)
                    .help("Zoom out")
                }

                Button {
                    fit(tree: tree, size: size)
                } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                }
                .buttonStyle(.glass)
                .help("Fit tree")

                if compact {
                    Button {
                        resetViewport()
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                    }
                    .buttonStyle(.glass)
                    .help("Reset view")
                } else {
                    Button {
                        zoom = (zoom * 1.2).clamped(to: 0.2...3.5)
                    } label: {
                        Image(systemName: "plus.magnifyingglass")
                    }
                    .buttonStyle(.glass)
                    .help("Zoom in")
                }
            }
        }
    }

    private func minimap(tree: PromptCacheTreeSnapshot) -> some View {
        Canvas { context, size in
            let layout = PromptCacheTreeLayout(tree: tree)
            let bounds = layout.bounds.insetBy(dx: -30, dy: -30)
            let sx = size.width / max(bounds.width, 1)
            let sy = size.height / max(bounds.height, 1)
            let scale = min(sx, sy)
            func mini(_ point: CGPoint) -> CGPoint {
                CGPoint(
                    x: (point.x - bounds.minX) * scale,
                    y: (point.y - bounds.minY) * scale
                )
            }
            for edge in tree.edges {
                guard let a = layout.positions[edge.parentID],
                      let b = layout.positions[edge.childID]
                else { continue }
                var path = Path()
                path.move(to: mini(a))
                path.addLine(to: mini(b))
                context.stroke(path, with: .color(.secondary.opacity(0.35)), lineWidth: 1)
            }
            for node in tree.nodes {
                guard let point = layout.positions[node.id] else { continue }
                let rect = CGRect(center: mini(point), radius: selectedNodeID == node.id ? 3 : 2)
                context.fill(Path(ellipseIn: rect), with: .color(color(for: node).opacity(0.75)))
            }
        }
        .padding(6)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: Theme.Radius.small, style: .continuous))
    }

    private func draw(tree: PromptCacheTreeSnapshot, context: inout GraphicsContext) {
        let layout = PromptCacheTreeLayout(tree: tree)

        for edge in tree.edges {
            guard let from = layout.positions[edge.parentID],
                  let to = layout.positions[edge.childID]
            else { continue }
            let a = screenPoint(from)
            let b = screenPoint(to)
            var path = Path()
            path.move(to: a)
            path.addCurve(
                to: b,
                control1: CGPoint(x: a.x + 48 * zoom, y: a.y),
                control2: CGPoint(x: b.x - 48 * zoom, y: b.y)
            )
            context.stroke(
                path,
                with: .color(.secondary.opacity(edge.tokenCount > 1 ? 0.32 : 0.18)),
                lineWidth: max(1, min(3, CGFloat(edge.tokenCount) / 96) * zoom)
            )
        }

        for node in tree.nodes {
            guard let point = layout.positions[node.id] else { continue }
            let p = screenPoint(point)
            let radius: CGFloat = node.parentID == nil ? 7 : node.hasSnapshot ? 8 : 5
            let rect = CGRect(center: p, radius: radius)
            let selected = node.id == selectedNodeID
            context.fill(Path(ellipseIn: rect), with: .color(color(for: node)))
            context.stroke(
                Path(ellipseIn: rect.insetBy(dx: -2, dy: -2)),
                with: .color(selected ? .accentColor : .clear),
                lineWidth: selected ? 2 : 0
            )

            if zoom > 0.55 {
                let label = node.parentID == nil
                    ? "root"
                    : "\(node.tokenOffset)"
                context.draw(
                    Text(label)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary),
                    at: CGPoint(x: p.x + 18, y: p.y),
                    anchor: .leading
                )
            }
        }
    }

    private func fit(tree: PromptCacheTreeSnapshot, size: CGSize) {
        let layout = PromptCacheTreeLayout(tree: tree)
        let bounds = layout.bounds.insetBy(dx: -80, dy: -80)
        let sx = size.width / max(bounds.width, 1)
        let sy = size.height / max(bounds.height, 1)
        zoom = min(max(min(sx, sy), 0.2), 2.2)
        pan = CGSize(
            width: size.width / 2 - bounds.midX * zoom,
            height: size.height / 2 - bounds.midY * zoom
        )
    }

    private func resetViewport() {
        zoom = 1
        pan = CGSize(width: 36, height: 36)
    }

    private func screenPoint(_ point: CGPoint) -> CGPoint {
        CGPoint(x: point.x * zoom + pan.width, y: point.y * zoom + pan.height)
    }

    private func inverseScreenPoint(_ point: CGPoint) -> CGPoint {
        CGPoint(x: (point.x - pan.width) / zoom, y: (point.y - pan.height) / zoom)
    }

    private func color(for node: PromptCacheTreeNodeSnapshot) -> Color {
        if node.parentID == nil { return .secondary }
        if node.storageState == .ssdOnly || node.storageState == .pendingWriteBodyDropped { return .teal }
        switch node.checkpointType {
        case "system": return .indigo
        case "leaf": return .green
        case "branchPoint": return .orange
        default: return node.hasSnapshot ? .blue : .gray.opacity(0.45)
        }
    }
}

private struct PromptCacheTreeLayout {
    let positions: [String: CGPoint]
    let bounds: CGRect

    init(tree: PromptCacheTreeSnapshot) {
        var children: [String: [PromptCacheTreeNodeSnapshot]] = [:]
        for node in tree.nodes {
            if let parentID = node.parentID {
                children[parentID, default: []].append(node)
            }
        }
        for key in children.keys {
            children[key]?.sort {
                if $0.tokenOffset != $1.tokenOffset { return $0.tokenOffset < $1.tokenOffset }
                return $0.id < $1.id
            }
        }

        let roots = tree.nodes.filter { $0.parentID == nil }
        var row = 0
        var positions: [String: CGPoint] = [:]

        func walk(_ node: PromptCacheTreeNodeSnapshot, depth: Int) {
            let descendants = children[node.id] ?? []
            if descendants.isEmpty {
                positions[node.id] = CGPoint(x: CGFloat(depth) * 132, y: CGFloat(row) * 34)
                row += 1
                return
            }
            let start = row
            for child in descendants {
                walk(child, depth: depth + 1)
            }
            let end = max(start, row - 1)
            positions[node.id] = CGPoint(
                x: CGFloat(depth) * 132,
                y: CGFloat(start + end) * 17
            )
        }

        for root in roots {
            walk(root, depth: 0)
        }

        self.positions = positions
        self.bounds = positions.values.reduce(CGRect.null) { rect, point in
            rect.union(CGRect(x: point.x, y: point.y, width: 1, height: 1))
        }
    }
}

private extension CGFloat {
    func clamped(to range: ClosedRange<CGFloat>) -> CGFloat {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}

private extension CGPoint {
    func distance(to other: CGPoint) -> CGFloat {
        hypot(x - other.x, y - other.y)
    }
}

private extension CGRect {
    init(center: CGPoint, radius: CGFloat) {
        self.init(
            x: center.x - radius,
            y: center.y - radius,
            width: radius * 2,
            height: radius * 2
        )
    }
}
