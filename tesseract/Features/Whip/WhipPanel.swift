//
//  WhipPanel.swift
//  tesseract
//
//  The surface the whip is drawn on: one transparent, click-through panel
//  spanning every display.
//
//  Configuration mirrors `OverlayPanel`, which is the one window setup in this
//  app already proven to float over other applications and full-screen spaces.
//  The important property is `.nonactivatingPanel` — grabbing the whip must
//  never pull focus out of whatever you were typing in.
//
//  It stays click-through by default and is made interactive only for the
//  instant the grab affordance is showing, which is how the whip can hang over
//  every app all day without ever eating a click meant for something else.
//

import AppKit
import Combine
import Foundation

@MainActor
final class WhipPanel {

    private(set) var panel: NSPanel?
    private(set) var renderView: WhipRenderView?
    private var cancellables = Set<AnyCancellable>()

    /// Fires when the displays change, so the owner can rebuild the simulation
    /// bounds and recheck that the whip is still lying somewhere visible.
    var onCanvasChanged: ((CGRect) -> Void)?

    /// The panel's frame in Cocoa global coordinates. The simulation works in
    /// panel-local space, so this is also the offset between the two.
    private(set) var canvasFrame: CGRect = .zero

    // MARK: - Lifecycle

    func show() {
        if panel == nil { createPanel() }
        panel?.orderFrontRegardless()
        startScreenObservation()
    }

    func hide() {
        panel?.orderOut(nil)
    }

    func tearDown() {
        cancellables.removeAll()
        panel?.orderOut(nil)
        panel = nil
        renderView = nil
    }

    /// Flips between click-through and grabbable. Called from the tick, so it
    /// must be cheap and idempotent — AppKit no-ops an unchanged value, and the
    /// guard keeps us from touching the window server at 120 Hz regardless.
    func setInteractive(_ interactive: Bool) {
        guard let panel, panel.ignoresMouseEvents == interactive else { return }
        panel.ignoresMouseEvents = !interactive
    }

    func apply(_ frame: WhipFrame) {
        renderView?.apply(frame)
    }

    // MARK: - Construction

    private func createPanel() {
        let union = WhipCollisionWorld.displayUnion()
        let frame = union.isNull ? CGRect(x: 0, y: 0, width: 1440, height: 900) : union
        canvasFrame = frame

        let panel = NSPanel(
            contentRect: frame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false)

        panel.level = .statusBar
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        panel.isReleasedWhenClosed = false
        panel.ignoresMouseEvents = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        // No window shadow: the whip casts its own inside the content, shaped to
        // the whip rather than to a screen-sized rectangle.
        panel.hasShadow = false
        panel.hidesOnDeactivate = false

        // Mirrors `OverlayPanel` exactly, including adding the content as a
        // *subview* rather than replacing `contentView`. Deliberately no
        // `isFloatingPanel` and no `worksWhenModal`: setting `isFloatingPanel`
        // after `hidesOnDeactivate` silently turns hiding back on, and this
        // panel must survive Tesseract not being the active app — it hangs over
        // whatever you are actually using.
        let view = WhipRenderView(frame: panel.contentView?.bounds ?? .zero)
        view.autoresizingMask = [.width, .height]
        panel.contentView?.addSubview(view)

        self.panel = panel
        self.renderView = view
    }

    // MARK: - Screens

    private func startScreenObservation() {
        guard cancellables.isEmpty else { return }
        let sources: [(NotificationCenter, Notification.Name)] = [
            (.default, NSApplication.didChangeScreenParametersNotification),
            (NSWorkspace.shared.notificationCenter, NSWorkspace.activeSpaceDidChangeNotification),
            (NSWorkspace.shared.notificationCenter, NSWorkspace.didWakeNotification),
        ]
        for (center, name) in sources {
            center.publisher(for: name)
                .receive(on: DispatchQueue.main)
                .sink { [weak self] _ in self?.refreshCanvas() }
                .store(in: &cancellables)
        }
    }

    private func refreshCanvas() {
        guard let panel else { return }
        let union = WhipCollisionWorld.displayUnion()
        guard !union.isNull else { return }
        guard union != canvasFrame else {
            panel.orderFrontRegardless()
            return
        }
        canvasFrame = union
        panel.setFrame(union, display: false)
        panel.orderFrontRegardless()
        onCanvasChanged?(union)
    }
}
