//
//  WhipController.swift
//  tesseract
//
//  The whip's coordinator: it owns the simulation, the panel, the window-rect
//  world and the synthesiser, and drives all four from one display link.
//
//  Two behaviours here are load-bearing and worth stating plainly.
//
//  **It never eats a click.** The panel is click-through by default. It is made
//  interactive only while ⌥ is held *and* the cursor is within the handle's grab
//  radius — a small disc, for as long as a modifier is down. Every other click
//  on the Mac goes where it was aimed. The cursor still shoves the thong around
//  as it passes, because that is done by watching `NSEvent.mouseLocation` from
//  the tick rather than by receiving any event.
//
//  **It costs nothing at rest.** A real whip damps to a stop, and when this one
//  does the display link is invalidated, the audio engine is torn down and the
//  window-list cache is dropped. What remains is a 10 Hz timer reading the
//  cursor position. Coming near it, or holding ⌥, starts everything again.
//

import AppKit
import Foundation
import Observation
import QuartzCore

@MainActor
final class WhipController {

    // MARK: - Collaborators

    private let panel = WhipPanel()
    private let world = WhipCollisionWorld()
    private let audio = WhipAudio()

    private var simulation: WhipSimulation

    // MARK: - Tick state

    private var displayLink: CADisplayLink?
    private var idleTimer: Timer?
    private var observationTasks: [Task<Void, Never>] = []
    private var lastTickTime: CFTimeInterval = 0
    private var accumulator: CGFloat = 0

    /// Physics substep. Verlet's effective damping depends on the timestep, so a
    /// variable one makes the whip feel heavier whenever the machine is busy.
    /// 240 Hz is fixed and comfortably above any display refresh rate.
    private static let fixedStep: CGFloat = 1.0 / 240.0
    /// Ceiling on the time one frame may advance. Without it, a stall hands the
    /// solver a huge accumulated dt and the whip detonates.
    private static let maxFrameTime: CGFloat = 0.05

    private var lastCursor: CGPoint?
    private var isGrabbed = false
    private var isArmed = false

    // MARK: - Enablement

    private(set) var isEnabled = false

    /// Turns the whole feature on or off. Off tears everything down — there is
    /// no panel, no timer and no audio engine left behind.
    func setEnabled(_ enabled: Bool) {
        let allowed = enabled && !Self.reduceMotionEnabled
        if enabled && !allowed {
            // Worth a line: from the outside this looks exactly like the
            // setting not working, and this is the one place that says why.
            Log.whip.info("whip: switched on but suppressed — Reduce Motion is enabled")
        }
        guard allowed != isEnabled else { return }
        isEnabled = allowed

        if allowed {
            Log.whip.info("whip: enabled")
            panel.onCanvasChanged = { [weak self] canvas in self?.canvasChanged(to: canvas) }
            panel.show()
            configureCanvas()
            resetToRestingPlace()
            startTicking()
        } else {
            Log.whip.info("whip: disabled")
            stopTicking()
            idleTimer?.invalidate()
            idleTimer = nil
            audio.stop()
            panel.tearDown()
        }
    }

    /// Whether sounds are permitted, mirroring the app-wide Play Sounds
    /// preference so the whip is silent wherever dictation would be.
    func setSoundEnabled(_ enabled: Bool) {
        audio.isEnabled = enabled
        // Switching sound back on mid-swing should be audible at once rather
        // than at the next wake. `start()` no-ops if the engine is already up.
        if enabled && displayLink != nil { audio.start() }
    }

    /// Reduce Motion turns this off outright rather than slowing it down. The
    /// feature *is* motion; a still whip is not a gentler version of it, it is
    /// a black line lying on your screen for no reason.
    static var reduceMotionEnabled: Bool {
        NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
    }

    // MARK: - Init

    init() {
        simulation = WhipSimulation(origin: .zero)
    }

    // MARK: - Settings

    /// Follows the two settings this feature reads, now and whenever they
    /// change, using the same `Observations` idiom `AppBindings` uses for every
    /// other long-lived settings rule. The initial emission is load-bearing:
    /// enabled-at-launch is what puts the whip on screen at launch.
    func bind(to settings: SettingsManager) {
        for task in observationTasks { task.cancel() }
        observationTasks = [
            Task { [weak self] in
                for await sounds in Observations({ settings.playSounds }) {
                    self?.setSoundEnabled(sounds)
                }
            },
            Task { [weak self] in
                for await enabled in Observations({ settings.whipEnabled }) {
                    self?.setEnabled(enabled)
                }
            },
        ]
    }

    // MARK: - Canvas

    private func configureCanvas() {
        let canvas = panel.canvasFrame
        world.canvasOrigin = canvas.origin
        world.canvasBounds = CGRect(origin: .zero, size: canvas.size)
        if let number = panel.panel?.windowNumber {
            world.excludedWindowNumbers = [number]
        }
        world.invalidate()
    }

    private func canvasChanged(to canvas: CGRect) {
        configureCanvas()
        // A display was unplugged out from under the whip — put it back
        // somewhere it can be seen rather than leaving it in dead space.
        if !world.canvasBounds.insetBy(dx: -40, dy: -40).contains(simulation.gripPoint) {
            resetToRestingPlace()
        }
        wake()
    }

    /// Hangs the whip from near the top of the display the cursor is on, so it
    /// falls into view and drapes over whatever is there.
    private func resetToRestingPlace() {
        let canvas = panel.canvasFrame
        guard !canvas.isEmpty else { return }
        let screen = OverlayScreenLocator.preferredScreen()?.frame ?? canvas
        let local = CGPoint(
            x: screen.midX - canvas.origin.x,
            y: screen.maxY - canvas.origin.y - 140)
        simulation = WhipSimulation(origin: local)
        simulation.setGrabTarget(local)
    }

    // MARK: - Tick lifecycle

    private func startTicking() {
        guard displayLink == nil, let view = panel.renderView else { return }
        idleTimer?.invalidate()
        idleTimer = nil

        lastTickTime = CACurrentMediaTime()
        accumulator = 0

        // Deliberately the *screen's* display link, not the view's. A view-bound
        // link is gated on that view being visible, and this panel starts fully
        // transparent with nothing drawn in it — the window server treats it as
        // occluded, pauses the link, and it never gets a frame in which to draw
        // itself into existence. The screen's link has no such dependency.
        let screen = panel.panel?.screen ?? NSScreen.main
        let link =
            screen?.displayLink(target: self, selector: #selector(tick(_:)))
            ?? view.displayLink(target: self, selector: #selector(tick(_:)))
        link.add(to: .main, forMode: .common)
        displayLink = link

        view.onGrab = { [weak self] in self?.beginGrab() }
        view.onRelease = { [weak self] in self?.endGrab() }

        audio.start()
    }

    private func stopTicking() {
        displayLink?.invalidate()
        displayLink = nil
        audio.stop()
        world.invalidate()
    }

    /// The sleep state: everything expensive is gone and a 10 Hz timer watches
    /// for a reason to come back.
    private func enterIdle() {
        guard isEnabled else { return }
        stopTicking()
        guard idleTimer == nil else { return }

        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.pollWhileIdle() }
        }
        RunLoop.main.add(timer, forMode: .common)
        idleTimer = timer
    }

    private func pollWhileIdle() {
        guard isEnabled else { return }
        let cursor = localCursor()
        let option = NSEvent.modifierFlags.contains(.option)

        // Wake for anything that could plausibly become an interaction: the
        // cursor closing on the whip, or ⌥ going down anywhere near it.
        let reach = simulation.parameters.cursorPushRadius + 24
        let nearThong = simulation.positions.contains {
            hypot($0.x - cursor.x, $0.y - cursor.y) < reach
        }
        let nearGrip =
            hypot(
                simulation.gripPoint.x - cursor.x, simulation.gripPoint.y - cursor.y)
            < simulation.parameters.grabRadius + 40

        guard nearThong || (option && nearGrip) else { return }
        wake()
    }

    private func wake() {
        guard isEnabled else { return }
        simulation.wake()
        startTicking()
    }

    // MARK: - Grab

    private func beginGrab() {
        guard isArmed || isGrabbed else { return }
        isGrabbed = true
        simulation.setGrabTarget(localCursor())
        simulation.beginGrab()
        wake()
    }

    private func endGrab() {
        guard isGrabbed else { return }
        isGrabbed = false
        simulation.endGrab()
    }

    // MARK: - The tick

    @objc private func tick(_ link: CADisplayLink) {
        guard isEnabled else { return }

        let now = CACurrentMediaTime()
        var elapsed = CGFloat(now - lastTickTime)
        lastTickTime = now
        if elapsed <= 0 { return }
        elapsed = min(elapsed, Self.maxFrameTime)

        let cursor = localCursor()
        let delta = CGVector(
            dx: cursor.x - (lastCursor?.x ?? cursor.x),
            dy: cursor.y - (lastCursor?.y ?? cursor.y))
        lastCursor = cursor

        updateGrabAffordance(cursor: cursor)

        if isGrabbed {
            simulation.setGrabTarget(cursor)
            // Safety net for a mouse-up we never received — the panel can go
            // click-through mid-drag if anything unexpected happens, and a whip
            // welded to the cursor forever would be a genuinely bad bug.
            if NSEvent.pressedMouseButtons & 0x1 == 0 { endGrab() }
        }

        let input = WhipInput(
            grabPoint: isGrabbed ? cursor : nil,
            cursor: cursor,
            cursorDelta: delta)

        let currentWorld = world.world(now: now)

        // Fixed-step accumulator. Several substeps per displayed frame is normal
        // and is what keeps the constraint solve stiff enough to crack.
        accumulator += elapsed
        var lastTick = WhipTick()
        var steps = 0
        while accumulator >= Self.fixedStep && steps < 16 {
            lastTick = simulation.step(
                dt: Self.fixedStep,
                input: steps == 0 ? input : WhipInput(grabPoint: input.grabPoint, cursor: cursor),
                world: currentWorld,
                time: now)
            accumulator -= Self.fixedStep
            steps += 1
        }
        if steps == 0 { return }

        audio.update(
            tipSpeed: lastTick.tipSpeed,
            cracked: lastTick.cracked,
            crackIntensity: lastTick.crackIntensity)
        if lastTick.cracked {
            Log.whip.debug("whip: crack at \(Int(lastTick.tipSpeed)) pt/s")
        }

        render()

        if lastTick.asleep && !isGrabbed && !isArmed {
            audio.silence()
            enterIdle()
        }
    }

    /// Decides, every frame, whether the panel should be grabbable. This is the
    /// whole click-safety story: interactive only while ⌥ is down and the cursor
    /// is on the handle, or while a grab is already in progress.
    private func updateGrabAffordance(cursor: CGPoint) {
        let option = NSEvent.modifierFlags.contains(.option)
        let distance = hypot(
            simulation.gripPoint.x - cursor.x, simulation.gripPoint.y - cursor.y)
        isArmed = option && distance <= simulation.parameters.grabRadius
        panel.setInteractive(isArmed || isGrabbed)
    }

    private func render() {
        let points = simulation.positions
        guard points.count > WhipSimulation.thongStartIndex else { return }

        var frame = WhipFrame()
        frame.thongPoints = Array(points[(WhipSimulation.handleTipIndex)...])
        frame.thongRadii = Array(simulation.radii[(WhipSimulation.handleTipIndex)...])
        frame.grip = points[WhipSimulation.gripIndex]
        frame.handleTip = points[WhipSimulation.handleTipIndex]
        frame.handleRadius = simulation.parameters.handleRadius
        frame.isArmed = isArmed
        frame.isGrabbed = isGrabbed
        panel.apply(frame)
    }

    // MARK: - Helpers

    /// The cursor in simulation space. `NSEvent.mouseLocation` is a static read
    /// with no monitor and no permission behind it — the same call
    /// `OverlayScreenLocator` already makes.
    private func localCursor() -> CGPoint {
        let global = NSEvent.mouseLocation
        let origin = panel.canvasFrame.origin
        return CGPoint(x: global.x - origin.x, y: global.y - origin.y)
    }
}
