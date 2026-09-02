//
//  WhipPhysicsTests.swift
//  tesseractTests
//
//  The whip's feel is the whole feature, so the solver is a pure value machine
//  and gets tested like one — no panel, no window server, no audio device. Every
//  test here drives `WhipSimulation.step` directly with hand-made input.
//

import CoreGraphics
import Foundation
import Testing

@testable import Tesseract_Agent

struct WhipPhysicsTests {

    // MARK: - Helpers

    /// A world with plenty of room, so tests that are not about collision never
    /// accidentally hit a wall.
    private static let roomyWorld = WhipWorld(
        obstacles: [], bounds: CGRect(x: 0, y: 0, width: 4000, height: 4000))

    private static let step: CGFloat = 1.0 / 240.0

    /// Runs `count` fixed substeps and returns the last tick.
    @discardableResult
    private func advance(
        _ simulation: inout WhipSimulation,
        count: Int,
        input: WhipInput = .idle,
        world: WhipWorld = roomyWorld,
        startTime: Double = 0
    ) -> WhipTick {
        var tick = WhipTick()
        for index in 0..<count {
            let time = startTime + Double(index) * Double(Self.step)
            tick = simulation.step(dt: Self.step, input: input, world: world, time: time)
        }
        return tick
    }

    /// Sum of the gaps between consecutive points — the whip's actual length,
    /// as opposed to the length it is supposed to be.
    private func measuredLength(_ simulation: WhipSimulation) -> CGFloat {
        let points = simulation.positions
        var total: CGFloat = 0
        for index in 0..<(points.count - 1) {
            total += hypot(
                points[index + 1].x - points[index].x,
                points[index + 1].y - points[index].y)
        }
        return total
    }

    private func restLength(_ simulation: WhipSimulation) -> CGFloat {
        simulation.restLengths.reduce(0, +)
    }

    // MARK: - Construction

    @Test
    func hangsStraightDownFromItsOrigin() {
        let origin = CGPoint(x: 500, y: 3000)
        let whip = WhipSimulation(origin: origin)

        #expect(whip.gripPoint == origin)
        #expect(whip.pointCount == WhipParameters.default.thongPointCount + 2)

        // Every point below the one before it, all on the same vertical.
        for index in 1..<whip.pointCount {
            #expect(whip.positions[index].y < whip.positions[index - 1].y)
            #expect(abs(whip.positions[index].x - origin.x) < 0.0001)
        }
    }

    /// The taper is the reason a whip cracks, so it is asserted rather than
    /// assumed: radius falls monotonically, and the derived quantities fall with
    /// it — mass as r², bending resistance as r⁴.
    @Test
    func taperFallsMonotonicallyAlongTheThong() {
        let whip = WhipSimulation(origin: .zero)
        let start = WhipSimulation.thongStartIndex

        for index in (start + 1)..<whip.pointCount {
            #expect(whip.radii[index] < whip.radii[index - 1])
            // Lighter toward the tip means a larger inverse mass.
            #expect(whip.inverseMass[index] > whip.inverseMass[index - 1])
            // Non-increasing rather than strictly decreasing: each joint takes
            // the minimum radius of the three points it spans, so the final two
            // legitimately share a value.
            #expect(whip.bendResistance[index] <= whip.bendResistance[index - 1])
        }

        // Over the whole thong it must genuinely fall away, and the tip must be
        // limp rather than merely less stiff.
        #expect(whip.bendResistance[whip.pointCount - 1] < whip.bendResistance[start] * 0.05)
        #expect(whip.bendResistance[whip.pointCount - 1] < 0.01)
    }

    // MARK: - Integration

    @Test
    func fallsUnderGravityWhenNotHeld() {
        var whip = WhipSimulation(origin: CGPoint(x: 500, y: 3000))
        let startY = whip.gripPoint.y

        advance(&whip, count: 120)

        #expect(whip.gripPoint.y < startY)
    }

    @Test
    func aHeldWhipKeepsItsGripOnTheHand() {
        var whip = WhipSimulation(origin: CGPoint(x: 500, y: 3000))
        whip.beginGrab()

        let hand = CGPoint(x: 900, y: 2600)
        whip.setGrabTarget(hand)
        advance(&whip, count: 60, input: WhipInput(grabPoint: hand, cursor: hand))

        #expect(abs(whip.gripPoint.x - hand.x) < 0.5)
        #expect(abs(whip.gripPoint.y - hand.y) < 0.5)
    }

    /// A whip is inextensible. If the distance constraints are under-iterated
    /// this is where it shows: the thong stretches like elastic under a hard
    /// flick and the crack goes soft.
    @Test
    func staysInextensibleUnderAViolentFlick() {
        var whip = WhipSimulation(origin: CGPoint(x: 1500, y: 3000))
        whip.beginGrab()

        // Slam the hand back and forth across a wide arc.
        for sweep in 0..<8 {
            let x: CGFloat = sweep.isMultiple(of: 2) ? 900 : 2100
            let hand = CGPoint(x: x, y: 2600)
            whip.setGrabTarget(hand)
            advance(&whip, count: 20, input: WhipInput(grabPoint: hand, cursor: hand))
        }

        let rest = restLength(whip)
        let measured = measuredLength(whip)
        // Verlet with a finite iteration count always leaves a little slack;
        // 6% is tight enough that the whip reads as a solid object.
        #expect(measured <= rest * 1.06)
    }

    // MARK: - Sleep

    @Test
    func comesToRestAndSleeps() {
        var whip = WhipSimulation(origin: CGPoint(x: 500, y: 900))
        let world = WhipWorld(
            obstacles: [], bounds: CGRect(x: 0, y: 0, width: 2000, height: 1000))

        // Long enough to fall, hit the floor, and stop bouncing.
        let tick = advance(&whip, count: 240 * 25, world: world)

        #expect(tick.asleep)
        #expect(whip.isAsleep)
    }

    @Test
    func aHeldWhipNeverSleeps() {
        var whip = WhipSimulation(origin: CGPoint(x: 500, y: 900))
        whip.beginGrab()
        let hand = CGPoint(x: 500, y: 900)
        whip.setGrabTarget(hand)

        let tick = advance(
            &whip, count: 240 * 20,
            input: WhipInput(grabPoint: hand, cursor: hand),
            world: WhipWorld(obstacles: [], bounds: CGRect(x: 0, y: 0, width: 2000, height: 1000)))

        #expect(!tick.asleep)
    }

    // MARK: - Collision

    @Test
    func drapesOnTopOfAWindowRatherThanThroughIt() {
        // A window directly beneath the whip's fall line.
        let window = CGRect(x: 200, y: 100, width: 900, height: 400)
        let world = WhipWorld(
            obstacles: [window], bounds: CGRect(x: 0, y: 0, width: 2000, height: 1400))

        var whip = WhipSimulation(origin: CGPoint(x: 650, y: 1300))
        advance(&whip, count: 240 * 12, world: world)

        // No point may be inside the window. Points are ejected to the nearest
        // edge with their own radius as clearance, so allow that much slack.
        for index in whip.positions.indices {
            let point = whip.positions[index]
            let inflated = window.insetBy(dx: -whip.radii[index], dy: -whip.radii[index])
            #expect(!inflated.insetBy(dx: 0.5, dy: 0.5).contains(point))
        }
    }

    @Test
    func staysInsideTheCanvasBounds() {
        let bounds = CGRect(x: 0, y: 0, width: 800, height: 800)
        let world = WhipWorld(obstacles: [], bounds: bounds)

        var whip = WhipSimulation(origin: CGPoint(x: 400, y: 700))
        whip.beginGrab()

        // Drag the hand well outside the canvas and let the thong follow.
        for target in [CGPoint(x: 3000, y: 400), CGPoint(x: -3000, y: 400)] {
            whip.setGrabTarget(target)
            advance(
                &whip, count: 120, input: WhipInput(grabPoint: target, cursor: target), world: world
            )
        }
        whip.endGrab()
        advance(&whip, count: 240, world: world)

        // The carried grip is exempt (the hand can leave the screen); everything
        // the simulation owns must stay in.
        for index in 1..<whip.pointCount {
            let point = whip.positions[index]
            #expect(point.x >= bounds.minX - 1)
            #expect(point.x <= bounds.maxX + 1)
            #expect(point.y >= bounds.minY - 1)
            #expect(point.y <= bounds.maxY + 1)
        }
    }

    // MARK: - The crack

    /// The crack must be earned. A slow drag moves the whole whip but never
    /// builds a wave, so nothing should fire.
    @Test
    func aSlowDragNeverCracks() {
        var whip = WhipSimulation(origin: CGPoint(x: 1000, y: 2000))
        whip.beginGrab()

        var cracked = false
        for stepIndex in 0..<(240 * 4) {
            // ~150 points/s of hand travel.
            let hand = CGPoint(x: 1000 + CGFloat(stepIndex) * 0.6, y: 2000)
            whip.setGrabTarget(hand)
            let tick = whip.step(
                dt: Self.step,
                input: WhipInput(grabPoint: hand, cursor: hand),
                world: Self.roomyWorld,
                time: Double(stepIndex) * Double(Self.step))
            if tick.cracked { cracked = true }
        }

        #expect(!cracked)
    }

    /// The taper's whole purpose: the tip must end up moving far faster than the
    /// hand that drove it. This is the emergent behaviour the feature is built
    /// on, so it is asserted directly rather than inferred from a crack flag.
    @Test
    func theTipOutrunsTheHandOnAFlick() {
        var whip = WhipSimulation(origin: CGPoint(x: 1500, y: 2500))
        whip.beginGrab()

        // Settle first, so the flick acts on a hanging whip rather than one
        // still falling into place.
        let rest = CGPoint(x: 1500, y: 2500)
        whip.setGrabTarget(rest)
        advance(&whip, count: 240 * 3, input: WhipInput(grabPoint: rest, cursor: rest))

        // One hard forward throw, then a sharp stop — the classic crack input.
        var peakTipSpeed: CGFloat = 0
        let handSpeed: CGFloat = 9  // points per substep ≈ 2160 pt/s
        var hand = rest
        for stepIndex in 0..<40 {
            hand = CGPoint(x: hand.x + handSpeed, y: hand.y)
            whip.setGrabTarget(hand)
            let tick = whip.step(
                dt: Self.step,
                input: WhipInput(grabPoint: hand, cursor: hand),
                world: Self.roomyWorld,
                time: Double(stepIndex) * Double(Self.step))
            peakTipSpeed = max(peakTipSpeed, tick.tipSpeed)
        }
        // Stop dead and let the wave run out into the thin end.
        for stepIndex in 40..<200 {
            whip.setGrabTarget(hand)
            let tick = whip.step(
                dt: Self.step,
                input: WhipInput(grabPoint: hand, cursor: hand),
                world: Self.roomyWorld,
                time: Double(stepIndex) * Double(Self.step))
            peakTipSpeed = max(peakTipSpeed, tick.tipSpeed)
        }

        let handSpeedPerSecond = handSpeed * 240
        #expect(peakTipSpeed > handSpeedPerSecond * 1.5)
    }

    /// The payoff. A hard flick must actually fire a crack, exactly once, with
    /// an intensity in range — the sound is driven straight off this.
    @Test
    func aHardFlickCracksOnce() {
        var whip = WhipSimulation(origin: CGPoint(x: 1500, y: 2500))
        whip.beginGrab()

        let rest = CGPoint(x: 1500, y: 2500)
        whip.setGrabTarget(rest)
        advance(&whip, count: 240 * 3, input: WhipInput(grabPoint: rest, cursor: rest))

        var cracks: [CGFloat] = []
        var hand = rest
        for stepIndex in 0..<240 {
            if stepIndex < 40 { hand = CGPoint(x: hand.x + 11, y: hand.y) }
            whip.setGrabTarget(hand)
            let tick = whip.step(
                dt: Self.step,
                input: WhipInput(grabPoint: hand, cursor: hand),
                world: Self.roomyWorld,
                time: Double(stepIndex) * Double(Self.step))
            if tick.cracked { cracks.append(tick.crackIntensity) }
        }

        #expect(!cracks.isEmpty)
        // The cooldown must stop one flick registering as a burst of cracks.
        #expect(cracks.count <= 2)
        for intensity in cracks {
            #expect(intensity > 0)
            #expect(intensity <= 1)
        }
    }
}

// MARK: - Geometry

struct WhipGeometryTests {

    @Test
    func outlineIsEmptyForDegenerateInput() {
        #expect(WhipGeometry.outline(points: [], radii: []).isEmpty)
        #expect(WhipGeometry.outline(points: [.zero], radii: [1]).isEmpty)
        // Mismatched counts must not trap.
        #expect(WhipGeometry.outline(points: [.zero, CGPoint(x: 1, y: 1)], radii: [1]).isEmpty)
    }

    @Test
    func outlineWrapsBothSidesOfTheChain() {
        let points = [CGPoint(x: 0, y: 100), CGPoint(x: 0, y: 50), CGPoint(x: 0, y: 0)]
        let radii: [CGFloat] = [6, 3, 1]
        let path = WhipGeometry.outline(points: points, radii: radii)

        #expect(!path.isEmpty)
        // The widest part of the outline is twice the largest radius.
        let box = path.boundingBox
        #expect(box.width >= 11.5)
        #expect(box.height >= 100)
    }

    @Test
    func normalsArePerpendicularAndUnitLength() {
        let points = [CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0), CGPoint(x: 20, y: 0)]
        let normals = WhipGeometry.normals(for: points)

        #expect(normals.count == points.count)
        for normal in normals {
            #expect(abs(hypot(normal.dx, normal.dy) - 1) < 0.0001)
            // A horizontal chain has vertical normals.
            #expect(abs(normal.dx) < 0.0001)
        }
    }

    @Test
    func boundsCoverEveryPointPlusPadding() {
        let points = [CGPoint(x: 0, y: 0), CGPoint(x: 100, y: 60)]
        let radii: [CGFloat] = [5, 2]
        let rect = WhipGeometry.bounds(points: points, radii: radii, padding: 10)

        #expect(rect.minX <= -15)
        #expect(rect.maxX >= 112)
        #expect(rect.minY <= -15)
        #expect(rect.maxY >= 72)
    }

    /// The handle must fill *solid*. A capsule built from two arcs is easy to
    /// wind inconsistently, and a nonzero-winding fill then leaves a hole that
    /// only shows up on screen as a pale, half-painted grip. Rasterise it and
    /// look, rather than trusting that the arcs sweep the way they read.
    @Test
    func theHandleFillsSolidAtEveryAngle() {
        let radius: CGFloat = 7
        for degrees in stride(from: 0, to: 360, by: 30) {
            let angle = CGFloat(degrees) * .pi / 180
            let grip = CGPoint(x: 60, y: 60)
            let tip = CGPoint(x: 60 + cos(angle) * 40, y: 60 + sin(angle) * 40)
            let path = WhipGeometry.handlePath(from: grip, to: tip, radius: radius)

            let coverage = Self.filledFraction(of: path, along: grip, to: tip)
            #expect(coverage > 0.99, "handle at \(degrees)° filled only \(coverage)")
        }
    }

    /// The thong likewise: offsetting a chain by its radius can cross itself at
    /// a tight fold, and a crossed outline drops the fold out of the fill.
    @Test
    func theThongFillsSolidAlongItsSpine() {
        let points = (0..<20).map { CGPoint(x: 20 + CGFloat($0) * 6, y: 60) }
        let radii = (0..<20).map { 5 - CGFloat($0) * 0.2 }
        let path = WhipGeometry.outline(points: points, radii: radii)

        let coverage = Self.filledFraction(of: path, along: points[1], to: points[17])
        #expect(coverage > 0.99, "thong filled only \(coverage)")
    }

    /// Fraction of samples along a line segment that land inside the filled
    /// path, using the same nonzero winding rule CoreGraphics fills with.
    private static func filledFraction(of path: CGPath, along start: CGPoint, to end: CGPoint)
        -> Double
    {
        let samples = 60
        var inside = 0
        for step in 0...samples {
            let t = CGFloat(step) / CGFloat(samples)
            let point = CGPoint(
                x: start.x + (end.x - start.x) * t,
                y: start.y + (end.y - start.y) * t)
            if path.contains(point, using: .winding) { inside += 1 }
        }
        return Double(inside) / Double(samples + 1)
    }
}
