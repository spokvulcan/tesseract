//
//  WhipPhysics.swift
//  tesseract
//
//  The whip simulation — a pure, `nonisolated` value machine with no window,
//  no panel and no AppKit in it, so the feel is unit-testable the way
//  `SkillClusterWrap` is.
//
//  One Verlet chain models the whole object. Points 0…1 are the handle (a
//  rigid two-point body: two points give position *and* orientation for free
//  under the same integrator); points 2… are the tapered thong. That uniformity
//  is the trick — the wrist flick is not special-cased anywhere, it falls out
//  of the handle's second point swinging while the first is carried.
//
//  The taper is what makes a whip crack, so it is modelled properly rather
//  than faked: for a circular cross-section linear density goes as r² and
//  bending stiffness as r⁴ (Euler-Bernoulli). A section at half the radius
//  therefore carries a quarter the mass and bends ~16× more easily. As the
//  wave travels into the thin end it runs out of mass to move, and momentum
//  conservation forces the velocity up until the tip outruns everything. The
//  crack is *detected*, never scripted.
//

import CoreGraphics
import Foundation

// MARK: - Parameters

/// Every number that decides how the whip feels, in one value. Kept separate
/// from the state so a tuning pass is a diff on literals, not a hunt through
/// the solver.
nonisolated struct WhipParameters: Sendable, Equatable {

    /// Thong segment count. More points buy a smoother wave and a sharper
    /// crack; the solver is O(points × iterations) and both are tiny.
    var thongPointCount: Int = 36

    /// Handle length, points. The rigid part you actually hold.
    var handleLength: CGFloat = 44
    /// Thong length, points.
    var thongLength: CGFloat = 540

    /// Handle half-width.
    var handleRadius: CGFloat = 7.0
    /// Thong half-width where it meets the handle.
    var thongBaseRadius: CGFloat = 4.6
    /// Thong half-width at the tip (the fall/cracker).
    var tipRadius: CGFloat = 0.55

    /// Downward acceleration, points/s². Tuned by feel rather than to real
    /// gravity: screen points are not metres, and a whip that falls at 9.8 m/s²
    /// scaled to a display reads as floating in syrup.
    var gravity: CGFloat = 3400

    /// Per-substep velocity retention. Below 1 the chain loses energy and can
    /// therefore come to rest, which is what lets the simulation sleep. At the
    /// 240 Hz substep this is ~0.79 per second — enough to settle, loose enough
    /// that the thong stays lively.
    var linearDamping: CGFloat = 0.9990
    /// Quadratic air drag coefficient, applied implicitly. Scales with speed, so
    /// it barely touches a resting whip and bites hard on a fast one — this is
    /// what makes a big swing cost effort.
    var airDrag: CGFloat = 0.0004

    /// Gauss-Seidel passes per substep. Distance constraints need enough
    /// iterations to stay inextensible under a hard flick; too few and the whip
    /// visibly stretches.
    var constraintIterations: Int = 14

    /// Multiplier on the r⁴ bending term. 0 is a pure rope, 1 is as stiff as
    /// the taper allows.
    var bendStiffness: CGFloat = 0.85

    /// Bounce retained on collision with a window edge.
    var restitution: CGFloat = 0.18
    /// Tangential velocity retained on collision — below 1 the thong grips and
    /// drapes rather than sliding off every surface.
    var friction: CGFloat = 0.62
    /// Separation speed (points/s) below which a contact is treated as resting
    /// rather than bouncing.
    var restingSpeed: CGFloat = 90

    /// Kinetic energy below which the whip is considered still.
    var sleepEnergy: CGFloat = 26
    /// Consecutive still substeps before the simulation sleeps.
    var sleepFrames: Int = 90

    /// How close the cursor gets before it starts shoving the thong aside.
    var cursorPushRadius: CGFloat = 52
    /// How hard it shoves, per point of cursor travel.
    var cursorPushStrength: CGFloat = 0.9

    /// Tip speed (points/s) a peak must exceed to count as a crack.
    var crackSpeed: CGFloat = 2600
    /// Minimum gap between cracks, seconds — one flick is one crack.
    var crackCooldown: Double = 0.11

    /// Distance from the handle's grip point within which a grab registers.
    var grabRadius: CGFloat = 34

    static let `default` = WhipParameters()
}

// MARK: - Input

/// What the outside world does to the whip in one tick. A value, so a test can
/// drive a flick without a mouse.
nonisolated struct WhipInput: Sendable, Equatable {
    /// Where the hand is, when the handle is being carried.
    var grabPoint: CGPoint?
    /// Live cursor position, used for the brush-past shove when not grabbed.
    var cursor: CGPoint?
    /// Cursor travel since the previous tick, which is what gives the shove its
    /// strength — a stationary cursor resting on the thong does nothing.
    var cursorDelta: CGVector = .zero

    static let idle = WhipInput()
}

// MARK: - Tick result

/// What one tick produced, for the renderer and the synthesiser to read. The
/// audio and the drawing consume the same numbers, so they cannot desync.
nonisolated struct WhipTick: Sendable, Equatable {
    /// Tip speed in points/s — drives the whoosh's centre frequency and gain.
    var tipSpeed: CGFloat = 0
    /// A crack fired this tick.
    var cracked: Bool = false
    /// Loudness of that crack, 0…1, from how far past threshold the peak went.
    var crackIntensity: CGFloat = 0
    /// Total kinetic energy, the sleep signal.
    var energy: CGFloat = 0
    /// The simulation is at rest and can stop stepping entirely.
    var asleep: Bool = false
}

// MARK: - Collision world

/// The surfaces the whip can hit, as axis-aligned rects in simulation space.
/// A value with no knowledge of where the rects came from, so the physics tests
/// never touch the window server.
nonisolated struct WhipWorld: Sendable, Equatable {
    /// Solid rects — real application windows.
    var obstacles: [CGRect] = []
    /// The outer bounds the whip is kept inside (the union of the displays).
    var bounds: CGRect = .zero

    static let empty = WhipWorld()
}

// MARK: - Simulation

/// The whip itself: positions, the previous positions Verlet integrates from,
/// and the per-point constants the taper implies.
nonisolated struct WhipSimulation: Sendable {

    // MARK: Stored state

    private(set) var positions: [CGPoint]
    private(set) var previous: [CGPoint]

    /// Half-width at each point — drives drawing, mass and stiffness alike.
    private(set) var radii: [CGFloat]
    /// Inverse mass per point (0 = pinned). Mass goes as r².
    private(set) var inverseMass: [CGFloat]
    /// Rest length of each segment between consecutive points.
    private(set) var restLengths: [CGFloat]
    /// Bending resistance at each interior point, normalised to 0…1. Goes as r⁴.
    private(set) var bendResistance: [CGFloat]

    /// Surface normal each point was last pushed along this substep, or zero for
    /// points not in contact. Preallocated so the solve never allocates.
    private var contactNormals: [CGVector]

    var parameters: WhipParameters

    /// True while the hand carries the handle.
    private(set) var isGrabbed = false

    private var stillFrames = 0
    private var lastCrackTime: Double = 0
    private var previousTipSpeed: CGFloat = 0
    private var risingTipSpeed = false

    /// Index of the handle's grip point — the end you hold.
    static let gripIndex = 0
    /// Index of the handle's far end, where the thong is attached.
    static let handleTipIndex = 1
    /// First thong point.
    static let thongStartIndex = 2

    var gripPoint: CGPoint { positions[Self.gripIndex] }
    var tipPoint: CGPoint { positions[positions.count - 1] }
    var pointCount: Int { positions.count }

    // MARK: Construction

    /// Builds a whip hanging straight down from `origin`, at rest.
    init(origin: CGPoint, parameters: WhipParameters = .default) {
        self.parameters = parameters

        let thongCount = max(4, parameters.thongPointCount)
        let total = 2 + thongCount

        var positions: [CGPoint] = []
        var radii: [CGFloat] = []
        positions.reserveCapacity(total)
        radii.reserveCapacity(total)

        // The handle: two points, `handleLength` apart, pointing down.
        positions.append(origin)
        positions.append(CGPoint(x: origin.x, y: origin.y - parameters.handleLength))
        radii.append(parameters.handleRadius)
        radii.append(parameters.handleRadius)

        // The thong: an exponential taper from base to tip. Exponential rather
        // than linear because real whips are built that way — a linear taper
        // keeps too much mass at the far end and the crack goes soft.
        let thongSegment = parameters.thongLength / CGFloat(thongCount)
        let ratio = parameters.tipRadius / max(parameters.thongBaseRadius, 0.0001)
        for index in 0..<thongCount {
            let t = CGFloat(index) / CGFloat(max(thongCount - 1, 1))
            let y = origin.y - parameters.handleLength - thongSegment * CGFloat(index + 1)
            positions.append(CGPoint(x: origin.x, y: y))
            radii.append(parameters.thongBaseRadius * pow(ratio, t))
        }

        self.positions = positions
        self.previous = positions
        self.radii = radii
        self.contactNormals = Array(repeating: .zero, count: total)

        // Mass ∝ r² × segment length. The handle is deliberately heavy: it is
        // what you swing, and a light handle feels like waving a straw.
        var inverseMass: [CGFloat] = []
        inverseMass.reserveCapacity(total)
        for index in 0..<total {
            let radius = radii[index]
            let length = index < Self.thongStartIndex ? parameters.handleLength : thongSegment
            let mass = radius * radius * length * (index < Self.thongStartIndex ? 2.2 : 1.0)
            inverseMass.append(1 / max(mass, 0.0001))
        }
        self.inverseMass = inverseMass

        var restLengths: [CGFloat] = []
        restLengths.reserveCapacity(total - 1)
        for index in 0..<(total - 1) {
            restLengths.append(index == 0 ? parameters.handleLength : thongSegment)
        }
        self.restLengths = restLengths

        // Bending resistance ∝ r⁴, normalised against the handle so the thickest
        // joint is ~1 and the last fingers of thong are ~0 (pure rope).
        //
        // A joint is only as stiff as its thinnest part, so each interior point
        // takes the *minimum* radius across the three points it spans. Using the
        // point's own radius made the handle-to-thong neck fully rigid, which
        // welded the thong to the handle's direction and killed the flick.
        // The handle itself needs no bending term at all: it is a single segment,
        // and one distance constraint already makes it rigid.
        let reference = pow(parameters.handleRadius, 4)
        var bend: [CGFloat] = []
        bend.reserveCapacity(total)
        for index in 0..<total {
            let previous = index > 0 ? radii[index - 1] : radii[index]
            let next = index < total - 1 ? radii[index + 1] : radii[index]
            let joint = min(radii[index], min(previous, next))
            bend.append(min(1, pow(joint, 4) / max(reference, 0.0001)))
        }
        self.bendResistance = bend
    }

    // MARK: Grab

    mutating func beginGrab() {
        isGrabbed = true
        wake()
    }

    mutating func endGrab() {
        isGrabbed = false
    }

    /// Nudges the simulation out of its sleep state.
    mutating func wake() {
        stillFrames = 0
    }

    var isAsleep: Bool { stillFrames >= parameters.sleepFrames }

    /// Teleports the whole whip so the grip lands on `point`, preserving shape
    /// and velocity. Used when the resting place is no longer on any display.
    mutating func translate(to point: CGPoint) {
        let delta = CGVector(dx: point.x - positions[0].x, dy: point.y - positions[0].y)
        for index in positions.indices {
            positions[index].x += delta.dx
            positions[index].y += delta.dy
            previous[index].x += delta.dx
            previous[index].y += delta.dy
        }
    }

    // MARK: Step

    /// Advances by exactly `dt` seconds. Callers drive this at a fixed rate from
    /// an accumulator — Verlet with a variable timestep changes its effective
    /// damping every frame, which reads as the whip getting heavier when the
    /// machine is busy.
    mutating func step(dt: CGFloat, input: WhipInput, world: WhipWorld, time: Double) -> WhipTick {
        guard dt > 0 else { return WhipTick(asleep: isAsleep) }

        integrate(dt: dt, input: input)
        solve(world: world, dt: dt)

        // Tip speed is read after the solve so it reflects where the tip
        // actually ended up, constraints and collisions included.
        let last = positions.count - 1
        let tipVelocity = CGVector(
            dx: positions[last].x - previous[last].x,
            dy: positions[last].y - previous[last].y)
        let tipSpeed = hypot(tipVelocity.dx, tipVelocity.dy) / dt

        var tick = WhipTick()
        tick.tipSpeed = tipSpeed
        tick.energy = kineticEnergy(dt: dt)

        detectCrack(tipSpeed: tipSpeed, time: time, into: &tick)

        // Sleep bookkeeping: a carried whip never sleeps, however still the
        // hand is holding it.
        if tick.energy < parameters.sleepEnergy && !isGrabbed {
            stillFrames += 1
        } else {
            stillFrames = 0
        }
        tick.asleep = isAsleep

        previousTipSpeed = tipSpeed
        return tick
    }

    // MARK: Integration

    private mutating func integrate(dt: CGFloat, input: WhipInput) {
        let gravityStep = -parameters.gravity * dt * dt
        let damping = parameters.linearDamping

        for index in positions.indices {
            let current = positions[index]
            var velocity = CGVector(
                dx: current.x - previous[index].x,
                dy: current.y - previous[index].y)

            velocity.dx *= damping
            velocity.dy *= damping

            // Quadratic air drag, in the implicit form `v / (1 + k·|v|·dt)`.
            // The explicit form (`v -= k·|v|²·v·dt`) has to be clamped to stop
            // it overshooting into negative velocity, and that clamp silently
            // became the common case: above roughly 1700 pt/s it saturated and
            // zeroed the whole chain's velocity every step, so no wave could
            // ever reach the tip. This form is unconditionally stable and never
            // changes sign.
            let speedPerSecond = hypot(velocity.dx, velocity.dy) / dt
            if speedPerSecond > 0 {
                let dragFactor = 1 / (1 + parameters.airDrag * speedPerSecond * dt)
                velocity.dx *= dragFactor
                velocity.dy *= dragFactor
            }

            previous[index] = current
            positions[index] = CGPoint(
                x: current.x + velocity.dx,
                y: current.y + velocity.dy + gravityStep)
        }

        applyCursorShove(input: input)
    }

    /// The brush-past: a moving cursor pushes thong points aside without any
    /// click ever being involved. Strength scales with how fast the cursor is
    /// travelling, so resting the pointer on the whip does nothing and swiping
    /// through it sends a ripple.
    private mutating func applyCursorShove(input: WhipInput) {
        guard !isGrabbed, let cursor = input.cursor else { return }
        let travel = hypot(input.cursorDelta.dx, input.cursorDelta.dy)
        guard travel > 0.5 else { return }

        let radius = parameters.cursorPushRadius
        let radiusSquared = radius * radius

        for index in Self.thongStartIndex..<positions.count {
            let dx = positions[index].x - cursor.x
            let dy = positions[index].y - cursor.y
            let distanceSquared = dx * dx + dy * dy
            guard distanceSquared < radiusSquared, distanceSquared > 0.0001 else { continue }

            // Falls off smoothly to zero at the radius so there is no edge to
            // feel when the cursor crosses the boundary.
            let falloff = 1 - sqrt(distanceSquared) / radius
            let push = parameters.cursorPushStrength * falloff * min(travel, 60)
            let distance = sqrt(distanceSquared)
            positions[index].x += dx / distance * push
            positions[index].y += dy / distance * push
        }
        stillFrames = 0
    }

    // MARK: Constraint solve

    /// One constraint solve. Collision is projected *inside* the iteration loop
    /// alongside the other constraints, and the velocity response is applied
    /// once at the end.
    ///
    /// Resolving collisions in a single pass after the loop was a real bug: the
    /// distance solve and the collision pass took turns undoing each other, and
    /// a whip coiled on the floor sat in a limit cycle shimmering forever
    /// instead of settling — so it never slept, and never stopped drawing.
    private mutating func solve(world: WhipWorld, dt: CGFloat) {
        for index in contactNormals.indices { contactNormals[index] = .zero }

        for _ in 0..<parameters.constraintIterations {
            if isGrabbed { pinGrip() }
            solveDistances()
            solveBending()
            projectCollisions(world: world)
        }
        if isGrabbed { pinGrip() }
        applyContactVelocities(dt: dt)
    }

    /// While carried, the grip is wherever the hand is. Setting `previous` to
    /// match would kill the handle's velocity and with it the wrist flick, so
    /// only the position is forced.
    private mutating func pinGrip() {
        positions[Self.gripIndex] = grabTarget
    }

    private var grabTarget: CGPoint = .zero

    /// Moves the hand. Kept separate from `step` so the grab target survives
    /// across substeps within one frame.
    mutating func setGrabTarget(_ point: CGPoint) {
        grabTarget = point
        stillFrames = 0
    }

    private mutating func solveDistances() {
        for index in 0..<(positions.count - 1) {
            let a = index
            let b = index + 1
            let rest = restLengths[index]

            let dx = positions[b].x - positions[a].x
            let dy = positions[b].y - positions[a].y
            let distance = hypot(dx, dy)
            guard distance > 0.0001 else { continue }

            var weightA = inverseMass[a]
            let weightB = inverseMass[b]
            // A carried grip is infinitely heavy — the hand wins every argument.
            if isGrabbed && a == Self.gripIndex { weightA = 0 }
            let total = weightA + weightB
            guard total > 0 else { continue }

            let difference = (distance - rest) / distance
            positions[a].x += dx * difference * (weightA / total)
            positions[a].y += dy * difference * (weightA / total)
            positions[b].x -= dx * difference * (weightB / total)
            positions[b].y -= dy * difference * (weightB / total)
        }
    }

    /// Bending, as a one-sided distance constraint across each joint: the two
    /// points either side of an interior point are pushed apart toward the span
    /// they would have if the joint were straight (`L₀ + L₁`), by an amount
    /// proportional to the local r⁴ stiffness.
    ///
    /// The obvious formulation — pull the middle point toward the midpoint of
    /// its neighbours — is wrong here and was a real bug: it is a bending
    /// constraint *only* when both segments are the same length, and this whip
    /// joins a 44pt handle to 15pt thong segments. It therefore also tried to
    /// equalise them, injecting a large correction every iteration and firing
    /// the whip off the top of the screen. Constraining the span instead is
    /// correct for unequal segments, conserves momentum, and respects mass.
    ///
    /// One-sided because bending resists *folding*: a joint already straight has
    /// nothing to do, and the pair can never exceed `L₀ + L₁` anyway.
    private mutating func solveBending() {
        guard positions.count > 2 else { return }
        let scale = parameters.bendStiffness

        for index in 1..<(positions.count - 1) {
            let stiffness = bendResistance[index] * scale
            guard stiffness > 0.0005 else { continue }

            let a = index - 1
            let b = index + 1
            let straight = restLengths[index - 1] + restLengths[index]

            let dx = positions[b].x - positions[a].x
            let dy = positions[b].y - positions[a].y
            let distance = hypot(dx, dy)
            guard distance > 0.0001, distance < straight else { continue }

            var weightA = inverseMass[a]
            let weightB = inverseMass[b]
            // The hand wins: a carried grip absorbs the reaction rather than
            // being shoved around by the whip's own stiffness.
            if isGrabbed && a == Self.gripIndex { weightA = 0 }
            let total = weightA + weightB
            guard total > 0 else { continue }

            let difference = (distance - straight) / distance * stiffness
            positions[a].x += dx * difference * (weightA / total)
            positions[a].y += dy * difference * (weightA / total)
            positions[b].x -= dx * difference * (weightB / total)
            positions[b].y -= dy * difference * (weightB / total)
        }
    }

    // MARK: Collision

    /// Positional collision only — no velocity is touched here, because this
    /// runs many times per substep. Each point remembers the surface normal it
    /// was last pushed along, and the velocity response happens once afterwards.
    private mutating func projectCollisions(world: WhipWorld) {
        for index in positions.indices {
            if isGrabbed && index == Self.gripIndex { continue }
            let radius = radii[index]

            for obstacle in world.obstacles {
                let inflated = obstacle.insetBy(dx: -radius, dy: -radius)
                guard inflated.contains(positions[index]) else { continue }
                pushOut(index: index, from: inflated)
            }

            if !world.bounds.isEmpty {
                clampToBounds(index: index, bounds: world.bounds, radius: radius)
            }
        }
    }

    /// The velocity half of collision, applied once per substep to whatever
    /// ended up in contact: normal component scaled by restitution, tangential
    /// by friction. The friction term is why the thong drapes over a window's
    /// top edge instead of sliding straight off it.
    private mutating func applyContactVelocities(dt: CGFloat) {
        for index in contactNormals.indices {
            let normal = contactNormals[index]
            guard normal.dx != 0 || normal.dy != 0 else { continue }
            reflect(index: index, normal: normal, dt: dt)
        }
    }

    /// Ejects a point to the nearest edge of the rect it fell inside and records
    /// the normal it left along.
    private mutating func pushOut(index: Int, from rect: CGRect) {
        let point = positions[index]
        let toLeft = point.x - rect.minX
        let toRight = rect.maxX - point.x
        let toBottom = point.y - rect.minY
        let toTop = rect.maxY - point.y

        let minimum = min(min(toLeft, toRight), min(toBottom, toTop))
        var normal = CGVector(dx: 0, dy: 1)

        if minimum == toTop {
            positions[index].y = rect.maxY
            normal = CGVector(dx: 0, dy: 1)
        } else if minimum == toBottom {
            positions[index].y = rect.minY
            normal = CGVector(dx: 0, dy: -1)
        } else if minimum == toLeft {
            positions[index].x = rect.minX
            normal = CGVector(dx: -1, dy: 0)
        } else {
            positions[index].x = rect.maxX
            normal = CGVector(dx: 1, dy: 0)
        }

        contactNormals[index] = normal
    }

    private mutating func clampToBounds(index: Int, bounds: CGRect, radius: CGFloat) {
        if positions[index].x < bounds.minX + radius {
            positions[index].x = bounds.minX + radius
            contactNormals[index] = CGVector(dx: 1, dy: 0)
        } else if positions[index].x > bounds.maxX - radius {
            positions[index].x = bounds.maxX - radius
            contactNormals[index] = CGVector(dx: -1, dy: 0)
        }

        if positions[index].y < bounds.minY + radius {
            positions[index].y = bounds.minY + radius
            contactNormals[index] = CGVector(dx: 0, dy: 1)
        } else if positions[index].y > bounds.maxY - radius {
            positions[index].y = bounds.maxY - radius
            contactNormals[index] = CGVector(dx: 0, dy: -1)
        }
    }

    /// Velocity in Verlet lives in the gap between `positions` and `previous`,
    /// so a bounce is written by moving `previous`, never `positions`.
    private mutating func reflect(index: Int, normal: CGVector, dt: CGFloat) {
        let velocity = CGVector(
            dx: positions[index].x - previous[index].x,
            dy: positions[index].y - previous[index].y)

        let alongNormal = velocity.dx * normal.dx + velocity.dy * normal.dy
        let normalX = normal.dx * alongNormal
        let normalY = normal.dy * alongNormal
        let tangentX = velocity.dx - normalX
        let tangentY = velocity.dy - normalY

        // Resting contact. A separation slower than this is not a bounce, it is
        // the whip lying on the surface — bouncing it anyway is the other half
        // of why a coiled rope shimmers instead of settling.
        let restingThreshold = parameters.restingSpeed * dt
        let restitution = abs(alongNormal) < restingThreshold ? 0 : parameters.restitution

        let bouncedX = tangentX * parameters.friction - normalX * restitution
        let bouncedY = tangentY * parameters.friction - normalY * restitution

        previous[index] = CGPoint(
            x: positions[index].x - bouncedX,
            y: positions[index].y - bouncedY)
    }

    // MARK: Energy and crack detection

    private func kineticEnergy(dt: CGFloat) -> CGFloat {
        var total: CGFloat = 0
        for index in positions.indices {
            let dx = (positions[index].x - previous[index].x) / dt
            let dy = (positions[index].y - previous[index].y) / dt
            let mass = 1 / max(inverseMass[index], 0.0001)
            total += 0.5 * mass * (dx * dx + dy * dy)
        }
        // Normalised by point count so the threshold survives a change in
        // segment resolution.
        return total / CGFloat(positions.count) / 10000
    }

    /// A crack is the instant the tip's speed *peaks* above threshold — the
    /// moment the wave runs out of thong and the tip has nowhere left to put
    /// its momentum. Detecting the peak rather than the crossing means the
    /// sound lands where the whip visibly snaps, not early on the way up.
    private mutating func detectCrack(tipSpeed: CGFloat, time: Double, into tick: inout WhipTick) {
        defer { risingTipSpeed = tipSpeed > previousTipSpeed }

        guard tipSpeed > parameters.crackSpeed else { return }
        guard risingTipSpeed, tipSpeed <= previousTipSpeed else { return }
        guard time - lastCrackTime > parameters.crackCooldown else { return }

        lastCrackTime = time
        tick.cracked = true
        // How far past threshold the peak reached, softly clamped, so a gentle
        // snap is quiet and a full-arm swing is loud.
        let excess = (previousTipSpeed - parameters.crackSpeed) / parameters.crackSpeed
        tick.crackIntensity = min(1, max(0.15, excess))
    }
}
