//
//  ReadingMeterTests.swift
//  tesseractTests
//
//  The TestFlight diagnostics (#515): each segment's speed from the speech
//  engine's begin and end marks, and the report a tester copies.
//

import Foundation
import Synchronization
import Testing

@testable import Tesseract_Agent

struct ReadingMeterTests {
    /// A clock the test moves by hand.
    private final class Clock: Sendable {
        let instant = Mutex(ContinuousClock.now)
        func advance(_ seconds: Double) {
            instant.withLock { $0 = $0.advanced(by: .seconds(seconds)) }
        }
    }

    @Test func eachSegmentsSpeedComesFromItsMarks() {
        let clock = Clock()
        let meter = ReadingMeter(
            framesPerSecond: 12.5, now: { clock.instant.withLock { $0 } }, thermalState: { .fair })
        meter.event("burst.begin", "segment 0")
        clock.advance(2)
        meter.event("burst.end", "segment 0, 100 frames")
        meter.event("burst.begin", "segment 1")
        clock.advance(4)
        meter.event("burst.end", "segment 1, 125 frames")

        let segments = meter.segments
        #expect(segments.count == 2)
        // 100 frames are 8 s of audio, made in 2 s.
        #expect(abs(segments[0].realTimeFactor - 0.25) < 1e-9)
        #expect(abs(segments[1].realTimeFactor - 0.4) < 1e-9)
        #expect(segments.allSatisfy { $0.thermalState == .fair })
    }

    @Test func aNewReadingStartsTheCountOver() {
        let meter = ReadingMeter()
        meter.event("burst.begin", "segment 0")
        meter.event("burst.end", "segment 0, 10 frames")
        meter.event("burst.begin", "segment 1")
        meter.event("burst.end", "segment 1, 10 frames")
        meter.event("burst.begin", "segment 0")
        meter.event("burst.end", "segment 0, 10 frames")
        #expect(meter.segments.count == 1)
    }

    @Test func theReportSaysWhatATesterNeeds() {
        let segments = [
            ReadingMeter.Segment(seconds: 2, audioSeconds: 8, thermalState: .nominal),
            ReadingMeter.Segment(seconds: 3, audioSeconds: 6, thermalState: .serious),
            ReadingMeter.Segment(seconds: 1, audioSeconds: 5, thermalState: .fair),
        ]
        let report = ReadingMeter.report(
            app: "Tesseract 1.0 (1)", device: "iPhone17,2", system: "iOS 27.0",
            voice: "System Voice", segments: segments)
        #expect(
            report == """
                Tesseract 1.0 (1)
                iPhone17,2 · iOS 27.0
                Voice: System Voice
                Last reading: 3 segments, 19 s of audio, real-time factor 0.25 median, 0.50 worst
                Thermal state: nominal at the start, serious at the warmest
                """)
        #expect(
            ReadingMeter.report(app: "T", device: "D", system: "S", voice: "V", segments: [])
                .hasSuffix("Last reading: none yet"))
    }
}
