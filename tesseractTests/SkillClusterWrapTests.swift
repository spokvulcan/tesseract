//
//  SkillClusterWrapTests.swift
//  tesseractTests
//
//  Tests the Skill Cluster's wrap computation (ADR-0030): pills fan leftward
//  from the bubble, most-used nearest, wrapping upward when out of width.
//  Pure geometry — indices in, rows out. Row 0 is the bottom row (the one at
//  bubble height); within a row, earlier indices sit further right.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct SkillClusterWrapTests {

    @Test func everythingFitsInOneRow() {
        let rows = SkillClusterWrap.rows(
            itemWidths: [80, 60, 70], available: 300, spacing: 10)
        #expect(rows == [[0, 1, 2]])
    }

    @Test func overflowWrapsIntoARowAbove() {
        // 100 + 10 + 100 = 210 fits in 220; the third pill starts row 1.
        let rows = SkillClusterWrap.rows(
            itemWidths: [100, 100, 100], available: 220, spacing: 10)
        #expect(rows == [[0, 1], [2]])
    }

    @Test func anItemWiderThanAvailableStillGetsARow() {
        let rows = SkillClusterWrap.rows(
            itemWidths: [500, 80], available: 220, spacing: 10)
        #expect(rows == [[0], [1]])
    }

    @Test func noItemsMeansNoRows() {
        let rows = SkillClusterWrap.rows(itemWidths: [], available: 220, spacing: 10)
        #expect(rows.isEmpty)
    }
}
