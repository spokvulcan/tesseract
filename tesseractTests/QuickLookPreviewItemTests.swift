//
//  QuickLookPreviewItemTests.swift
//  tesseractTests
//
//  Regression for the 2026-06-15 Quick Look crash (clicking a pasted image).
//  QuickLook reads `previewItemURL`/`previewItemTitle` on a background
//  NSOperationQueue during its async-loading determination. Under the build's
//  MainActor-default isolation an unannotated `QLPreviewItem` gets a main-actor
//  `@objc` getter that traps the executor check off-main → SIGTRAP. The item
//  must be `nonisolated`.
//
//  The seam mirrors QuickLook's access: read through the `any QLPreviewItem`
//  existential (the same nonisolated `@objc` path the framework uses) from a
//  detached, off-main task. A main-actor-isolated getter would trap here.
//

import Foundation
import Quartz
import Testing

@testable import Tesseract_Agent

struct QuickLookPreviewItemTests {

    @Test func previewItemAccessorsAreReadableOffMainActor() async {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ql-isolation-probe.png")
        // `any QLPreviewItem` is not Sendable; the unsafe escape lets the
        // detached task capture it exactly as QuickLook holds the item across
        // its background queue hop.
        nonisolated(unsafe) let item: any QLPreviewItem = QuickLookPreviewItem(url: url)

        let (readURL, readTitle) = await Task.detached {
            (item.previewItemURL, item.previewItemTitle)
        }.value

        #expect(readURL == url)
        #expect(readTitle == url.lastPathComponent)
    }
}
