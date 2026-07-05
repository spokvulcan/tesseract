//
//  AppshotEncoderTests.swift
//  tesseractTests
//
//  Tests the Appshot capture encoder: PNG bytes out, never over the byte cap —
//  a capture the user triggered must not be rejected by our own ingest cap, so
//  an oversized window downscales to fit instead.
//

import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import Tesseract_Agent

struct AppshotEncoderTests {

    /// A deterministic noise image — noise defeats PNG compression, so a modest
    /// pixel count still produces a large encoding.
    private func makeNoiseImage(width: Int, height: Int) -> CGImage {
        var seed: UInt64 = 0x5EED_1234
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for index in bytes.indices {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            bytes[index] = UInt8(truncatingIfNeeded: seed >> 33)
        }
        let context = CGContext(
            data: &bytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return context.makeImage()!
    }

    private func decoded(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    @Test
    func imageUnderTheCapKeepsItsDimensions() throws {
        let image = makeNoiseImage(width: 64, height: 48)
        let data = try #require(AppshotEncoder.pngDataFittingCap(image, cap: 1024 * 1024))
        let roundTripped = try #require(decoded(data))
        #expect(data.count <= 1024 * 1024)
        #expect(roundTripped.width == 64)
        #expect(roundTripped.height == 48)
    }

    @Test
    func oversizedImageDownscalesUnderTheCap() throws {
        let image = makeNoiseImage(width: 512, height: 384)
        // Noise at this size encodes to several hundred KB; force a downscale.
        let cap = 64 * 1024
        let data = try #require(AppshotEncoder.pngDataFittingCap(image, cap: cap))
        let roundTripped = try #require(decoded(data))
        #expect(data.count <= cap)
        #expect(roundTripped.width < 512)
        #expect(roundTripped.height < 384)
        #expect(roundTripped.width > 0)
    }

    @Test
    func outputIsDecodablePNG() throws {
        let image = makeNoiseImage(width: 64, height: 64)
        let data = try #require(AppshotEncoder.pngDataFittingCap(image, cap: 1024 * 1024))
        #expect(ImageIngest.isDecodableImage(data))
        // PNG magic bytes — ingest stores this as image/png.
        #expect(data.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
    }
}
