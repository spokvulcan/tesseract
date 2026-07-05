//
//  AppshotEncoder.swift
//  tesseract
//

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Encodes a captured window image to PNG bytes that fit under a byte cap.
/// An Appshot is created by the app itself, so unlike pasted images (where the
/// cap is a typed rejection) an over-cap capture downscales to fit — rejecting
/// our own capture would make the hotkey feel broken on large Retina windows.
nonisolated enum AppshotEncoder {

    /// PNG-encode `image`; while the encoding exceeds `cap`, redraw at a
    /// proportionally smaller size and retry. Returns nil only if encoding
    /// fails outright.
    static func pngDataFittingCap(_ image: CGImage, cap: Int) -> Data? {
        var current = image
        for _ in 0..<8 {
            guard let data = pngData(from: current) else { return nil }
            if data.count <= cap { return data }
            // Scale area by the byte overshoot (bytes track pixel count for a
            // given content), padded a little so we rarely need a second pass.
            let ratio = (Double(cap) / Double(data.count)).squareRoot() * 0.95
            guard let scaled = draw(current, scaledBy: ratio) else { return nil }
            current = scaled
        }
        return nil
    }

    private static func pngData(from image: CGImage) -> Data? {
        let data = NSMutableData()
        guard
            let destination = CGImageDestinationCreateWithData(
                data as CFMutableData, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }

    private static func draw(_ image: CGImage, scaledBy ratio: Double) -> CGImage? {
        let width = max(1, Int(Double(image.width) * ratio))
        let height = max(1, Int(Double(image.height) * ratio))
        guard
            let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }
}
