import CoreGraphics
import CoreText
import Foundation
import MLX

/// Rasterizes text into an MLXArray image using CoreText.
/// Produces white text on black background, output in [-1, 1] range as [1, 3, H, W].
enum GlyphRasterizer {

    /// Rasterize multiple text strings into a single image.
    /// Texts are rendered vertically distributed — first text largest (title), rest smaller.
    static func rasterize(
        texts: [String],
        width: Int,
        height: Int,
        fontName: String = "Helvetica-Bold"
    ) -> MLXArray {
        guard !texts.isEmpty else {
            return MLXArray.zeros([1, 3, height, width]).asType(.bfloat16)
        }

        let bytesPerRow = width * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return MLXArray.zeros([1, 3, height, width]).asType(.bfloat16)
        }

        // Black background
        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        let padding = CGFloat(width) * 0.05
        let usableWidth = CGFloat(width) - 2 * padding
        let usableHeight = CGFloat(height) - 2 * padding

        if texts.count == 1 {
            // Single text: center it, auto-size to fill
            renderSingleText(
                texts[0], fontName: fontName,
                in: CGRect(x: padding, y: padding, width: usableWidth, height: usableHeight),
                context: context
            )
        } else {
            // Multiple texts: allocate vertical space proportionally
            // First text gets more space (title), rest share equally
            let titleFraction: CGFloat = texts.count <= 3 ? 0.4 : 0.3
            let bodyFraction = (1.0 - titleFraction) / CGFloat(texts.count - 1)
            let lineSpacing: CGFloat = usableHeight * 0.02

            var currentY = padding
            for (i, text) in texts.enumerated() {
                let fraction = i == 0 ? titleFraction : bodyFraction
                let slotHeight = usableHeight * fraction - lineSpacing
                let rect = CGRect(x: padding, y: currentY, width: usableWidth, height: slotHeight)
                renderSingleText(text, fontName: fontName, in: rect, context: context)
                currentY += usableHeight * fraction
            }
        }

        // Extract pixel data → MLXArray
        guard let data = context.data else {
            return MLXArray.zeros([1, 3, height, width]).asType(.bfloat16)
        }

        let buffer = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
        var rgbPixels = [Float](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            for x in 0..<width {
                // CoreGraphics has origin at bottom-left, flip Y
                let srcY = height - 1 - y
                let srcIdx = (srcY * width + x) * 4
                let dstIdx = (y * width + x)

                // Convert [0, 255] → [-1, 1]
                let r = Float(buffer[srcIdx]) / 127.5 - 1.0
                let g = Float(buffer[srcIdx + 1]) / 127.5 - 1.0
                let b = Float(buffer[srcIdx + 2]) / 127.5 - 1.0

                rgbPixels[0 * width * height + dstIdx] = r
                rgbPixels[1 * width * height + dstIdx] = g
                rgbPixels[2 * width * height + dstIdx] = b
            }
        }

        return MLXArray(rgbPixels, [1, 3, height, width]).asType(.bfloat16)
    }

    /// Convenience: rasterize a single text string.
    static func rasterize(
        text: String,
        width: Int,
        height: Int,
        fontName: String = "Helvetica-Bold"
    ) -> MLXArray {
        rasterize(texts: [text], width: width, height: height, fontName: fontName)
    }

    // MARK: - Private

    /// Render a single text string centered within the given rect.
    private static func renderSingleText(
        _ text: String,
        fontName: String,
        in rect: CGRect,
        context: CGContext
    ) {
        let fontSize = autoSizeFontToFit(
            text: text, fontName: fontName,
            width: rect.width, height: rect.height
        )

        let font = CTFontCreateWithName(fontName as CFString, fontSize, nil)
        // Use CTParagraphStyle for center alignment (no AppKit dependency)
        var alignment = CTTextAlignment.center
        let alignmentSetting = CTParagraphStyleSetting(
            spec: .alignment,
            valueSize: MemoryLayout<CTTextAlignment>.size,
            value: &alignment
        )
        let paragraphStyle = CTParagraphStyleCreate([alignmentSetting], 1)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: CGColor(red: 1, green: 1, blue: 1, alpha: 1),
            .paragraphStyle: paragraphStyle
        ]
        let attrString = NSAttributedString(string: text, attributes: attributes)

        let framesetter = CTFramesetterCreateWithAttributedString(attrString as CFAttributedString)
        let suggestedSize = CTFramesetterSuggestFrameSizeWithConstraints(
            framesetter,
            CFRange(location: 0, length: attrString.length),
            nil,
            CGSize(width: rect.width, height: CGFloat.greatestFiniteMagnitude),
            nil
        )

        // Center vertically within the slot
        let yOffset = max(0, (rect.height - suggestedSize.height) / 2)
        let centeredRect = CGRect(
            x: rect.origin.x,
            y: rect.origin.y + yOffset,
            width: rect.width,
            height: suggestedSize.height
        )
        let path = CGPath(rect: centeredRect, transform: nil)
        let frame = CTFramesetterCreateFrame(
            framesetter,
            CFRange(location: 0, length: attrString.length),
            path,
            nil
        )
        CTFrameDraw(frame, context)
    }

    /// Find the largest font size that fits the text within given bounds.
    private static func autoSizeFontToFit(
        text: String,
        fontName: String,
        width: CGFloat,
        height: CGFloat
    ) -> CGFloat {
        var low: CGFloat = 10
        var high: CGFloat = min(width, height) * 0.8
        var best: CGFloat = low

        for _ in 0..<20 {
            let mid = (low + high) / 2
            let font = CTFontCreateWithName(fontName as CFString, mid, nil)
            let attrs: [NSAttributedString.Key: Any] = [.font: font]
            let attrStr = NSAttributedString(string: text, attributes: attrs)
            let framesetter = CTFramesetterCreateWithAttributedString(attrStr as CFAttributedString)
            let size = CTFramesetterSuggestFrameSizeWithConstraints(
                framesetter,
                CFRange(location: 0, length: attrStr.length),
                nil,
                CGSize(width: width, height: CGFloat.greatestFiniteMagnitude),
                nil
            )

            if size.width <= width && size.height <= height {
                best = mid
                low = mid + 1
            } else {
                high = mid - 1
            }
        }

        return best
    }
}
