import AppKit
import Foundation

/// Shared image fixtures for the conversation-shape tests. The cross-adapter
/// parity suites (`MessageConverterTests`, `AgentConversationBuilderTests`)
/// must exercise both edges with the SAME bytes — a divergent fixture would
/// not fail loudly, it would just silently test different payloads.
enum ImageTestFixtures {
    /// 1×1 valid PNG.
    static let tinyPNGBase64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

    static var tinyPNGData: Data { Data(base64Encoded: tinyPNGBase64)! }

    /// A picture worth looking at in a gallery render: a flyer with a title
    /// and a date line on a colored card.
    @MainActor
    static func flyerPNG(title: String, line: String, hue: CGFloat) -> Data {
        let size = NSSize(width: 480, height: 640)
        let image = NSImage(size: size, flipped: false) { rect in
            NSColor(calibratedHue: hue, saturation: 0.55, brightness: 0.9, alpha: 1).setFill()
            rect.fill()
            let title = NSAttributedString(
                string: title,
                attributes: [
                    .font: NSFont.boldSystemFont(ofSize: 54), .foregroundColor: NSColor.white,
                ])
            title.draw(in: NSRect(x: 36, y: 300, width: 408, height: 260))
            let line = NSAttributedString(
                string: line,
                attributes: [.font: NSFont.systemFont(ofSize: 30), .foregroundColor: NSColor.white])
            line.draw(in: NSRect(x: 36, y: 120, width: 408, height: 140))
            return true
        }
        let rep = NSBitmapImageRep(data: image.tiffRepresentation!)!
        return rep.representation(using: .png, properties: [:])!
    }
}
