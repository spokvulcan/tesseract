import AppKit
import SwiftUI

// Mirrors GlobalOverlayHUD's proofreading → committed-beat switch inside
// OverlayPanel's borderless non-activating NSPanel.
let args = CommandLine.arguments
func flag(_ f: String) -> Bool { args.contains(f) }

@MainActor final class Model: ObservableObject {
    @Published var beat = flag("--buttons-from-start")
}
let model = Model()

struct HUD: View {
    @ObservedObject var model: Model
    var body: some View {
        if flag("--no-glass") { core } else { GlassEffectContainer { core } }
    }
    @ViewBuilder var glassed: some View { EmptyView() }
    var core: some View {
        ZStack(alignment: .bottom) {
            if !model.beat {
                HStack(spacing: 8) {
                    Image(systemName: "wand.and.sparkles")
                    Text("• • • • •")
                }
                .padding(.horizontal, 12).padding(.vertical, 6)
                .frame(width: 200, height: 36)
                .modifier(Glass())
                .modifier(Trans())
            } else {
                HStack(alignment: .center, spacing: 8) {
                    Image(systemName: "checkmark").foregroundStyle(.green)
                    Text("Inserted").font(.system(size: 11, weight: .semibold)).lineLimit(1)
                    if !flag("--no-spacer") { Spacer(minLength: 4) }
                    if !flag("--no-buttons") {
                        beatButton("flag")
                        if !flag("--one-button") { beatButton("pencil") }
                    }
                }
                .padding(.horizontal, 12).padding(.vertical, 6)
                .frame(width: 220, height: 36)
                .modifier(Glass())
                .modifier(Trans())
            }
        }
        .frame(width: 320, height: 80, alignment: .bottom)
    }
    @ViewBuilder func beatButton(_ name: String) -> some View {
        if flag("--tap-gesture") {
            Image(systemName: name).frame(width: 22, height: 22).contentShape(Circle()).onTapGesture
            {}
        } else {
            let b = Button(action: {}) {
                Image(systemName: name).frame(width: 22, height: 22).contentShape(Circle())
            }
            if flag("--unfocusable") {
                b.buttonStyle(.plain).focusable(false)
            } else if flag("--bordered") {
                b
            } else {
                b.buttonStyle(.plain)
            }
        }
    }
}
struct Glass: ViewModifier {
    func body(content: Content) -> some View {
        if flag("--no-glass") { content } else { content.glassEffect(.regular, in: .capsule) }
    }
}
struct Trans: ViewModifier {
    func body(content: Content) -> some View {
        if flag("--no-animation") {
            content
        } else {
            content.transition(.scale(scale: 0.85, anchor: .bottom).combined(with: .opacity))
        }
    }
}

final class NoKeyPanel: NSPanel { override var canBecomeKey: Bool { false } }
let app = NSApplication.shared
app.setActivationPolicy(.accessory)

let rect = NSRect(x: 400, y: 300, width: 320, height: 80)
let panel: NSWindow
if flag("--plain-window") {
    panel = NSWindow(contentRect: rect, styleMask: [.titled], backing: .buffered, defer: false)
} else {
    let p =
        flag("--cannot-become-key")
        ? NoKeyPanel(
            contentRect: rect, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered,
            defer: false)
        : NSPanel(
            contentRect: rect, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered,
            defer: false)
    p.level = .statusBar
    p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
    p.isReleasedWhenClosed = false
    p.ignoresMouseEvents = true
    p.isOpaque = false
    p.backgroundColor = .clear
    p.hasShadow = false
    p.hidesOnDeactivate = false
    panel = p
}
if flag("--hosting-controller") {
    panel.contentViewController = NSHostingController(rootView: AnyView(HUD(model: model)))
} else {
    let hosting = NSHostingView(rootView: AnyView(HUD(model: model)))
    hosting.frame = panel.contentView?.bounds ?? .zero
    hosting.autoresizingMask = [.width, .height]
    panel.contentView?.addSubview(hosting)
}
if flag("--no-autorecalc") { panel.autorecalculatesKeyViewLoop = false }
panel.orderFrontRegardless()

// Watchdog off the main thread: after the switch, is the main thread still serving blocks?
let responded = NSLock(); var pings = 0
Thread.detachNewThread {
    Thread.sleep(forTimeInterval: 1.0)
    DispatchQueue.main.async {
        panel.ignoresMouseEvents = false  // the beat window flips the panel interactive
        if flag("--no-animation") {
            model.beat = true
        } else {
            withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) { model.beat = true }
        }
    }
    Thread.sleep(forTimeInterval: 3.0)
    DispatchQueue.main.async {
        responded.lock(); pings += 1; responded.unlock()
    }
    Thread.sleep(forTimeInterval: 3.0)
    responded.lock(); let ok = pings > 0; responded.unlock()
    print(
        ok
            ? "[hang-harness] OK: main thread responsive after beat switch"
            : "[hang-harness] HANG: main thread unresponsive after beat switch")
    exit(ok ? 0 : 2)
}
app.run()
