//
//  SpeechOverlayPanel.swift
//  tesseract
//
//  The NSPanel hosting the **Speech Overlay** (`SpeechOverlayView`). It
//  shows while the Read-Along follows speech and the settings want it: not
//  with the style Off, and, in the automatic scope, not while the Speech page
//  is in front, since the page already shows what is being read. Changing a
//  setting with nothing being read previews it for a few seconds.
//

import AppKit
import Observation
import SwiftUI

@MainActor
final class SpeechOverlayPanel {
    private let model: SpeechOverlayModel
    private let settings: SettingsManager
    private let readAlong: SpeechReadAlong
    private let isSpeechPageInFront: @MainActor () -> Bool

    private var panel: NSPanel?
    private var panelLook: Look?
    private var hideTask: Task<Void, Never>?
    private var previewTimer: Timer?
    private var observations: [Task<Void, Never>] = []

    /// What the panel is built for: its style and text size set its frame.
    private struct Look: Equatable {
        let style: SpeechOverlayStyle
        let size: SpeechOverlaySize
    }

    init(
        readAlong: SpeechReadAlong, settings: SettingsManager, coordinator: SpeechCoordinator,
        isSpeechPageInFront: @escaping @MainActor () -> Bool,
        openSpeechPage: @escaping () -> Void
    ) {
        self.readAlong = readAlong
        self.settings = settings
        self.isSpeechPageInFront = isSpeechPageInFront
        self.model = SpeechOverlayModel(
            readAlong: readAlong, settings: settings, coordinator: coordinator,
            openSpeechPage: openSpeechPage)
    }

    /// Starts following the Read-Along and the settings.
    func start() {
        guard observations.isEmpty else { return }
        observations.append(
            Task { [weak self] in
                guard let self else { return }
                for await look in Observations({ self.wantedLook }) {
                    if let look { self.present(look) } else { self.dismiss() }
                }
            })
        observations.append(
            Task { [weak self] in
                guard let self else { return }
                var isFirst = true
                for await _ in Observations({
                    (
                        self.settings.speechOverlayStyleRaw, self.settings.speechOverlaySizeRaw,
                        self.settings.speechOverlayTintRaw
                    )
                }) {
                    // The first value is the launch state, not a change.
                    if isFirst { isFirst = false } else { self.preview() }
                }
            })
    }

    /// The overlay the moment calls for, or nil for none.
    private var wantedLook: Look? {
        let previewing = model.previewWords != nil
        guard readAlong.isActive || previewing else { return nil }
        let style = settings.speechOverlayStyle
        guard style != .off else { return nil }
        if !previewing, settings.speechOverlayScope == .automatic, isSpeechPageInFront() {
            return nil
        }
        return Look(style: style, size: settings.speechOverlaySize)
    }

    // MARK: - Preview

    private static let previewText =
        "This is how the overlay reads along with you. Each word lights up as it is spoken, two lines at a time."

    /// Walks sample words through the chosen look, unless speech is playing
    /// (then the change shows on the real thing).
    private func preview() {
        previewTimer?.invalidate()
        guard !readAlong.isActive, settings.speechOverlayStyle != .off else {
            model.previewWords = nil
            return
        }
        model.previewWords = Self.previewText.split(separator: " ").map(String.init)
        model.previewWord = 0
        let timer = Timer(timeInterval: 0.28, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, let words = self.model.previewWords else { return }
                if self.model.previewWord + 1 < words.count {
                    self.model.previewWord += 1
                } else {
                    self.previewTimer?.invalidate()
                    self.previewTimer = nil
                    self.model.previewWords = nil
                }
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        previewTimer = timer
    }

    // MARK: - Presenting

    private func present(_ look: Look) {
        hideTask?.cancel()
        hideTask = nil
        if panel == nil || panelLook != look {
            panel?.orderOut(nil)
            panel = makePanel(look)
            panelLook = look
            model.isPresented = false
        }
        model.style = look.style
        panel?.orderFrontRegardless()
        withAnimation(.spring(response: 0.42, dampingFraction: 0.82)) {
            model.isPresented = true
        }
    }

    private func dismiss() {
        guard panel != nil, hideTask == nil else { return }
        withAnimation(.easeIn(duration: 0.22)) {
            model.isPresented = false
        }
        hideTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled, let self else { return }
            self.panel?.orderOut(nil)
            self.panel = nil
            self.panelLook = nil
            self.hideTask = nil
        }
    }

    private func makePanel(_ look: Look) -> NSPanel? {
        guard let screen = NSScreen.main else { return nil }
        // The notch's height where there is one, else the menu bar's.
        let topInset = max(
            screen.safeAreaInsets.top, screen.frame.maxY - screen.visibleFrame.maxY, 24)
        let frame: NSRect
        let contentWidth: CGFloat
        switch look.style {
        case .captions:
            contentWidth = min(Self.captionsWidth(look.size), screen.visibleFrame.width - 80)
            frame = NSRect(
                x: screen.visibleFrame.midX - contentWidth / 2, y: screen.visibleFrame.minY + 28,
                width: contentWidth, height: 220)
        default:
            contentWidth = Self.islandWidth(look.size)
            let height = topInset + 200
            frame = NSRect(
                x: screen.frame.midX - (contentWidth + 40) / 2, y: screen.frame.maxY - height,
                width: contentWidth + 40, height: height)
        }

        let panel = NSPanel(
            contentRect: frame, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered,
            defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = .screenSaver
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = false
        panel.contentView = NSHostingView(
            rootView: SpeechOverlayRoot(model: model, topInset: topInset, width: contentWidth))
        return panel
    }

    private static func islandWidth(_ size: SpeechOverlaySize) -> CGFloat {
        switch size {
        case .small: 440
        case .medium: 520
        case .large: 640
        }
    }

    private static func captionsWidth(_ size: SpeechOverlaySize) -> CGFloat {
        switch size {
        case .small: 640
        case .medium: 780
        case .large: 940
        }
    }
}
