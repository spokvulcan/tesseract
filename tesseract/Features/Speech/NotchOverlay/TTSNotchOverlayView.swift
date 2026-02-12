//
//  TTSNotchOverlayView.swift
//  tesseract
//
//  Notch overlay view for TTS playback with word-by-word highlighting.
//  Adapted from textream (github.com/f/textream) NotchOverlayView.
//

import SwiftUI
import os

// MARK: - NotchFrameTracker

@Observable
@MainActor
final class NotchFrameTracker {
    var visibleHeight: CGFloat = 37 {
        didSet { updatePanel() }
    }
    var visibleWidth: CGFloat = 200 {
        didSet { updatePanel() }
    }
    weak var panel: NSPanel?
    var screenMidX: CGFloat = 0
    var screenMaxY: CGFloat = 0

    func updatePanel() {
        guard let panel else { return }
        let x = screenMidX - visibleWidth / 2
        let y = screenMaxY - visibleHeight
        panel.setFrame(NSRect(x: x, y: y, width: visibleWidth, height: visibleHeight), display: false)
    }
}

// MARK: - TTSNotchOverlayView

struct TTSNotchOverlayView: View {
    let words: [String]
    let totalCharCount: Int
    @Bindable var wordTracker: TTSWordTracker
    let menuBarHeight: CGFloat
    let baseTextHeight: CGFloat
    let maxExtraHeight: CGFloat
    var frameTracker: NotchFrameTracker

    @State private var expansion: CGFloat = 0
    @State private var contentVisible = false
    @State private var extraHeight: CGFloat = 0
    @State private var dragStartHeight: CGFloat = -1
    @State private var isHovering: Bool = false

    private let topInset: CGFloat = 16
    private let collapsedInset: CGFloat = 8
    private let notchHeight: CGFloat = 37
    private let notchWidth: CGFloat = 200

    private let highlightColor: Color = .yellow
    private let font: NSFont = .systemFont(ofSize: 20, weight: .semibold)

    var isDone: Bool {
        wordTracker.isGenerationComplete && totalCharCount > 0 && wordTracker.recognizedCharCount >= totalCharCount
    }

    private var currentTopInset: CGFloat {
        collapsedInset + (topInset - collapsedInset) * expansion
    }

    private var currentBottomRadius: CGFloat {
        8 + (18 - 8) * expansion
    }

    var body: some View {
        GeometryReader { geo in
            let targetHeight = menuBarHeight + baseTextHeight + extraHeight
            let currentHeight = notchHeight + (targetHeight - notchHeight) * expansion
            let currentWidth = notchWidth + (geo.size.width - notchWidth) * expansion

            ZStack(alignment: .top) {
                DynamicIslandShape(
                    topInset: currentTopInset,
                    bottomRadius: currentBottomRadius
                )
                .fill(.black)
                .frame(width: currentWidth, height: currentHeight)

                if contentVisible {
                    VStack(spacing: 0) {
                        Spacer().frame(height: menuBarHeight)

                        if isDone {
                            doneView
                        } else {
                            prompterView
                        }
                    }
                    .padding(.horizontal, topInset)
                    .frame(width: geo.size.width, height: targetHeight)
                    .transition(.opacity)
                }
            }
            .frame(width: currentWidth, height: currentHeight, alignment: .top)
            .frame(width: geo.size.width, height: geo.size.height, alignment: .top)
        }
        .onChange(of: extraHeight) { _, _ in updateFrameTracker() }
        .onAppear {
            Log.speech.info("[NotchView] onAppear — words=\(self.words.count), totalCharCount=\(self.totalCharCount), expansion=\(self.expansion)")
            withAnimation(.easeOut(duration: 0.4)) {
                expansion = 1
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                Log.speech.info("[NotchView] revealing content (0.35s delay)")
                withAnimation(.easeOut(duration: 0.25)) {
                    contentVisible = true
                }
            }
        }
        .onChange(of: wordTracker.shouldDismiss) { oldVal, shouldDismiss in
            Log.speech.info("[NotchView] shouldDismiss changed: \(oldVal) → \(shouldDismiss)")
            if shouldDismiss {
                Log.speech.info("[NotchView] triggering dismiss animation")
                withAnimation(.easeIn(duration: 0.15)) {
                    contentVisible = false
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    withAnimation(.easeIn(duration: 0.3)) {
                        expansion = 0
                    }
                }
            }
        }
        .animation(.easeInOut(duration: 0.5), value: isDone)
        .onChange(of: isDone) { oldVal, done in
            Log.speech.info("[NotchView] isDone changed: \(oldVal) → \(done) (genComplete=\(self.wordTracker.isGenerationComplete), charCount=\(self.wordTracker.recognizedCharCount)/\(self.totalCharCount))")
            if done {
                Log.speech.info("[NotchView] scheduling dismiss in 1.0s")
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    Log.speech.info("[NotchView] setting shouldDismiss=true (1.0s after isDone)")
                    wordTracker.shouldDismiss = true
                }
            }
        }
        .onChange(of: expansion) { oldVal, newVal in
            if abs(oldVal - newVal) > 0.4 {
                Log.speech.info("[NotchView] expansion: \(String(format: "%.2f", oldVal)) → \(String(format: "%.2f", newVal))")
            }
        }
        .onChange(of: contentVisible) { _, visible in
            Log.speech.info("[NotchView] contentVisible: \(visible)")
        }
    }

    private func updateFrameTracker() {
        let targetHeight = menuBarHeight + baseTextHeight + extraHeight
        frameTracker.visibleHeight = targetHeight
        frameTracker.visibleWidth = frameTracker.visibleWidth // triggers update via didSet
    }

    private var prompterView: some View {
        VStack(spacing: 0) {
            SpeechScrollView(
                words: words,
                highlightedCharCount: wordTracker.recognizedCharCount,
                font: font,
                highlightColor: highlightColor,
                onWordTap: { charOffset in
                    wordTracker.jumpTo(charOffset: charOffset)
                }
            )
            .padding(.horizontal, 12)
            .padding(.top, 6)
            .transition(.move(edge: .top).combined(with: .opacity))

            Group {
                HStack(alignment: .center, spacing: 8) {
                    Spacer()

                    Button {
                        Log.speech.info("[NotchView] close button pressed")
                        wordTracker.shouldDismiss = true
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(.white.opacity(0.6))
                            .frame(width: 24, height: 24)
                            .background(.white.opacity(0.15))
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                }
                .frame(height: 24)
                .padding(.horizontal, 12)
                .padding(.bottom, 10)

                // Resize handle
                if isHovering {
                    VStack(spacing: 0) {
                        Spacer().frame(height: 4)
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.white.opacity(0.25))
                            .frame(width: 36, height: 4)
                        Spacer().frame(height: 8)
                    }
                    .frame(height: 16)
                    .frame(maxWidth: .infinity)
                    .contentShape(Rectangle())
                    .simultaneousGesture(
                        DragGesture(minimumDistance: 2, coordinateSpace: .global)
                            .onChanged { value in
                                if dragStartHeight < 0 {
                                    dragStartHeight = extraHeight
                                }
                                let newExtra = dragStartHeight + value.translation.height
                                extraHeight = max(0, min(maxExtraHeight, newExtra))
                            }
                            .onEnded { _ in
                                dragStartHeight = -1
                            }
                    )
                    .onHover { hovering in
                        if hovering {
                            NSCursor.resizeUpDown.push()
                        } else {
                            NSCursor.pop()
                        }
                    }
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .onHover { hovering in
                withAnimation(.easeInOut(duration: 0.2)) {
                    isHovering = hovering
                }
            }
            .transition(.opacity)
        }
    }

    private var doneView: some View {
        VStack {
            Spacer()
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("Done!")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.white)
            }
            Spacer()
        }
        .transition(.scale.combined(with: .opacity))
    }
}
