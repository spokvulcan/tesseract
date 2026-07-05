//
//  OnboardingTourView.swift
//  tesseract
//
//  The Welcome Window's content: backdrop, ambient mark, the six Chapters
//  with directional transitions, and the navigation chrome. Owns the Handoff
//  (main window first, then this window dissolves — never zero windows) and
//  the close-=-skip semantics (`CONTEXT.md` → Onboarding tour, ADR-0021).
//

import SwiftUI

struct OnboardingTourView: View {
    let container: DependencyContainer

    @State private var controller: OnboardingTourController?
    @State private var direction: CGFloat = 1
    @State private var isDissolving = false
    @Namespace private var markNamespace

    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            if let controller {
                OnboardingBackdrop(chapter: controller.chapter)
                    .gesture(WindowDragGesture())

                VStack(spacing: 0) {
                    header(controller)
                    chapterStage(controller)
                    footer(controller)
                }
            }
        }
        .frame(width: 780, height: 580)
        .opacity(isDissolving ? 0 : 1)
        .scaleEffect(isDissolving ? 0.98 : 1)
        .onAppear {
            if controller == nil {
                controller = makeController()
            }
        }
        .onDisappear(perform: completeIfAbandoned)
    }

    // MARK: - Regions

    @ViewBuilder
    private func header(_ controller: OnboardingTourController) -> some View {
        HStack(spacing: 8) {
            Spacer()
            if controller.chapter != .welcome {
                if !controller.isSetupComplete {
                    Text(
                        controller.setupProgress.formatted(
                            .percent.precision(.fractionLength(0)))
                    )
                    .font(.system(size: 10.5).monospacedDigit())
                    .foregroundStyle(.tertiary)
                    .contentTransition(.numericText())
                }
                TesseractMarkView(progress: controller.setupProgress)
                    .frame(width: 30, height: 30)
                    .matchedGeometryEffect(
                        id: OnboardingMarkID.shared, in: markNamespace,
                        isSource: controller.chapter != .welcome)
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 12)
        .frame(height: 46)
        .animation(.easeInOut(duration: 0.3), value: controller.isSetupComplete)
    }

    @ViewBuilder
    private func chapterStage(_ controller: OnboardingTourController) -> some View {
        ZStack {
            chapterContent(controller)
                .id(controller.chapter)
                .transition(chapterTransition)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func chapterContent(_ controller: OnboardingTourController) -> some View {
        switch controller.chapter {
        case .welcome:
            WelcomeChapter(controller: controller, markNamespace: markNamespace)
        case .agent:
            AgentChapter()
        case .dictation:
            DictationChapter(controller: controller)
        case .voice:
            VoiceChapter(controller: controller)
        case .server:
            ServerChapter()
        case .ready:
            ReadyChapter(controller: controller)
        }
    }

    @ViewBuilder
    private func footer(_ controller: OnboardingTourController) -> some View {
        ZStack {
            ChapterDots(current: controller.chapter) { target in
                move(controller, to: target)
            }

            HStack(spacing: 12) {
                if !controller.isLastChapter {
                    Button("Skip Tour") { finish(controller) }
                        .buttonStyle(.plain)
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                        .keyboardShortcut(.cancelAction)
                } else {
                    // Escape stays "leave the tour" on the last chapter too,
                    // where the visible Skip control has retired.
                    Button("") { finish(controller) }
                        .keyboardShortcut(.cancelAction)
                        .opacity(0)
                        .frame(width: 0, height: 0)
                        .accessibilityHidden(true)
                }

                Spacer()

                if controller.canGoBack {
                    Button {
                        goBack(controller)
                    } label: {
                        Image(systemName: "chevron.left")
                            .frame(minWidth: 20)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .keyboardShortcut(.leftArrow, modifiers: [])
                    .accessibilityLabel("Back")
                }

                Button {
                    if controller.isLastChapter {
                        finish(controller)
                    } else {
                        advance(controller)
                    }
                } label: {
                    Text(controller.isLastChapter ? "Start Using Tesseract" : "Continue")
                        .frame(minWidth: controller.isLastChapter ? 150 : 90)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.defaultAction)
            }

            // Right-arrow advance, window-wide, without a second visible control.
            Button("") {
                if !controller.isLastChapter { advance(controller) }
            }
            .keyboardShortcut(.rightArrow, modifiers: [])
            .opacity(0)
            .frame(width: 0, height: 0)
            .accessibilityHidden(true)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
        .padding(.top, 6)
    }

    // MARK: - Navigation

    private var navigationAnimation: Animation {
        reduceMotion
            ? .easeInOut(duration: 0.18)
            : .spring(response: 0.55, dampingFraction: 0.88)
    }

    private var chapterTransition: AnyTransition {
        if reduceMotion {
            return .opacity
        }
        return .asymmetric(
            insertion: .offset(x: 56 * direction).combined(with: .opacity),
            removal: .offset(x: -56 * direction).combined(with: .opacity)
        )
    }

    private func advance(_ controller: OnboardingTourController) {
        direction = 1
        withAnimation(navigationAnimation) { controller.advance() }
    }

    private func goBack(_ controller: OnboardingTourController) {
        direction = -1
        withAnimation(navigationAnimation) { controller.goBack() }
    }

    private func move(
        _ controller: OnboardingTourController, to target: OnboardingTourController.Chapter
    ) {
        direction = target.rawValue >= controller.chapter.rawValue ? 1 : -1
        withAnimation(navigationAnimation) { controller.go(to: target) }
    }

    // MARK: - Handoff & skip

    /// Finish and skip are one act: mark completed, bring the main window in
    /// first, then dissolve — never a beat with zero windows on screen.
    private func finish(_ controller: OnboardingTourController) {
        controller.complete()
        openWindow(id: WindowID.main)
        withAnimation(.easeIn(duration: reduceMotion ? 0.05 : 0.35)) {
            isDissolving = true
        }
        Task {
            try? await Task.sleep(for: .milliseconds(reduceMotion ? 80 : 400))
            dismissWindow(id: WindowID.onboarding)
        }
    }

    /// The red close button path: the window is already going away, so just
    /// honor close-=-skip and make sure a window exists afterwards.
    private func completeIfAbandoned() {
        guard let controller, !controller.didComplete else { return }
        controller.complete()
        openWindow(id: WindowID.main)
    }

    private func makeController() -> OnboardingTourController {
        OnboardingTourController(
            settings: container.settingsManager,
            downloadManager: container.modelDownloadManager,
            speechToTextModelID: container.settingsManager.selectedSpeechToTextModelID,
            voiceModelID: ModelDefinition.defaultTextToSpeechModelID
        )
    }
}
