//
//  WelcomeChapter.swift
//  tesseract
//
//  Chapter 1 — the brand moment. Privacy is the identity, not a later slide;
//  the live tesseract is the hero (it will shrink into the corner and become
//  the ambient download indicator); the hardware-aware model pick is disclosed
//  transparently and setup begins here.
//

import SwiftUI

struct WelcomeChapter: View {
    let controller: OnboardingTourController
    var markNamespace: Namespace.ID

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var headlineProgress: Double = 0

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 4)

            TesseractMarkView(progress: controller.setupProgress)
                .frame(width: 168, height: 168)
                .matchedGeometryEffect(
                    id: OnboardingMarkID.shared, in: markNamespace,
                    isSource: controller.chapter == .welcome)

            Spacer(minLength: 18)

            Text("Welcome to Tesseract")
                .font(OnboardingType.titleFont)
                .tracking(OnboardingType.titleTracking)
                .textRenderer(GlyphReveal(progress: headlineProgress))

            OnboardingType.subtitle("Everything runs on this Mac. Nothing ever leaves it.")
                .padding(.top, OnboardingType.rhythm)

            Spacer(minLength: 22)

            ModelPickCard(controller: controller)

            Spacer(minLength: 8)
        }
        .padding(.horizontal, 48)
        .onAppear {
            controller.beginSetupIfNeeded()
            if reduceMotion {
                headlineProgress = 1
            } else {
                withAnimation(.easeOut(duration: 1.1).delay(0.15)) {
                    headlineProgress = 1
                }
            }
        }
    }
}

/// The transparent hardware-aware pick (ADR-0021): what was chosen, how big it
/// is, why — and a quiet "Change" for the opinionated.
struct ModelPickCard: View {
    let controller: OnboardingTourController

    private var chosen: ModelDefinition? {
        ModelDefinition.withID(controller.chosenAgentModelID)
    }

    private var memoryGigabytes: Int {
        Int(controller.physicalMemoryBytes / (1 << 30))
    }

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "brain")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(.tint)
                .frame(width: 26)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(chosen?.displayName ?? controller.chosenAgentModelID)
                        .font(OnboardingType.body)
                        .fontWeight(.semibold)
                    if let size = chosen?.sizeDescription {
                        Text(size)
                            .font(OnboardingType.body)
                            .foregroundStyle(.tertiary)
                    }
                }
                Text(pickStatusLine)
                    .font(OnboardingType.body)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 12)

            Menu("Change") {
                ForEach(ModelDefinition.models(in: .agent)) { model in
                    Button {
                        controller.selectAgentModel(model.id)
                    } label: {
                        if model.id == controller.chosenAgentModelID {
                            Label(
                                "\(model.displayName)  \(model.sizeDescription)",
                                systemImage: "checkmark")
                        } else {
                            Text("\(model.displayName)  \(model.sizeDescription)")
                        }
                    }
                }
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .font(OnboardingType.body)
            .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: 480)
        .onboardingCard()
    }

    private var pickStatusLine: String {
        switch controller.status(for: controller.chosenAgentModelID) {
        case .downloaded:
            return "Already installed — chosen for your \(memoryGigabytes) GB Mac"
        case .downloading(let progress):
            let percent = progress.formatted(.wholePercent)
            return "Downloading now — chosen for your \(memoryGigabytes) GB Mac · \(percent)"
        case .verifying:
            return "Verifying — chosen for your \(memoryGigabytes) GB Mac"
        case .notDownloaded:
            return "Chosen for your \(memoryGigabytes) GB Mac — downloads in the background"
        case .error:
            return "Download hit a snag — it will retry from the Models page"
        }
    }
}

/// The shared-element id the welcome hero and the ambient corner mark morph
/// between.
enum OnboardingMarkID {
    static let shared = "onboarding-tesseract-mark"
}
