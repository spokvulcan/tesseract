//
//  HoldToTalk.swift
//  tesseract
//
//  The one hold-to-talk gesture for a **Voice Input**'s mic button: the
//  agent composer's, Today's, the Onboarding Tour's.
//

import SwiftUI

extension View {
    /// Hold to talk: pressing starts a take on `voice`, letting go finishes
    /// it, and the words land wherever its surface puts them. `onPress` runs
    /// when a take starts; `canStart` false leaves the press alone.
    func holdToTalk(
        _ voice: AgentVoiceInputController?, canStart: Bool = true,
        onPress: @escaping () -> Void = {}
    ) -> some View {
        modifier(HoldToTalk(voice: voice, canStart: canStart, onPress: onPress))
    }
}

private struct HoldToTalk: ViewModifier {
    let voice: AgentVoiceInputController?
    let canStart: Bool
    let onPress: () -> Void
    @State private var isHolding = false

    func body(content: Content) -> some View {
        content.gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !isHolding, canStart, let voice else { return }
                    isHolding = true
                    onPress()
                    voice.start()
                }
                .onEnded { _ in
                    isHolding = false
                    voice?.finishCapture()
                }
        )
    }
}
