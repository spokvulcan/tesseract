//
//  DictatedTake.swift
//  tesseract
//
//  One dictated take as the **Lens** (PRD #612) reopens it: the committed
//  text, what the Learned Words caught in it, the app it went to, and
//  whether it was pasted there.
//

import Foundation

nonisolated struct DictatedTake: Equatable, Sendable {
    /// The take's **Correction Pair**; nil when pairs are not recorded.
    let pairID: UUID?
    /// The committed text, without the trailing space the paste adds.
    let text: String
    /// What the Learned Words caught, positioned in `text`.
    let catches: [LearnedWordCatch]
    /// The app in front when the take started (its Learned Words' app).
    let app: TargetApp?
    /// Whether the text was pasted (auto-insert on).
    let pasted: Bool
    let at: Date
    /// The app in front when the paste landed: where a fix is put back. Nil
    /// when it was not read; `app` stands in.
    let pastedInto: TargetApp?
    /// ⇧ held it, or the setting says always: it waits in the Lens, not
    /// pasted yet.
    let held: Bool

    init(
        pairID: UUID?, text: String, catches: [LearnedWordCatch], app: TargetApp?,
        pasted: Bool, at: Date = Date(), pastedInto: TargetApp? = nil, held: Bool = false
    ) {
        self.pairID = pairID
        self.text = text
        self.catches = catches
        self.app = app
        self.pasted = pasted
        self.at = at
        self.pastedInto = pastedInto
        self.held = held
    }

    /// What the paste typed: the text and the space after it.
    var pastedText: String { text + " " }
}
