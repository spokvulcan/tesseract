//
//  DayThreadPictures.swift
//  tesseract
//
//  Pictures in the Day Thread (ADR-0090): the model sees a picture in the
//  turn it is shown — from the thread's latest user message on, with every
//  tool round that turn runs. Before that message a picture is a line in its
//  place, so neither a moment nor a cold re-prefill of the whole thread feeds
//  the day's pictures to the vision tower again. The owner still sees them in
//  Today's Chat: only what the model reads changes.
//

import Foundation

nonisolated enum DayThreadPictures {

    /// The thread as the model reads it: what `defaultConvertToLlm` renders,
    /// with every picture before the latest user message out of sight. Both
    /// the day agent's own turns and the moments render through here.
    static func llmMessages(_ messages: [any AgentMessageProtocol]) -> [LLMMessage] {
        let turnStart = messages.lastIndex { $0.asUser != nil } ?? messages.endIndex
        return messages.enumerated().compactMap { index, message in
            guard index < turnStart else { return message.toLLMMessage() }
            if let user = message.asUser, !user.images.isEmpty {
                return user.toLLMMessage(replacingImagesWith: note(for: user.images))
            }
            let rendered = message.toLLMMessage()
            if case .toolResult(let id, let content, let images)? = rendered, !images.isEmpty {
                let note = toolNote(count: images.count)
                return .toolResult(
                    toolCallId: id, content: content.isEmpty ? note : content + "\n" + note,
                    images: [])
            }
            return rendered
        }
    }

    /// The line that stands where an earlier turn's pictures were: how many,
    /// their names when they say something, and that the model can ask to
    /// see them again.
    static func note(for images: [ImageAttachment]) -> String {
        let names = images.compactMap(\.filename).filter(isTellingName).prefix(3)
        let named = names.isEmpty ? "" : ": " + names.joined(separator: ", ")
        if images.count == 1 {
            return "[The owner showed you a picture here\(named). You no longer see it; "
                + "ask to see it again if you need it.]"
        }
        return "[The owner showed you \(images.count) pictures here\(named). You no longer "
            + "see them; ask to see them again if you need them.]"
    }

    /// What stands where an earlier tool result's images were.
    static func toolNote(count: Int) -> String {
        count == 1
            ? "[The tool also returned an image here. You no longer see it.]"
            : "[The tool also returned \(count) images here. You no longer see them.]"
    }

    /// A pasted or dropped picture is named after the gesture, which says
    /// nothing; a picked file's name or an Appshot's window does.
    private static func isTellingName(_ name: String) -> Bool {
        !name.hasPrefix("pasted-image") && !name.hasPrefix("dropped-image")
    }
}
