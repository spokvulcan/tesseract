//
//  DayThreadPicturesTests.swift
//  tesseractTests
//
//  Pictures in the Day Thread (ADR-0090), with no model: the model sees a
//  picture in the turn it's shown and a line in its place after — a tool's
//  images too — and an earlier turn reads the same however the thread grows,
//  so the prefix cache keeps it. Through a real Day Thread over the in-memory
//  arbiter: the owner's turn hands the model the picture and the next turn
//  the line, a moment after a picture reads the line, a picture with no words
//  asks what it means for the day, the trace counts it, and a turn stopped
//  before it reached the thread gives its pictures back to Today's composer.
//  And the Jarvis panel takes pictures by the composer's rules.
//

import Foundation
import MLXLMCommon
import Testing

@testable import Tesseract_Agent

@MainActor
@Suite(.timeLimit(.minutes(1)))
struct DayThreadPicturesTests {

    private func picture(_ name: String? = nil) -> ImageAttachment {
        ImageAttachment(data: ImageTestFixtures.tinyPNGData, mimeType: "image/png", filename: name)
    }

    private func images(_ message: LLMMessage) -> [ImageAttachment] {
        switch message {
        case .user(_, let images), .toolResult(_, _, let images): images
        case .system, .assistant: []
        }
    }

    private func text(_ message: LLMMessage) -> String {
        switch message {
        case .user(let content, _), .system(let content), .toolResult(_, let content, _): content
        case .assistant(let content, _, _): content
        }
    }

    private let at = Date(timeIntervalSince1970: 1_791_000_000)

    // MARK: - The rule

    @Test func theLatestTurnSeesItsPictures() {
        let asked = UserMessage(
            content: "Add this to my calendar", images: [picture()], timestamp: at)
        let rendered = DayThreadPictures.llmMessages([
            UserMessage(content: "[Day Opening]", timestamp: at, turnOrigin: .moment),
            AssistantMessage.create(content: "Good morning."),
            asked,
        ])
        #expect(rendered.count == 3)
        #expect(rendered.last == asked.toLLMMessage())
        #expect(images(rendered[2]).count == 1)
    }

    @Test func anEarlierPictureIsALineInItsPlace() throws {
        let shown = UserMessage(
            content: "Add this to my calendar", images: [picture("school-letter.png")],
            timestamp: at)
        let rendered = DayThreadPictures.llmMessages([
            shown,
            AssistantMessage.create(content: "Added the parents' evening, 14 October at 17:00."),
            UserMessage(content: "Thanks", timestamp: at),
        ])
        #expect(images(rendered[0]).isEmpty)
        // The Now Tag first, the line where the picture was, then the words.
        let lines = text(rendered[0]).components(separatedBy: "\n")
        #expect(lines.count == 3)
        #expect(lines[0] == shown.nowTag)
        #expect(lines[1] == DayThreadPictures.note(for: shown.images))
        #expect(lines[2] == "Add this to my calendar")
        // Everything else reads as ever.
        #expect(text(rendered[1]) == "Added the parents' evening, 14 October at 17:00.")
        #expect(text(rendered[2]).hasSuffix("Thanks"))
    }

    @Test func aMomentAfterAPictureReadsTheLine() {
        let thread: [any AgentMessageProtocol] = [
            UserMessage(content: "What's this?", images: [picture(), picture()], timestamp: at),
            AssistantMessage.create(content: "Two tickets for Saturday."),
            UserMessage(
                content: "[Evening Wrap-up]\nWrap up the day.", timestamp: at, turnOrigin: .moment),
        ]
        let rendered = DayThreadPictures.llmMessages(thread)
        #expect(rendered.allSatisfy { images($0).isEmpty })
        #expect(text(rendered[0]).contains("2 pictures"))
    }

    @Test func aToolsImagesAreSeenOnlyInTheirTurn() {
        let screenshot = ToolResultMessage(
            toolCallId: "call-1", toolName: "browser.screenshot",
            content: [
                .text("Screenshot of the booking page"),
                .image(data: ImageTestFixtures.tinyPNGData, mimeType: "image/png"),
            ])
        let turn: [any AgentMessageProtocol] = [
            UserMessage(content: "Check the booking", timestamp: at),
            AssistantMessage.create(content: ""),
            screenshot,
        ]
        // In its own turn the screenshot is seen.
        #expect(images(DayThreadPictures.llmMessages(turn)[2]).count == 1)
        // A turn later it is a line after the tool's words.
        let later = DayThreadPictures.llmMessages(
            turn + [
                AssistantMessage.create(content: "It's booked."),
                UserMessage(content: "Good", timestamp: at),
            ])
        #expect(images(later[2]).isEmpty)
        #expect(
            text(later[2])
                == "Screenshot of the booking page\n" + DayThreadPictures.toolNote(count: 1))
    }

    @Test func theLineNamesWhatThePicturesSay() {
        #expect(
            DayThreadPictures.note(for: [picture("school-letter.png")])
                == "[The owner showed you a picture here: school-letter.png. You no longer see it; "
                + "ask to see it again if you need it.]")
        // A pasted or dropped picture's name says nothing.
        #expect(
            DayThreadPictures.note(for: [picture("pasted-image.png")])
                == "[The owner showed you a picture here. You no longer see it; "
                + "ask to see it again if you need it.]")
        #expect(
            DayThreadPictures.note(for: [
                picture("dropped-image"), picture("Safari — Booking.png"), picture(),
            ])
                == "[The owner showed you 3 pictures here: Safari — Booking.png. You no longer see "
                + "them; ask to see them again if you need them.]")
        #expect(DayThreadPictures.toolNote(count: 2).contains("2 images"))
    }

    /// The prefix cache keys on what the model reads: once a picture is a
    /// line, its turn must read the same in every later request.
    @Test func anEarlierTurnReadsTheSameAsTheThreadGrows() {
        let head: [any AgentMessageProtocol] = [
            UserMessage(content: "[Day Opening]", timestamp: at, turnOrigin: .moment),
            UserMessage(content: "Add this", images: [picture("flyer.jpg")], timestamp: at),
            AssistantMessage.create(content: "Added the market, Saturday at 9."),
            UserMessage(
                content: "[Breakpoint]\nWhile you were away", timestamp: at, turnOrigin: .moment),
        ]
        let later =
            head + [
                AssistantMessage.create(content: #"{"line": "Nothing needs you."}"#),
                UserMessage(content: "Move it to 10", timestamp: at),
            ]
        let first = DayThreadPictures.llmMessages(head)
        let second = DayThreadPictures.llmMessages(later)
        #expect(Array(second.prefix(head.count)) == first)
        #expect(DayThreadPictures.llmMessages(later) == second)
    }

    // MARK: - Asking

    @Test func picturesWithNoWordsAskWhatTheyMeanForTheDay() {
        #expect(DayThread.question("", pictures: 1) == "What in this picture matters for my day?")
        #expect(DayThread.question("", pictures: 3) == "What in these pictures matters for my day?")
        #expect(DayThread.question("Add these dates", pictures: 2) == "Add these dates")
        #expect(DayThread.question("Hello", pictures: 0) == "Hello")
    }

    // MARK: - Through the Day Thread

    @Test func theOwnersTurnSeesThePictureAndTheNextTurnReadsTheLine() async throws {
        let fixture = DayThreadFixture()
        let thread = fixture.thread
        thread.send("", images: [picture("school-letter.png")], from: .today)
        #expect(await observe(until: { !thread.chat.isGenerating }))
        thread.send("And the second date?")
        #expect(await observe(until: { !thread.chat.isGenerating }))

        let calls = fixture.generated.value
        try #require(calls.count == 2)
        let firstAsk = try #require(calls[0].last)
        #expect(images(firstAsk).count == 1)
        #expect(text(firstAsk).hasSuffix("What in this picture matters for my day?"))
        // The next turn: the picture is a line, and nothing carries pixels.
        #expect(calls[1].allSatisfy { images($0).isEmpty })
        #expect(calls[1].contains { text($0).contains("school-letter.png") })
        // The thread keeps the picture for Today's Chat.
        let stored = thread.chat.items.compactMap { item -> UserMessage? in
            if case .user(let user) = item { return user }
            return nil
        }
        #expect(stored.first?.images.count == 1)
        // The trace counts it, from Today, with no words.
        let records = fixture.trace.records(since: .distantPast)
        let shown = try #require(records.first { $0.traceEvent == .picturesShown })
        #expect(shown.fields?["count"] == .int(1))
        #expect(shown.fields?["surface"] == .string("today"))
        #expect(shown.fields?["words"] == .bool(false))
    }

    @Test func aMomentAfterAPictureTurnReadsTheLine() async throws {
        let fixture = DayThreadFixture()
        let thread = fixture.thread
        thread.send("Add this to my calendar", images: [picture()], from: .panel)
        #expect(await observe(until: { !thread.chat.isGenerating }))

        let outcome = await thread.runMoment(
            MomentRequest(
                kind: .breakpoint, trigger: .presenceReturned,
                text: "[Breakpoint]\nWhile you were away"))
        guard case .reply = outcome else {
            Issue.record("the moment should reply: \(outcome)")
            return
        }
        let conversation = try #require(fixture.completions.calls.last)
        #expect(conversation.images.isEmpty)
        #expect(conversation.messages.contains { $0.content.contains("You no longer see it") })
    }

    @Test func aStoppedTurnGivesItsPicturesBack() async throws {
        let fixture = DayThreadFixture(gateDelay: .seconds(30))
        let thread = fixture.thread
        let shown = picture("receipt.heic")
        thread.send("Log this", images: [shown], from: .today)
        #expect(thread.chat.isGenerating)
        thread.chat.cancelGeneration()
        #expect(fixture.restored.value?.text == "Log this")
        #expect(fixture.restored.value?.images == [shown])
    }

    // MARK: - The Jarvis panel

    @Test func thePanelTakesPicturesByTheComposersRules() {
        let payload = ImageGesturePayload(
            attachments: (0..<3).map { _ in picture() }, rejections: [.oversize(bytes: 20_000_000)])
        // Jarvis can't see pictures: nothing comes in, and the line says why.
        let refused = JarvisPanelController.take(payload, into: [], remedy: "Vision is turned off.")
        #expect(refused.pictures.isEmpty)
        #expect(refused.notice == "Vision is turned off.")
        // He can: up to eight, and the line says what didn't fit or come in.
        let waiting = (0..<6).map { _ in picture() }
        let taken = JarvisPanelController.take(payload, into: waiting, remedy: nil)
        #expect(taken.pictures.count == ComposerDraftController.maxPendingImages)
        #expect(taken.notice?.contains("Attached 2 of 3") == true)
        #expect(taken.notice?.contains("over 10 MB") == true)
        let clean = JarvisPanelController.take(
            ImageGesturePayload(attachments: [picture()]), into: [], remedy: nil)
        #expect(clean.pictures.count == 1)
        #expect(clean.notice == nil)
    }

    @Test func thePanelMakesRoomForWaitingPictures() {
        #expect(
            JarvisPanelController.height(forContent: 200, pictures: true)
                == JarvisPanelController.height(forContent: 200)
                + JarvisPanelController.pictureRowHeight)
        #expect(
            JarvisPanelController.height(forContent: 2000, pictures: true)
                == JarvisPanelController.size.height)
    }
}

// MARK: - Fixture

/// A Day Thread over the in-memory arbiter: its agent renders through the
/// Day Thread's rule and records what each owner turn hands the model, and
/// its moments go to a recording completion arm.
@MainActor
private struct DayThreadFixture {
    let thread: DayThread
    let trace = scratchTrace()
    let generated = Locked<[[LLMMessage]]>([])
    let completions = RecordingCompletionStarter()
    let restored = Locked<(text: String, images: [ImageAttachment])?>(nil)

    init(gateDelay: Duration? = nil) {
        let generated = generated
        let agent = Agent(
            config: AgentLoopConfig(
                model: AgentModelRef(id: "day-thread-pictures"),
                convertToLlm: DayThreadPictures.llmMessages,
                contextTransform: nil, getSteeringMessages: nil, getFollowUpMessages: nil),
            systemPrompt: "test",
            tools: [],
            generate: { _, messages, _, _ in
                generated.value.append(messages)
                return GenerationFixtures.eventStream([.text("Noted.")])
            })
        let arbiter = InMemoryInferenceArbiter()
        arbiter.gateDelay = gateDelay
        let host = ExtensionHost()
        let restored = restored
        thread = DayThread(
            agent: agent,
            store: DayThreadStore(
                backing: AgentConversationStore(directory: makeTempDir("day-thread-pictures")),
                day: DayKey(rawValue: "2026-10-10")),
            arbiter: arbiter,
            inferenceService: ServerInferenceService(
                completionStarter: completions, engine: UnusedManagedInference(),
                modelStateProvider: { nil }),
            toolRegistry: ToolRegistry(
                sandbox: PathSandbox(root: makeTempDir("day-thread-pictures-sandbox")),
                extensionHost: host),
            settings: SettingsManager(store: InMemorySettingsStore()),
            speechCoordinator: nil,
            contextManager: ContextManager(settings: .standard),
            summarize: { _ in "" },
            trace: trace,
            restoreDraft: { text, images in restored.value = (text, images) })
    }
}

/// The cache-aware arm, recording each conversation a moment sends.
private final class RecordingCompletionStarter: ServerCompletionStarting, @unchecked Sendable {
    private let recorded = Locked<[HTTPPrefixCacheConversation]>([])
    var calls: [HTTPPrefixCacheConversation] { recorded.value }

    func startServerCompletion(
        modelID: String, conversation: HTTPPrefixCacheConversation, toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters, renderContext: TemplateRenderContext,
        progressHandler: ServerInferenceProgressHandler?
    ) async throws -> HTTPServerGenerationStart {
        recorded.value.append(conversation)
        return HTTPServerGenerationStart(
            stream: GenerationFixtures.eventStream([.text(#"{"line": "Welcome back."}"#)]),
            cachedTokenCount: 0)
    }
}

/// The managed arm: a Day Thread request is always cache-aware here.
private final class UnusedManagedInference: ManagedInferenceStarting {
    struct Unused: Error {}

    func startPromptInference(prompt: String, parameters: AgentGenerateParameters) throws
        -> HTTPServerGenerationStart
    { throw Unused() }

    func startChatInference(
        systemPrompt: String, messages: [LLMMessage], toolSpecs: [ToolSpec]?,
        parameters: AgentGenerateParameters, renderContext: TemplateRenderContext,
        progressHandler: ServerInferenceProgressHandler?
    ) throws -> HTTPServerGenerationStart { throw Unused() }
}
