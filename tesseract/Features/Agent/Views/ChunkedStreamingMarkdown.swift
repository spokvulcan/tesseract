import SwiftUI

// MARK: - Chunked streaming markdown

/// Live-streaming markdown whose layout cost is O(new tail), not
/// O(accumulated document): completed block runs freeze into chunk
/// `ChatMarkdownView`s that SwiftUI never re-parses or re-measures (their
/// strings are append-only and compare by shared storage), so each publish
/// re-renders only the short live tail. This is the live text row's analog of
/// `ChunkedStreamingText` — without it, a long streaming answer re-typesets
/// the entire accumulated document on every publish
/// (`ResolvedStyledText.StringDrawing.sizeThatFits` pegs the main thread at
/// 100% once the answer passes a few KB).
///
/// Chunks cut only at blank-line boundaries outside fenced code blocks and
/// never before an indented (4-column) continuation line, so no CommonMark
/// block that can span a blank line is ever split. Constructs that span
/// *cut* boundaries anyway — a loose list continuing after a blank line, a
/// reference link resolved by a later definition — render with a minor seam
/// mid-stream and self-correct the moment the part commits, because the
/// committed row renders the whole text through one `ChatMarkdownView`.
/// Text selection cannot cross chunk boundaries while the stream is live;
/// the committed row restores full selection.
struct ChunkedStreamingMarkdown: View {
    let text: String

    /// Reference box, not a `@State` struct: the accumulator records scan
    /// bookkeeping on every publish, and a `@State` write would schedule a
    /// second body pass per publish just to note the watermark. A freeze
    /// landing one render late is invisible — chunks + tail always
    /// reassemble to the same text.
    @State private var frozen = Box()

    private final class Box {
        var accumulator = ChunkedMarkdownAccumulator()
    }

    var body: some View {
        let accumulator = frozen.accumulator
        let tail = accumulator.liveTail(of: text)
        // Spacing is per-seam (paragraph rhythm vs. heading headroom), so
        // the stack carries none and each element pads its own top. Frozen
        // seams were classified at freeze time; only the tail's seam is
        // derived here, because its first line may still be streaming.
        VStack(alignment: .leading, spacing: 0) {
            ForEach(accumulator.chunks.indices, id: \.self) { index in
                ChatMarkdownView(text: accumulator.chunks[index].text)
                    .padding(.top, accumulator.chunks[index].topSpacing)
            }
            ChatMarkdownView(text: String(tail))
                .padding(
                    .top,
                    accumulator.chunks.isEmpty
                        ? 0 : ChunkedMarkdownAccumulator.seamSpacing(before: tail))
        }
        .onChange(of: text, initial: true) { _, newValue in
            frozen.accumulator.freezeCompletedBlocks(of: newValue)
        }
    }
}

// MARK: - Accumulator

/// The chunking state behind `ChunkedStreamingMarkdown`, kept separately so
/// the cut rules are unit-testable. Same contract as
/// `ChunkedTextAccumulator`: the source text only ever *grows* by appending;
/// every non-append rewrite upstream shrinks it, so a shrink is the reset
/// signal.
///
/// A cut lands at the start of a non-blank line that follows a blank line,
/// but never inside a fenced code block and never before a line indented 4+
/// columns (an indented-code or loose-list continuation). Fence tracking is
/// deliberately approximate — the leading delimiter run of a line toggles
/// fence state per CommonMark's core rules, info strings aside — because a
/// misread fence only suppresses or misplaces a seam; it cannot corrupt the
/// text, and the committed row re-renders the document whole.
nonisolated struct ChunkedMarkdownAccumulator {
    /// One frozen markdown chunk plus the seam above it, classified once at
    /// freeze time so `body` never re-derives spacing for frozen content.
    struct Chunk: Equatable {
        let text: String
        /// The vertical gap drawn above this chunk: Textual's spacing for
        /// the chunk's lead block, or 0 for the first chunk.
        let topSpacing: CGFloat
    }

    /// Frozen chunks — append-only, never mutated afterward, so a chunk view
    /// keyed by its index is stable for the life of the stream. The chunk
    /// texts joined plus `liveTail` always equal the source text.
    private(set) var chunks: [Chunk] = []
    /// UTF-8 length of the consumed source prefix. Always lands at a line
    /// start, so re-deriving the index is grapheme-safe.
    private(set) var consumedUTF8: Int = 0
    private var lastSeenUTF8: Int = 0

    // Line-scanner state, persistent across appends so a line split over
    // two deltas classifies exactly once. All byte-level: every structural
    // character (space, tab, newline, `#`, fence delimiters) is ASCII, and
    // cuts land at line starts, which are grapheme boundaries.
    private var lineStartUTF8: Int = 0
    private var leadingColumns: Int = 0
    private var contentStarted: Bool = false
    private var runByte: UInt8 = 0
    private var runLength: Int = 0
    private var runEnded: Bool = false
    private var tailHasNonSpace: Bool = false
    private var tailHasBacktick: Bool = false
    private var previousLineBlank: Bool = false
    private var inFence: Bool = false
    private var fenceByte: UInt8 = 0
    private var fenceLength: Int = 0
    /// Start offset of the newest safe cut line seen so far (0 = none).
    private var lastSafeCutUTF8: Int = 0

    /// Freeze once the live tail outgrows this many UTF-8 bytes and a safe
    /// cut exists. Smaller than `ChunkedTextAccumulator`'s threshold: the
    /// markdown tail pays parse + typeset per publish, not just layout. A
    /// single giant paragraph or fenced block never freezes mid-block; its
    /// tail grows until the block completes.
    static let freezeThresholdUTF8 = 2048

    /// Everything past the frozen prefix — what the live chunk renders.
    /// Non-mutating so `body` can call it; a text that shrank since the last
    /// freeze renders whole until `freezeCompletedBlocks` resets.
    func liveTail(of text: String) -> Substring {
        guard consumedUTF8 > 0, text.utf8.count >= consumedUTF8 else {
            return text[...]
        }
        return text[text.index(atUTF8Offset: consumedUTF8)...]
    }

    /// Fold the latest full text: reset if it shrank (rewrite upstream),
    /// scan the freshly appended bytes for new safe cuts, then freeze the
    /// prefix up to the newest cut once the tail outgrows the threshold.
    mutating func freezeCompletedBlocks(of text: String) {
        let total = text.utf8.count
        if total < lastSeenUTF8 { self = .init() }

        if lastSeenUTF8 < total {
            scan(text, fromUTF8Offset: lastSeenUTF8)
        }
        lastSeenUTF8 = total

        guard total - consumedUTF8 > Self.freezeThresholdUTF8,
            lastSafeCutUTF8 > consumedUTF8
        else { return }

        let start = text.index(atUTF8Offset: consumedUTF8)
        let cut = text.index(atUTF8Offset: lastSafeCutUTF8)
        let chunk = String(text[start..<cut])
        chunks.append(
            Chunk(
                text: chunk,
                topSpacing: chunks.isEmpty ? 0 : Self.seamSpacing(before: chunk)))
        consumedUTF8 = lastSafeCutUTF8
    }

    // MARK: Seam spacing

    /// The vertical gap a chunk seam draws above the given content, matching
    /// what Textual's block stack would put between the same two blocks in
    /// one document (`ChatMarkdownBlockSpacing`): the paragraph rhythm, or a
    /// heading / thematic break's top headroom.
    static func seamSpacing(before next: some StringProtocol) -> CGFloat {
        let firstLine = next.prefix(while: { $0 != "\n" })
        let indent = firstLine.prefix(while: { $0 == " " }).count
        guard indent < 4 else { return ChatMarkdownBlockSpacing.betweenBlocks }

        let content = firstLine.dropFirst(indent)
        if isHeading(content) || isThematicBreak(content) {
            return ChatMarkdownBlockSpacing.beforeHeading
        }
        return ChatMarkdownBlockSpacing.betweenBlocks
    }

    private static func isHeading(_ line: some StringProtocol) -> Bool {
        let hashes = line.prefix(while: { $0 == "#" })
        guard (1...6).contains(hashes.count) else { return false }
        let rest = line.dropFirst(hashes.count)
        return rest.isEmpty || rest.first == " " || rest.first == "\t"
    }

    private static func isThematicBreak(_ line: some StringProtocol) -> Bool {
        let stripped = line.filter { $0 != " " && $0 != "\t" }
        guard stripped.count >= 3, let marker = stripped.first,
            marker == "-" || marker == "*" || marker == "_"
        else { return false }
        return stripped.allSatisfy { $0 == marker }
    }

    // MARK: Line scanning

    /// Scan appended bytes, classifying each completed line (blank, fence
    /// delimiter, content) and recording a safe cut at the start of every
    /// non-blank, non-indented line that follows a blank line outside a
    /// fence. O(delta) per publish.
    private mutating func scan(_ text: String, fromUTF8Offset start: Int) {
        let utf8 = text.utf8
        var offset = start
        for byte in utf8[text.index(atUTF8Offset: start)...] {
            if byte == 0x0A {
                finishLine()
                lineStartUTF8 = offset + 1
            } else if !contentStarted {
                switch byte {
                case 0x20: leadingColumns += 1
                case 0x09: leadingColumns += 4
                case 0x0D: break
                default:
                    contentStarted = true
                    if previousLineBlank, !inFence, leadingColumns < 4 {
                        lastSafeCutUTF8 = lineStartUTF8
                    }
                    runByte = byte
                    runLength = 1
                }
            } else if !runEnded, byte == runByte {
                runLength += 1
            } else {
                runEnded = true
                if byte != 0x20, byte != 0x09, byte != 0x0D {
                    tailHasNonSpace = true
                    if byte == 0x60 { tailHasBacktick = true }
                }
            }
            offset += 1
        }
    }

    /// A `\n` just arrived; classify the completed line and reset per-line
    /// scanner state.
    private mutating func finishLine() {
        let isDelimiterRun =
            (runByte == 0x60 || runByte == 0x7E)
            && runLength >= 3 && leadingColumns <= 3
        if contentStarted, isDelimiterRun {
            if inFence {
                // A close fence: the same delimiter, at least as long, with
                // nothing but whitespace after the run.
                if runByte == fenceByte, runLength >= fenceLength, !tailHasNonSpace {
                    inFence = false
                }
            } else if runByte != 0x60 || !tailHasBacktick {
                // An open fence. A backtick fence's info string cannot
                // contain a backtick (that line is inline code, not a
                // fence).
                inFence = true
                fenceByte = runByte
                fenceLength = runLength
            }
        }
        previousLineBlank = !contentStarted
        leadingColumns = 0
        contentStarted = false
        runByte = 0
        runLength = 0
        runEnded = false
        tailHasNonSpace = false
        tailHasBacktick = false
    }
}
