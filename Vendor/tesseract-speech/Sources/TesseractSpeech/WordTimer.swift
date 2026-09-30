// TesseractSpeech — word timing from the talker's own attention (ADR-0077).
// Pure.
//
// One head of Qwen3-TTS's talker looks at the text token being spoken,
// frame by frame. The timer follows it through the segment's text with a
// path that only moves forward, decided a few frames behind the generation,
// and starts each word where the path reaches it. The audio corrects one
// habit: the model moves on to the next word as a pause begins, while the
// voice says it as the pause ends, so a start inside a pause, or just before
// one, moves to where the sound resumes. Measured against Whisper on the
// shipped model: a median 40 ms off, 96% of words within 200 ms
// (docs/research/2026-09-27-word-timing-from-attention.md).

import Foundation

struct WordTimer {
    /// The measured recipe. Changing it means measuring again.
    struct Tuning: Equatable {
        /// Frames the path is decided behind the newest one (640 ms).
        var lag = 8
        /// A pause beginning within this many frames of a word's step (and
        /// before the next word's) is the pause before the word.
        var reach = 4
        /// Silent frames that make a pause (160 ms).
        var minimumPause = 2
        /// A frame this many dB under the recent peak is silent...
        var belowPeak: Float = 25
        /// ...and so is any frame under this level.
        var floor: Float = -70
        /// How far the recent peak falls each frame: 3 dB a second.
        var peakFall: Float = 0.24
    }

    let tuning: Tuning
    /// How many words the segment has, and the word each target token is in.
    let wordCount: Int
    private let tokenWord: [Int]
    private let referenceTokens: Int

    // The path. States: 0 is before the new text (a take's text, in the
    // in-context layout), 1...J the new text's tokens, J + 1 its EOS.
    private var score: [Float] = []
    /// For frames after the first undecided one: the step (0, 1 or 2) the
    /// best path into each state took from the frame before.
    private var steps: [[UInt8]] = []
    private var stepsBase = 1
    private var frames = 0
    private var undecided = 0
    private var lastDecided = 0

    /// The frame where the path first reached each word, in word order.
    private(set) var pathStarts: [Int] = []
    /// The frame where the path reached EOS: the step after the last word.
    private var eosFrame: Int?

    // The audio, one entry per frame.
    private var silent: [Bool] = []
    /// Frames kept by the silence cap before each frame (one more entry).
    private var keptBefore: [Int] = [0]
    private var peak: Float = -120

    private var finished = false
    private var resolved = 0
    private var lastStart = 0
    private var ready: [WordStart] = []

    /// `tokenOffsets`: where each token of `text` starts, in characters, in
    /// the order the model reads them; `referenceTokens`: the take's tokens
    /// ahead of them in each attention row.
    init(text: String, tokenOffsets: [Int], referenceTokens: Int, tuning: Tuning = Tuning()) {
        self.tuning = tuning
        self.referenceTokens = referenceTokens
        (tokenWord, wordCount) = Self.tokenWords(offsets: tokenOffsets, text: text)
    }

    private var tokenCount: Int { tokenWord.count }
    private var stateCount: Int { tokenCount + 2 }

    // MARK: - Input

    /// One frame's row from the alignment head: attention logits over the
    /// take's tokens, the text's, and EOS.
    mutating func appendAttention(_ row: [Float]) {
        guard !finished else { return }
        let emissions = emission(row)
        if frames == 0 {
            // The path opens before the text or on one of its first tokens.
            score = (0 ..< stateCount).map { $0 <= 3 ? emissions[$0] : -.infinity }
        } else {
            var next = [Float](repeating: -.infinity, count: stateCount)
            var taken = [UInt8](repeating: 0, count: stateCount)
            for state in 0 ..< stateCount {
                var best = -Float.infinity
                var bestStep: UInt8 = 0
                for step in 0 ... min(2, state) where score[state - step] > best {
                    best = score[state - step]
                    bestStep = UInt8(step)
                }
                next[state] = best + emissions[state]
                taken[state] = bestStep
            }
            score = next
            steps.append(taken)
        }
        frames += 1
        // Keep scores near zero; only differences matter.
        if let top = score.max(), top.isFinite { score = score.map { $0 - top } }
        decide(through: frames - 1 - tuning.lag, endingAt: nil)
        resolve()
    }

    /// Frames of the segment's audio, in order, as the silence cap saw them.
    mutating func appendAudio(_ levels: [SilenceCap.Frame]) {
        for level in levels {
            peak = max(level.loudness, peak - tuning.peakFall)
            silent.append(level.loudness < max(peak - tuning.belowPeak, tuning.floor))
            keptBefore.append(keptBefore[keptBefore.count - 1] + (level.kept ? 1 : 0))
        }
        resolve()
    }

    /// The generation is over: decide the rest, ending on EOS. With no
    /// rows at all there is nothing to time.
    mutating func finish() {
        guard !finished else { return }
        finished = true
        guard frames > 0 else { return }
        let end = score[stateCount - 1].isFinite ? stateCount - 1 : bestState()
        decide(through: frames - 1, endingAt: end)
        if eosFrame == nil { eosFrame = frames }
        while pathStarts.count < wordCount { pathStarts.append(frames) }
        resolve()
    }

    /// Word starts found since the last call, in word order; frames counted
    /// in the audio the silence cap kept.
    mutating func takeStarts() -> [WordStart] {
        defer { ready = [] }
        return ready
    }

    // MARK: - The path

    /// Log-probabilities of each state under this row: the row's softmax,
    /// with every take token summed into "before the text".
    private func emission(_ row: [Float]) -> [Float] {
        guard let top = row.max(), top.isFinite else {
            return [Float](repeating: 0, count: stateCount)
        }
        let weights = row.map { exp($0 - top) }
        let total = weights.reduce(0, +)
        func weight(_ index: Int) -> Float { index < weights.count ? weights[index] : 0 }
        var before: Float = 0
        for index in 0 ..< min(referenceTokens, weights.count) { before += weights[index] }
        var out = [Float](repeating: 0, count: stateCount)
        out[0] = log(before / total + 1e-6)
        for token in 0 ... tokenCount {
            // Tokens of the text, then EOS.
            out[token + 1] = log(weight(referenceTokens + token) / total + 1e-6)
        }
        return out
    }

    private func bestState() -> Int {
        score.indices.max { score[$0] < score[$1] } ?? 0
    }

    /// Decides frames up to `last` from the best path through the newest
    /// frame (ending in `endingAt`, else the best state). A decided state
    /// never goes back, and later paths start from it.
    private mutating func decide(through last: Int, endingAt end: Int?) {
        guard last >= undecided else { return }
        var state = end ?? bestState()
        var path = [Int](repeating: 0, count: frames - undecided)
        path[path.count - 1] = state
        var frame = frames - 1
        while frame > undecided {
            state -= Int(steps[frame - stepsBase][state])
            frame -= 1
            path[frame - undecided] = state
        }
        for frame in undecided ... last {
            lastDecided = max(path[frame - undecided], lastDecided)
            noteDecided(frame: frame, state: lastDecided)
        }
        undecided = last + 1
        // Steps into frames up to the new first undecided one are done with.
        let drop = min(max(undecided + 1 - stepsBase, 0), steps.count)
        steps.removeFirst(drop)
        stepsBase += drop
        for state in 0 ..< min(lastDecided, stateCount) { score[state] = -.infinity }
    }

    private mutating func noteDecided(frame: Int, state: Int) {
        let word: Int
        if state == 0 {
            word = -1
        } else if state <= tokenCount {
            word = tokenWord[state - 1]
        } else {
            word = wordCount
            if eosFrame == nil { eosFrame = frame }
        }
        while pathStarts.count < wordCount, pathStarts.count <= word {
            pathStarts.append(frame)
        }
    }

    // MARK: - Starts

    private mutating func resolve() {
        while resolved < wordCount, resolved < pathStarts.count {
            let step = pathStarts[resolved]
            let next: Int
            if resolved + 1 < pathStarts.count {
                next = pathStarts[resolved + 1]
            } else if resolved + 1 == wordCount, let eosFrame {
                next = eosFrame
            } else if finished {
                next = Int.max
            } else {
                return
            }
            guard let start = soundStart(step: step, windowEnd: min(step + tuning.reach, next))
            else { return }
            lastStart = max(start, lastStart)
            ready.append(
                WordStart(
                    word: resolved, frame: keptBefore[min(lastStart, keptBefore.count - 1)]))
            resolved += 1
        }
    }

    /// Where the word whose path step is at `step` starts: at the end of a
    /// pause that holds the step or begins before `windowEnd`, else at the
    /// step. Nil until the audio settles it.
    private func soundStart(step: Int, windowEnd: Int) -> Int? {
        let known = silent.count
        if !finished, known < max(step + 1, windowEnd) { return nil }
        var frame = step
        if frame < known, silent[frame] {
            while frame > 0, silent[frame - 1] { frame -= 1 }
        }
        while frame < known {
            if silent[frame] {
                var end = frame
                while end < known, silent[end] { end += 1 }
                // The pause is still going: wait to see where it ends.
                if end == known, !finished { return nil }
                let holdsStep = frame <= step && step < end
                let beginsInWindow = step <= frame && frame < windowEnd
                if end - frame >= tuning.minimumPause, holdsStep || beginsInWindow { return end }
                frame = end
            } else {
                if frame >= windowEnd { break }
                frame += 1
            }
        }
        return min(step, known)
    }

    // MARK: - Words

    /// The word each token is in: the word holding the token's first
    /// non-space character. A token of spaces goes with the word after it;
    /// past the last word is `wordCount`. Words split by
    /// `Character.separatesWords`, as the app splits them.
    static func tokenWords(offsets: [Int], text: String) -> (words: [Int], wordCount: Int) {
        let characters = Array(text)
        var wordAt = [Int](repeating: -1, count: characters.count)
        var count = 0
        var inWord = false
        for (index, character) in characters.enumerated() {
            if character.separatesWords {
                inWord = false
                continue
            }
            if !inWord {
                count += 1
                inWord = true
            }
            wordAt[index] = count - 1
        }
        // The word at or after each position.
        var following = [Int](repeating: count, count: characters.count + 1)
        var upcoming = count
        for index in stride(from: characters.count - 1, through: 0, by: -1) {
            if wordAt[index] >= 0 { upcoming = wordAt[index] }
            following[index] = upcoming
        }
        var words: [Int] = []
        words.reserveCapacity(offsets.count)
        for (token, offset) in offsets.enumerated() {
            let start = min(max(offset, 0), characters.count)
            let end =
                token + 1 < offsets.count
                ? min(max(offsets[token + 1], start), characters.count) : characters.count
            var index = start
            while index < end, wordAt[index] < 0 { index += 1 }
            words.append(index < end ? wordAt[index] : following[min(end, characters.count)])
        }
        return (words, count)
    }
}
