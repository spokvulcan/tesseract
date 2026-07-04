import Foundation
import MLXLMCommon

/// The **Argument Transcoder** (CONTEXT.md → Streaming tool calls, ADR-0020):
/// converts model-native in-flight tool-call text — the Qwen `<function=…>`
/// XML dialect and the `<tool_call>` JSON wrapper body — into OpenAI
/// `delta.tool_calls[].function.arguments` **Argument Fragment**s,
/// incrementally, as `.toolCallDelta` events arrive.
///
/// A pure, synchronous value: no I/O, no transport. The streaming completion
/// path feeds it every `AgentGeneration` event and sends each returned wire
/// delta as one SSE chunk. Rules it owns:
///
/// - **Engage at name-lock**: nothing streams for a call until the function
///   name is known; the first wire delta carries index, id, type, and name
///   (arguments empty), subsequent deltas carry index + an arguments piece.
/// - **No parseable strict prefix**: clients (the AI SDK) finalize a call the
///   moment accumulated arguments parse as JSON, so the closing `}` is
///   emitted exactly once at call close; schema-typed non-string parameter
///   values are buffered and emitted whole at parameter close; string values
///   stream progressively, JSON-escaped.
/// - **Transcoder-authoritative wire**: once fragments stream, they are the
///   canonical arguments. The parser's final `.toolCall` is cross-checked
///   *semantically* and a mismatch increments `crossCheckMismatchCount` for
///   the caller to log — full arguments are never re-sent.
/// - **Wire-Valid Close**: any termination after engagement (malformation,
///   cancel, intervention, max-tokens) synthesizes closers so the accumulated
///   fragments still parse as JSON.
///
/// Formats without a transcodable dialect keep the atomic
/// name-then-full-arguments emission (`.toolCall` → two wire deltas).
nonisolated struct ArgumentTranscoder {

    /// The two model-native dialects the transcoder understands. Keyed off
    /// the model's tool-call format identity (the same inference that selects
    /// the parser), never sniffed from text.
    private enum Dialect {
        case xmlFunction
        case jsonWrapper
    }

    private static func dialect(for format: ToolCallFormat) -> Dialect? {
        switch format {
        case .xmlFunction: return .xmlFunction
        case .json: return .jsonWrapper
        default: return nil
        }
    }

    private var machine: DialectMachine?
    private let makeMachine: () -> DialectMachine?
    private let mintID: () -> String

    /// Next wire `index` for a tool call within this turn (transcoded and
    /// atomic emissions share the counter).
    private var nextIndex = 0

    /// The engaged call currently on the wire, if any.
    private var activeIndex: Int?
    /// Concatenation of the arguments fragments streamed for the active call.
    private var accumulatedArguments = ""
    /// Fragments produced before name-lock (JSON wrapper bodies may order
    /// `arguments` before `name`); flushed right after the engagement delta.
    private var preEngageFragments: [String] = []

    /// Number of calls that streamed as fragments on the wire (engaged).
    private(set) var streamedFragmentCallCount = 0
    /// Semantic disagreements between streamed fragments and the parser's
    /// final tool call. The caller logs; the wire is never retro-corrected.
    private(set) var crossCheckMismatchCount = 0

    /// True once any call streamed as fragments — the caller uses this to
    /// suppress the malformed→text fallback and to override a `.stop` finish
    /// reason to `.tool_calls` (there is no retraction on the wire).
    var hasStreamedFragments: Bool { streamedFragmentCallCount > 0 }

    init(
        format: ToolCallFormat,
        toolSpecs: [ToolSpec]?,
        mintID: @escaping () -> String = { "call_\(UUID().uuidString)" }
    ) {
        let schema = ToolSchemaTypeIndex(toolSpecs: toolSpecs)
        let dialect = Self.dialect(for: format)
        let make: () -> DialectMachine? = {
            switch dialect {
            case .xmlFunction: return .xml(XMLFunctionMachine(schema: schema))
            case .jsonWrapper: return .json(JSONWrapperMachine())
            case nil: return nil
            }
        }
        self.makeMachine = make
        self.machine = make()
        self.mintID = mintID
    }

    // MARK: - Event ingestion

    /// Fold one generation event; returns the wire tool-call deltas to send
    /// (each as its own SSE chunk), in order.
    mutating func ingest(_ event: AgentGeneration) -> [OpenAI.ToolCall] {
        switch event {
        case .toolCallDelta(_, let argumentsDelta):
            return consumeDelta(argumentsDelta)
        case .toolCall(let call):
            return closeCall(parsed: call)
        case .malformedToolCall:
            return closeCall(parsed: nil)
        default:
            return []
        }
    }

    /// Wire-Valid Close for a stream that terminates (cancel, max-tokens,
    /// intervention) while a call is engaged. Idempotent.
    mutating func finish() -> [OpenAI.ToolCall] {
        guard machine?.isEngaged == true else { return [] }
        return closeCall(parsed: nil)
    }

    // MARK: - Private

    private mutating func consumeDelta(_ text: String) -> [OpenAI.ToolCall] {
        guard machine != nil else { return [] }
        let outputs = machine!.consume(text)
        return assemble(outputs)
    }

    /// Close the active call — from the parser's `.toolCall` (cross-check),
    /// from `.malformedToolCall`, or from stream termination (`parsed: nil`).
    private mutating func closeCall(parsed call: ToolCall?) -> [OpenAI.ToolCall] {
        defer {
            machine = makeMachine()
            activeIndex = nil
            accumulatedArguments = ""
            preEngageFragments = []
        }

        guard machine?.isEngaged == true, let index = activeIndex else {
            // Nothing streamed for this call: atomic name-then-full-arguments
            // emission (non-transcodable formats, bare-JSON blocks that
            // produced no deltas, or a block whose name never locked).
            guard let call else { return [] }
            return atomicEmission(call, index: allocateIndex())
        }

        var deltas: [OpenAI.ToolCall] = []
        if accumulatedArguments.isEmpty, let call {
            // Engaged but no arguments byte streamed yet (e.g. a JSON wrapper
            // whose `arguments` was a stringified object): one whole fragment
            // from the parse is still a transcoder-authoritative concatenation.
            let whole = ToolArgumentNormalizer.encode(call.function.arguments)
            accumulatedArguments = whole
            deltas.append(fragmentDelta(whole, index: index))
        } else {
            let closers = machine!.flushClose()
            if accumulatedArguments.isEmpty && closers.isEmpty {
                // Engaged with nothing streamed and no parse available: a
                // bare `{}` keeps the accumulation valid.
                accumulatedArguments = "{}"
                deltas.append(fragmentDelta("{}", index: index))
            } else if !closers.isEmpty {
                let joined = closers.joined()
                accumulatedArguments += joined
                deltas.append(fragmentDelta(joined, index: index))
            }
            if let call {
                crossCheck(streamed: accumulatedArguments, against: call)
            }
        }
        return deltas
    }

    private mutating func assemble(_ outputs: [MachineOutput]) -> [OpenAI.ToolCall] {
        var deltas: [OpenAI.ToolCall] = []
        var pendingFragment = ""

        func flushFragment() {
            guard !pendingFragment.isEmpty else { return }
            if let index = activeIndex {
                accumulatedArguments += pendingFragment
                deltas.append(fragmentDelta(pendingFragment, index: index))
            } else {
                preEngageFragments.append(pendingFragment)
            }
            pendingFragment = ""
        }

        for output in outputs {
            switch output {
            case .fragment(let piece):
                pendingFragment += piece
            case .engage(let name):
                flushFragment()
                let index = allocateIndex()
                activeIndex = index
                streamedFragmentCallCount += 1
                deltas.append(
                    OpenAI.ToolCall(
                        id: mintID(),
                        type: "function",
                        function: OpenAI.FunctionCall(name: name, arguments: ""),
                        index: index
                    ))
                // Arguments observed before name-lock stream right after.
                pendingFragment = preEngageFragments.joined()
                preEngageFragments = []
            case .callClosedAndReopening:
                // The machine closed one block and a new `<tool_call>` block
                // begins (a parse-failure chain): detach the finished call so
                // the next engagement gets a fresh index and id.
                flushFragment()
                activeIndex = nil
                accumulatedArguments = ""
            }
        }
        flushFragment()
        return deltas
    }

    private mutating func allocateIndex() -> Int {
        defer { nextIndex += 1 }
        return nextIndex
    }

    private func fragmentDelta(_ piece: String, index: Int) -> OpenAI.ToolCall {
        OpenAI.ToolCall(
            function: OpenAI.FunctionCall(arguments: piece),
            index: index
        )
    }

    /// The atomic two-delta emission, unchanged in shape from before the
    /// transcoder existed: first delta carries id/type/name with empty
    /// arguments, second the full arguments.
    private mutating func atomicEmission(_ call: ToolCall, index: Int) -> [OpenAI.ToolCall] {
        let id = mintID()
        return [
            OpenAI.ToolCall(
                id: id,
                type: "function",
                function: OpenAI.FunctionCall(name: call.function.name, arguments: ""),
                index: index
            ),
            OpenAI.ToolCall(
                function: OpenAI.FunctionCall(
                    arguments: ToolArgumentNormalizer.encode(call.function.arguments)),
                index: index
            ),
        ]
    }

    /// Semantic cross-check of the streamed accumulation against the parser's
    /// final call. Never corrects the wire — a byte comparison is impossible
    /// (dictionary key order), so tolerant value equivalence is the contract.
    private mutating func crossCheck(streamed: String, against call: ToolCall) {
        guard let data = streamed.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            crossCheckMismatchCount += 1
            return
        }
        let streamedTree = object.mapValues { JSONValue.from($0) }
        if !Self.equivalent(.object(streamedTree), .object(call.function.arguments)) {
            crossCheckMismatchCount += 1
        }
    }

    /// Value equivalence with numeric/boolean tolerance: the parser pipelines
    /// (`JSONValue.from` over `JSONSerialization`, or `Codable`) disagree on
    /// `0`/`1` vs booleans and int vs double, so `1 ≈ true ≈ 1.0`.
    static func equivalent(_ lhs: JSONValue, _ rhs: JSONValue) -> Bool {
        func numeric(_ value: JSONValue) -> Double? {
            switch value {
            case .int(let i): return Double(i)
            case .double(let d): return d
            case .bool(let b): return b ? 1 : 0
            default: return nil
            }
        }
        switch (lhs, rhs) {
        case (.null, .null):
            return true
        case (.string(let l), .string(let r)):
            return l == r
        case (.array(let l), .array(let r)):
            return l.count == r.count && zip(l, r).allSatisfy { equivalent($0, $1) }
        case (.object(let l), .object(let r)):
            return l.keys.sorted() == r.keys.sorted()
                && l.allSatisfy { key, value in
                    r[key].map { equivalent(value, $0) } ?? false
                }
        default:
            if let l = numeric(lhs), let r = numeric(rhs) { return l == r }
            return false
        }
    }
}

// MARK: - Machine plumbing

/// What a dialect machine tells the transcoder as raw text flows through.
private nonisolated enum MachineOutput {
    /// The function name locked — mint the id, allocate the index, emit the
    /// first wire delta.
    case engage(name: String)
    /// One piece of the arguments JSON (already wire-ready).
    case fragment(String)
    /// The current block closed and a new one is beginning in the same text
    /// run (no `.toolCall` event separated them).
    case callClosedAndReopening
}

private nonisolated enum DialectMachine {
    case xml(XMLFunctionMachine)
    case json(JSONWrapperMachine)

    var isEngaged: Bool {
        switch self {
        case .xml(let machine): return machine.isEngaged
        case .json(let machine): return machine.isEngaged
        }
    }

    mutating func consume(_ text: String) -> [MachineOutput] {
        switch self {
        case .xml(var machine):
            let out = machine.consume(text)
            self = .xml(machine)
            return out
        case .json(var machine):
            let out = machine.consume(text)
            self = .json(machine)
            return out
        }
    }

    /// Wire-Valid closers for the accumulated arguments from the current
    /// state (empty when the arguments already closed naturally).
    mutating func flushClose() -> [String] {
        switch self {
        case .xml(var machine):
            let out = machine.flushClose()
            self = .xml(machine)
            return out
        case .json(var machine):
            let out = machine.flushClose()
            self = .json(machine)
            return out
        }
    }
}

// MARK: - Shared helpers

/// funcName → paramName → declared JSON-schema `type`, extracted once from
/// the request's tool specs. Mirrors the vendor parser's schema lookup
/// (`getParameterType`) so streamed typing agrees with the final parse.
nonisolated struct ToolSchemaTypeIndex: Sendable {
    /// The vendor parser's `convertParameterValue` type buckets, classified
    /// once at index build so both emission paths (string streaming vs.
    /// buffered typed values) share one reading of the schema-type string.
    enum SchemaKind: Sendable {
        case string, integer, number, boolean, container

        init(rawType: String) {
            let t = rawType.lowercased()
            if ["string", "str", "text", "varchar", "char", "enum"].contains(t) {
                self = .string
            } else if t.hasPrefix("int") || t.hasPrefix("uint") || t.hasPrefix("long")
                || t.hasPrefix("short") || t.hasPrefix("unsigned")
            {
                self = .integer
            } else if t.hasPrefix("num") || t.hasPrefix("float") {
                self = .number
            } else if ["boolean", "bool", "binary"].contains(t) {
                self = .boolean
            } else if ["object", "array"].contains(t) || t.hasPrefix("dict")
                || t.hasPrefix("list")
            {
                self = .container
            } else {
                self = .string
            }
        }
    }

    private let kinds: [String: [String: SchemaKind]]

    init(toolSpecs: [ToolSpec]?) {
        var index: [String: [String: SchemaKind]] = [:]
        for tool in toolSpecs ?? [] {
            guard let function = tool["function"] as? [String: any Sendable],
                let name = function["name"] as? String,
                let parameters = function["parameters"] as? [String: any Sendable],
                let properties = parameters["properties"] as? [String: any Sendable]
            else { continue }
            var paramKinds: [String: SchemaKind] = [:]
            for (param, spec) in properties {
                guard let spec = spec as? [String: any Sendable],
                    let type = spec["type"] as? String
                else { continue }
                paramKinds[param] = SchemaKind(rawType: type)
            }
            index[name] = paramKinds
        }
        self.kinds = index
    }

    /// `nil` when the schema declares no type — callers treat that as
    /// `.string` (the parser's own untyped default).
    func parameterKind(function: String, parameter: String) -> SchemaKind? {
        kinds[function]?[parameter]
    }
}

/// JSON-escape a piece of string *content* (no surrounding quotes). Fragments
/// escaped this way concatenate into a valid JSON string body.
private nonisolated func jsonEscaped(_ content: String) -> String {
    var out = ""
    out.reserveCapacity(content.count)
    for character in content {
        switch character {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default:
            if let scalar = character.unicodeScalars.first,
                character.unicodeScalars.count == 1,
                scalar.value < 0x20
            {
                out += String(format: "\\u%04x", scalar.value)
            } else {
                out.append(character)
            }
        }
    }
    return out
}

/// Longest suffix of `text` that is a strict prefix of any marker — the tail
/// that must be withheld because more input could complete a marker.
private nonisolated func markerCandidateSuffixStart(
    of text: String, markers: [String]
) -> String.Index {
    let maxLen = (markers.map(\.count).max() ?? 1) - 1
    var length = min(maxLen, text.count)
    while length > 0 {
        let start = text.index(text.endIndex, offsetBy: -length)
        let suffix = text[start...]
        if markers.contains(where: { $0.count > length && $0.starts(with: suffix) }) {
            return start
        }
        length -= 1
    }
    return text.endIndex
}

// MARK: - XML dialect (Qwen3.5/3.6 `<function=…>` inside `<tool_call>`)

/// Incremental mirror of the vendor `XMLFunctionParser`'s reading of a block:
/// parameter values are typed from the request's tool schema, one leading and
/// one trailing newline are trimmed per value. String-typed values stream
/// progressively; everything else buffers and emits whole at parameter close.
private nonisolated struct XMLFunctionMachine {
    private enum State {
        /// Before `<function=` (skips the `<tool_call>` open tag and padding).
        case preamble
        /// Collecting the function name up to `>`.
        case functionName
        /// Between parameters: expecting `<parameter=` or `</function>`.
        case betweenParams
        /// Collecting a parameter name up to `>`.
        case parameterName
        /// Streaming a string-typed parameter value.
        case stringValue
        /// Buffering a non-string-typed parameter value.
        case bufferedValue
        /// After `</function>` / `</tool_call>`: arguments closed on the wire.
        case closed
    }

    private static let functionStartMarker = "<function="
    private static let paramMarker = "<parameter="
    private static let paramEndMarker = "</parameter>"
    private static let functionEndMarker = "</function>"
    private static let blockEndMarker = "</tool_call>"
    private static let blockStartMarker = "<tool_call>"

    private let schema: ToolSchemaTypeIndex
    private var state: State = .preamble
    private var pending = ""
    private var functionName = ""
    private var currentKey = ""
    private var valueBuffer = ""
    private var emittedParams = 0
    /// String-value edge handling: mirror the parser's one-leading /
    /// one-trailing newline trim.
    private var atValueStart = false
    private var heldNewline = false

    private(set) var isEngaged = false

    init(schema: ToolSchemaTypeIndex) {
        self.schema = schema
    }

    // The scanning loop is one state machine on purpose; splitting it would
    // scatter the holdback discipline across functions.
    // swiftlint:disable:next function_body_length
    mutating func consume(_ text: String) -> [MachineOutput] {
        pending += text
        var out: [MachineOutput] = []

        scanning: while true {
            switch state {
            case .preamble:
                if let range = pending.range(of: Self.functionStartMarker) {
                    pending = String(pending[range.upperBound...])
                    state = .functionName
                    continue scanning
                }
                pending = String(
                    pending[
                        markerCandidateSuffixStart(
                            of: pending, markers: [Self.functionStartMarker])...])
                break scanning

            case .functionName:
                guard let range = pending.range(of: ">") else { break scanning }
                functionName = String(pending[..<range.lowerBound])
                pending = String(pending[range.upperBound...])
                isEngaged = true
                out.append(.engage(name: functionName))
                state = .betweenParams
                continue scanning

            case .betweenParams:
                let markers = [
                    Self.paramMarker, Self.functionEndMarker,
                    Self.blockEndMarker, Self.blockStartMarker,
                ]
                guard let (marker, range) = earliestMarker(in: pending, markers: markers)
                else {
                    pending = String(
                        pending[markerCandidateSuffixStart(of: pending, markers: markers)...])
                    break scanning
                }
                pending = String(pending[range.upperBound...])
                switch marker {
                case Self.paramMarker:
                    currentKey = ""
                    state = .parameterName
                case Self.blockStartMarker:
                    // A new block begins with the current one unclosed (a
                    // parse-failure chain): close this call's arguments,
                    // signal the reopen, and restart from the preamble.
                    out.append(.fragment(closeObjectFragment()))
                    out.append(.callClosedAndReopening)
                    resetForNextBlock()
                default:  // </function> or </tool_call>
                    out.append(.fragment(closeObjectFragment()))
                    state = .closed
                }
                continue scanning

            case .parameterName:
                guard let range = pending.range(of: ">") else { break scanning }
                currentKey = String(pending[..<range.lowerBound])
                pending = String(pending[range.upperBound...])
                atValueStart = true
                heldNewline = false
                if isStringKind {
                    out.append(.fragment(parameterPrefix() + "\""))
                    emittedParams += 1
                    state = .stringValue
                } else {
                    valueBuffer = ""
                    state = .bufferedValue
                }
                continue scanning

            case .stringValue:
                let terminators = [
                    Self.paramEndMarker, Self.functionEndMarker, Self.blockEndMarker,
                ]
                if let (marker, range) = earliestMarker(in: pending, markers: terminators) {
                    let content = String(pending[..<range.lowerBound])
                    if let piece = streamableStringPiece(content, terminal: true) {
                        out.append(.fragment(piece))
                    }
                    out.append(.fragment("\""))
                    pending = String(pending[range.upperBound...])
                    if marker == Self.paramEndMarker {
                        state = .betweenParams
                    } else {
                        // `</function>` / `</tool_call>` inside a value ends
                        // the block at the parser too — close everything.
                        out.append(.fragment("}"))
                        state = .closed
                    }
                    continue scanning
                }
                let holdStart = markerCandidateSuffixStart(of: pending, markers: terminators)
                let content = String(pending[..<holdStart])
                pending = String(pending[holdStart...])
                if let piece = streamableStringPiece(content, terminal: false) {
                    out.append(.fragment(piece))
                }
                break scanning

            case .bufferedValue:
                let terminators = [
                    Self.paramEndMarker, Self.functionEndMarker, Self.blockEndMarker,
                ]
                if let (marker, range) = earliestMarker(in: pending, markers: terminators) {
                    valueBuffer += pending[..<range.lowerBound]
                    pending = String(pending[range.upperBound...])
                    out.append(
                        .fragment(parameterPrefix() + typedValueJSON(from: valueBuffer)))
                    emittedParams += 1
                    valueBuffer = ""
                    if marker == Self.paramEndMarker {
                        state = .betweenParams
                    } else {
                        out.append(.fragment("}"))
                        state = .closed
                    }
                    continue scanning
                }
                let holdStart = markerCandidateSuffixStart(of: pending, markers: terminators)
                valueBuffer += pending[..<holdStart]
                pending = String(pending[holdStart...])
                break scanning

            case .closed:
                // Only a new block matters here; partial close-tag bytes the
                // delta stream over-emits are ignorable.
                if let range = pending.range(of: Self.blockStartMarker) {
                    pending = String(pending[range.upperBound...])
                    out.append(.callClosedAndReopening)
                    resetForNextBlock()
                    continue scanning
                }
                pending = String(
                    pending[
                        markerCandidateSuffixStart(
                            of: pending, markers: [Self.blockStartMarker])...])
                break scanning
            }
        }

        return out
    }

    /// Wire-Valid closers from the current state. Empty when the arguments
    /// already closed (or nothing engaged).
    mutating func flushClose() -> [String] {
        defer { state = .closed }
        guard isEngaged else { return [] }
        switch state {
        case .preamble, .functionName, .closed:
            return []
        case .betweenParams, .parameterName, .bufferedValue:
            // A partially collected key or buffered value has emitted nothing
            // yet — dropping it keeps the accumulation valid.
            return [closeObjectFragment()]
        case .stringValue:
            // The held trailing newline is treated as terminal and dropped.
            return ["\"", "}"]
        }
    }

    // MARK: XML helpers

    /// Reset per-block state for a follow-on `<tool_call>` block; `pending`
    /// (the unconsumed text of the new block) is preserved.
    private mutating func resetForNextBlock() {
        state = .preamble
        functionName = ""
        currentKey = ""
        valueBuffer = ""
        emittedParams = 0
        atValueStart = false
        heldNewline = false
        isEngaged = false
    }

    private var currentParameterKind: ToolSchemaTypeIndex.SchemaKind {
        schema.parameterKind(function: functionName, parameter: currentKey) ?? .string
    }

    private var isStringKind: Bool { currentParameterKind == .string }

    private func parameterPrefix() -> String {
        (emittedParams == 0 ? "{" : ",") + "\"" + jsonEscaped(currentKey) + "\":"
    }

    private func closeObjectFragment() -> String {
        emittedParams == 0 ? "{}" : "}"
    }

    /// Apply the value-edge discipline to a run of raw string-value content:
    /// skip one leading newline at value start, hold back one trailing
    /// newline until it is known not to be terminal, JSON-escape the rest.
    private mutating func streamableStringPiece(
        _ content: String, terminal: Bool
    ) -> String? {
        var raw = content
        if atValueStart {
            if raw.isEmpty && !terminal { return nil }
            atValueStart = false
            if raw.hasPrefix("\n") { raw = String(raw.dropFirst()) }
        }
        if heldNewline {
            raw = "\n" + raw
            heldNewline = false
        }
        if terminal {
            if raw.hasSuffix("\n") { raw = String(raw.dropLast()) }
        } else if raw.hasSuffix("\n") {
            raw = String(raw.dropLast())
            heldNewline = true
        }
        guard !raw.isEmpty else { return nil }
        return jsonEscaped(raw)
    }

    /// Mirror of the vendor `convertParameterValue` for non-string schema
    /// types, rendered as wire JSON. Object/array values pass through the
    /// model's own JSON when it parses; failed conversions degrade to a JSON
    /// string of the raw value (the parser keeps the raw string then too).
    private func typedValueJSON(from buffered: String) -> String {
        var value = buffered
        if value.hasPrefix("\n") { value = String(value.dropFirst()) }
        if value.hasSuffix("\n") { value = String(value.dropLast()) }

        func quoted() -> String { "\"" + jsonEscaped(value) + "\"" }

        switch currentParameterKind {
        case .string:
            return quoted()
        case .integer:
            if let int = Int(value) { return String(int) }
            return quoted()
        case .number:
            if let double = Double(value), double.isFinite {
                if let int = Int(exactly: double) { return String(int) }
                return "\(double)"
            }
            return quoted()
        case .boolean:
            let normalized = value.lowercased().trimmingCharacters(in: .whitespaces)
            return ["true", "1", "yes", "on"].contains(normalized) ? "true" : "false"
        case .container:
            if let data = value.data(using: .utf8),
                (try? JSONSerialization.jsonObject(with: data)) != nil
            {
                return value
            }
            return quoted()
        }
    }

    private func earliestMarker(
        in text: String, markers: [String]
    ) -> (marker: String, range: Range<String.Index>)? {
        var earliest: (marker: String, range: Range<String.Index>)?
        for marker in markers {
            guard let range = text.range(of: marker) else { continue }
            if earliest == nil || range.lowerBound < earliest!.range.lowerBound {
                earliest = (marker, range)
            }
        }
        return earliest
    }
}

// MARK: - JSON wrapper dialect (`{"name": …, "arguments": {…}}`)

/// Incremental scanner over the `<tool_call>` JSON wrapper body: locks the
/// name from the wrapper's depth-1 `"name"` value, then streams the raw bytes
/// of the `"arguments"` object verbatim (they are already wire JSON — a
/// strict prefix of a balanced object never parses early). Non-object
/// `arguments` values stream nothing; the close path emits the parsed
/// arguments whole.
private nonisolated struct JSONWrapperMachine {
    private enum Phase {
        /// Before the wrapper's `{` (skips the `<tool_call>` open tag).
        case seekingObjectStart
        /// Inside the wrapper object, tokenizing depth-1 structure.
        case scanningWrapper
        /// Inside the `arguments` object value: verbatim passthrough.
        case streamingArguments
        /// Wrapper object closed; watch for a follow-on block.
        case closed
    }

    /// Where scanning stands inside the wrapper (depth-1 keys and the value
    /// spans it skips without streaming).
    private struct WrapperCursor {
        var depth = 1
        var inString = false
        var escaped = false
        /// Non-nil while collecting a depth-1 key.
        var keyBuffer: String?
        /// The completed depth-1 key whose value comes next.
        var pendingKey: String?
        /// Non-nil while collecting the `"name"` string value.
        var nameBuffer: String?
    }

    private var phase: Phase = .seekingObjectStart
    private var cursor = WrapperCursor()
    private var argumentsTracker = JSONSpanTracker()
    private var argumentsStreamed = false
    private(set) var isEngaged = false

    mutating func consume(_ text: String) -> [MachineOutput] {
        var out: [MachineOutput] = []
        var passthrough = ""

        func flushPassthrough() {
            if !passthrough.isEmpty {
                out.append(.fragment(passthrough))
                passthrough = ""
            }
        }

        for character in text {
            switch phase {
            case .seekingObjectStart:
                if character == "{" {
                    phase = .scanningWrapper
                    cursor = WrapperCursor()
                }

            case .scanningWrapper:
                scanWrapperCharacter(character, out: &out)

            case .streamingArguments:
                let (emit, finished) = argumentsTracker.consume(character)
                if let emit { passthrough += emit }
                if finished {
                    flushPassthrough()
                    phase = .scanningWrapper
                    cursor.pendingKey = nil
                }

            case .closed:
                // A follow-on block in the same text run (no event separated
                // them): reset for the next wrapper object.
                if character == "{" {
                    out.append(.callClosedAndReopening)
                    resetForNextBlock()
                    phase = .scanningWrapper
                    cursor = WrapperCursor()
                }
            }
        }

        flushPassthrough()
        return out
    }

    /// Wire-Valid closers for the accumulated fragments. Empty when the
    /// arguments closed naturally or nothing streamed yet (the transcoder
    /// substitutes `{}` / the parsed whole in that case).
    mutating func flushClose() -> [String] {
        defer { phase = .closed }
        guard phase == .streamingArguments else { return [] }
        return argumentsTracker.synthesizeClosers()
    }

    // MARK: JSON wrapper helpers

    private mutating func resetForNextBlock() {
        cursor = WrapperCursor()
        argumentsTracker = JSONSpanTracker()
        argumentsStreamed = false
        isEngaged = false
    }

    private mutating func scanWrapperCharacter(
        _ character: Character, out: inout [MachineOutput]
    ) {
        if cursor.inString {
            if cursor.escaped {
                cursor.escaped = false
                cursor.keyBuffer?.append(character)
                cursor.nameBuffer?.append(character)
            } else if character == "\\" {
                cursor.escaped = true
                cursor.keyBuffer?.append(character)
                cursor.nameBuffer?.append(character)
            } else if character == "\"" {
                cursor.inString = false
                if let raw = cursor.keyBuffer {
                    cursor.pendingKey = Self.unescapeJSONString(raw)
                    cursor.keyBuffer = nil
                } else if let raw = cursor.nameBuffer {
                    let name = Self.unescapeJSONString(raw)
                    cursor.nameBuffer = nil
                    cursor.pendingKey = nil
                    if !isEngaged {
                        isEngaged = true
                        out.append(.engage(name: name))
                    }
                }
            } else {
                cursor.keyBuffer?.append(character)
                cursor.nameBuffer?.append(character)
            }
            return
        }

        switch character {
        case "\"":
            cursor.inString = true
            cursor.escaped = false
            if cursor.depth == 1 {
                if cursor.pendingKey == "name", !isEngaged {
                    cursor.nameBuffer = ""
                } else if cursor.pendingKey == nil {
                    cursor.keyBuffer = ""
                } else {
                    // Some other depth-1 string value: skipped via inString.
                    cursor.pendingKey = nil
                }
            }
        case "{", "[":
            if cursor.depth == 1, cursor.pendingKey == "arguments",
                character == "{", !argumentsStreamed
            {
                argumentsStreamed = true
                argumentsTracker = JSONSpanTracker()
                _ = argumentsTracker.consume(character)
                phase = .streamingArguments
                out.append(.fragment("{"))
                return
            }
            cursor.depth += 1
            cursor.pendingKey = nil
        case "}", "]":
            cursor.depth -= 1
            if cursor.depth == 0 {
                phase = .closed
            }
        case ":", ",":
            break
        default:
            // Scalar value bytes (numbers, true/false/null): the pending key
            // resolves without streaming anything.
            if cursor.depth == 1, cursor.pendingKey != nil, !character.isWhitespace {
                cursor.pendingKey = nil
            }
        }
    }

    private static func unescapeJSONString(_ raw: String) -> String {
        guard raw.contains("\\") else { return raw }
        let quoted = "\"" + raw + "\""
        guard let data = quoted.data(using: .utf8),
            let decoded = try? JSONDecoder().decode(String.self, from: data)
        else { return raw }
        return decoded
    }
}

/// Structure tracker for the verbatim-streamed `arguments` object: follows
/// scopes, string state, and escape state so termination can synthesize
/// closers that keep the accumulated fragments parseable, and so the span's
/// natural end (the matching `}`) is detected. Scalar literals are withheld
/// until their delimiter arrives so a truncation never leaves a dangling
/// `tru` or `1.` on the wire.
private nonisolated struct JSONSpanTracker {
    private enum Scope {
        case object(Position)
        case array(Position)

        enum Position {
            /// Scope just opened: a first key (object) / value (array) is
            /// expected but has not started.
            case expectingFirst
            case afterComma
            /// A key string completed; `:` not yet seen.
            case afterKey
            case afterColon
            case afterValue
        }
    }

    private var scopes: [Scope] = []
    private var inString = false
    private var escaped = false
    /// Count of hex digits still owed by a `\uXXXX` escape in flight.
    private var unicodeDigitsPending = 0
    /// True while the in-flight string is an object key.
    private var stringIsKey = false
    /// In-flight scalar literal bytes, emitted only once delimited.
    private var scalarBuffer = ""
    private var started = false

    /// Consume one character of the span (verbatim passthrough). Returns the
    /// text to emit and whether the span just closed at its matching `}`.
    mutating func consume(_ character: Character) -> (emit: String?, finished: Bool) {
        if !started {
            guard character == "{" else { return (nil, false) }
            started = true
            scopes = [.object(.expectingFirst)]
            // The opening `{` is emitted by the caller with the engagement
            // bookkeeping, not here.
            return (nil, false)
        }
        guard !scopes.isEmpty else { return (nil, true) }

        if inString {
            if unicodeDigitsPending > 0 {
                unicodeDigitsPending -= 1
            } else if escaped {
                escaped = false
                if character == "u" { unicodeDigitsPending = 4 }
            } else if character == "\\" {
                escaped = true
            } else if character == "\"" {
                inString = false
                setPosition(stringIsKey ? .afterKey : .afterValue)
            }
            return (String(character), false)
        }

        if Self.isScalarCharacter(character) {
            scalarBuffer.append(character)
            return (nil, false)
        }

        var emit = ""
        if !scalarBuffer.isEmpty {
            emit = scalarBuffer
            scalarBuffer = ""
            setPosition(.afterValue)
        }

        var finished = false
        switch character {
        case "\"":
            inString = true
            escaped = false
            stringIsKey = expectingKey
        case "{":
            scopes.append(.object(.expectingFirst))
        case "[":
            scopes.append(.array(.expectingFirst))
        case "}", "]":
            scopes.removeLast()
            if scopes.isEmpty {
                finished = true
            } else {
                setPosition(.afterValue)
            }
        case ":":
            setPosition(.afterColon)
        case ",":
            setPosition(.afterComma)
        default:
            break  // whitespace
        }
        emit.append(character)
        return (emit, finished)
    }

    /// Closers that make the accumulated span parse as JSON from the current
    /// state: complete any in-flight escape, close the string, fill any
    /// half-open slot with `null`, then close every open scope. A partial
    /// scalar was withheld, so its slot is still unfilled and gets `null`.
    /// Only the innermost scope can have a half-open slot — every enclosing
    /// scope's slot is occupied by the scope being closed inside it.
    /// The repair guarantee assumes the streamed span was valid JSON up to
    /// the truncation point (a span that was already unparseable stays so).
    func synthesizeClosers() -> [String] {
        var closers: [String] = []
        var stringClosedAsKey = false
        var stringClosedAsValue = false

        if inString {
            if unicodeDigitsPending > 0 {
                closers.append(String(repeating: "0", count: unicodeDigitsPending))
            } else if escaped {
                closers.append("n")  // completes the dangling `\` as `\n`
            }
            closers.append("\"")
            stringClosedAsKey = stringIsKey
            stringClosedAsValue = !stringIsKey
        }

        for (depth, scope) in scopes.enumerated().reversed() {
            let isInnermost = depth == scopes.count - 1
            switch scope {
            case .object(let position):
                if stringClosedAsKey {
                    closers.append(":null")
                } else if isInnermost && !stringClosedAsValue {
                    switch position {
                    case .afterKey: closers.append(":null")
                    case .afterColon: closers.append("null")
                    case .afterComma: closers.append("\"\":null")
                    case .expectingFirst, .afterValue: break
                    }
                }
                closers.append("}")
            case .array(let position):
                if isInnermost, !stringClosedAsValue, position == .afterComma {
                    closers.append("null")
                }
                closers.append("]")
            }
            stringClosedAsKey = false
            stringClosedAsValue = false
        }
        return closers.isEmpty ? [] : [closers.joined()]
    }

    // MARK: Scope position

    private static func isScalarCharacter(_ character: Character) -> Bool {
        character.isNumber || character.isLetter
            || character == "+" || character == "-" || character == "."
    }

    private var expectingKey: Bool {
        if case .object(let position) = scopes.last {
            return position == .expectingFirst || position == .afterComma
        }
        return false
    }

    private mutating func setPosition(_ position: Scope.Position) {
        guard let last = scopes.last else { return }
        switch last {
        case .object: scopes[scopes.count - 1] = .object(position)
        case .array: scopes[scopes.count - 1] = .array(position)
        }
    }
}
