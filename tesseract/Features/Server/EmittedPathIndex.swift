//
//  EmittedPathIndex.swift
//  tesseract
//
//  The **Emitted Path Index** (ADR-0063): per model fingerprint, the SHA-256
//  of the rendered bytes from render start through a server-generated
//  end-of-turn marker maps to the **Emitted Path** — the token ids the
//  model actually saw for that history: the Cache Key Path as fed plus the
//  generated ids, ending with the canonical end-of-turn id.
//
//  Why a whole-prefix hash: a later request renders the same history to the
//  same bytes, so the index can hand back the model's own split (the one the
//  KV cache holds) instead of the canonical re-encode of identical text,
//  which byte-level BPE may split differently (`KN`+`I` for a model that fed
//  `K`+`NI`; spaced JSON the template renders compact). Only the bytes
//  after the marker are encoded canonically — a special token is a hard
//  pretoken boundary, so the concatenation is exactly what the model would
//  have been fed anyway.
//
//  What lives here is pure and model-free: the path value with its
//  terminal-token accounting, the marker derived from a probe render, the
//  running hash with a snapshot at every marker in one pass, register/resolve
//  with last-writer-wins on a key, and a byte budget over the stored ids with
//  least-recently-used eviction. Registration and the fidelity gate
//  (`EmittedPathRegistration`, `EmittedPathFidelity`) and the resolve
//  composition inside the Conversation Render (`EmittedPathResolve`) are the
//  layers above. Since ticket #476 the composition is what the request
//  feeds: the fed ids for every registered turn, the canonical encode only
//  for the bytes after the deepest hit.
//
//  Entry format (fingerprint, hash, path length, ids) is fixed so a later
//  ticket can persist entries across launches without a schema change.
//

import CryptoKit
import Foundation
import MLXLMCommon

// MARK: - Emitted Path

/// The token ids the model was fed for one stored turn: prompt ids as fed
/// (the Cache Key Path), then the generated ids, ending with the canonical
/// end-of-turn id. Decision 3 of ADR-0063 fixes the terminal-token
/// accounting: a turn that stopped on the end-of-turn id is the path as fed;
/// a foreign stop id (a template-specific extra EOS) keeps that id and
/// appends the canonical one; a token-limit cut appends the canonical one
/// after every generated id.
nonisolated struct EmittedPath: Equatable, Sendable {
    /// The full path, prompt through the end-of-turn id.
    let ids: [Int]
    /// How many leading ids are the prompt (the Cache Key Path).
    let promptCount: Int
    /// The generated ids that are content — everything the server streamed,
    /// without the stop id that ended the turn.
    let contentIDs: [Int]
    /// Whether the canonical end-of-turn id was appended (foreign stop or
    /// token-limit cut) rather than generated.
    let appendedEndOfTurn: Bool

    /// - Parameters:
    ///   - promptKeyPath: the prompt ids as fed.
    ///   - generatedTokens: every id the generation loop returned, in order,
    ///     including the stop id when one ended the turn.
    ///   - stoppedOn: the stop id the loop broke on, or `nil` for a cut.
    ///   - endOfTurnID: the template's canonical end-of-turn id.
    static func make(
        promptKeyPath: [Int],
        generatedTokens: [Int],
        stoppedOn: Int?,
        endOfTurnID: Int
    ) -> EmittedPath {
        let stoppedOnEndOfTurn = generatedTokens.last == endOfTurnID
        let stoppedOnForeignID =
            !stoppedOnEndOfTurn && stoppedOn != nil && generatedTokens.last == stoppedOn
        let content =
            (stoppedOnEndOfTurn || stoppedOnForeignID)
            ? Array(generatedTokens.dropLast()) : generatedTokens
        var ids = promptKeyPath
        ids.reserveCapacity(promptKeyPath.count + generatedTokens.count + 1)
        ids.append(contentsOf: generatedTokens)
        if !stoppedOnEndOfTurn { ids.append(endOfTurnID) }
        return EmittedPath(
            ids: ids,
            promptCount: promptKeyPath.count,
            contentIDs: content,
            appendedEndOfTurn: !stoppedOnEndOfTurn
        )
    }
}

// MARK: - End-of-turn marker

/// The template's end-of-turn marker as bytes and as its single token id —
/// the boundary every index key ends on and the token every path ends with.
/// Derived from a probe render rather than the tokenizer's EOS because the
/// two differ on some templates (Gemma's EOS is `<eos>`, its marker
/// `<end_of_turn>`).
nonisolated struct EndOfTurnMarker: Equatable, Sendable {
    let text: String
    let bytes: [UInt8]
    let tokenID: Int

    init(text: String, tokenID: Int) {
        self.text = text
        self.bytes = Array(text.utf8)
        self.tokenID = tokenID
    }

    /// Assistant content of the probe render: unlikely to occur in any
    /// template's own text, so the bytes after its last occurrence are the
    /// template's assistant-turn tail.
    static let probeContent = "EmittedPathProbe7f3a"

    /// Derives the marker from `probeRender`, a render of one user message
    /// and one assistant message whose content is `probeContent`, without a
    /// generation prompt. The tail after the content, trimmed of whitespace,
    /// must be exactly one token that encodes alone to that id — the hard
    /// boundary the standalone suffix encode relies on. Returns `nil` when
    /// the template has no such marker; the index then never registers or
    /// resolves for that model.
    static func derive(probeRender: String, tokenizer: any Tokenizer) -> EndOfTurnMarker? {
        let render = Array(probeRender.utf8)
        let needle = Array(probeContent.utf8)
        guard let contentEnd = lastOccurrenceEnd(of: needle, in: render) else { return nil }
        var tail = render[contentEnd...]
        while let first = tail.first, Self.isWhitespace(first) { tail.removeFirst() }
        while let last = tail.last, Self.isWhitespace(last) { tail.removeLast() }
        guard !tail.isEmpty, let text = String(bytes: tail, encoding: .utf8) else { return nil }
        guard let id = tokenizer.convertTokenToId(text),
            tokenizer.encode(text: text, addSpecialTokens: false) == [id]
        else { return nil }
        return EndOfTurnMarker(text: text, tokenID: id)
    }

    private static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09
    }

    /// Whether `marker` is a hard encoding boundary in `render`: at every
    /// occurrence, the encode of the bytes through it followed by the
    /// standalone encode of the rest equals the encode of the whole. The
    /// composition the resolve serves rests on exactly this equality, and
    /// a pretokenizer that treats standalone text specially (Metaspace's
    /// `prepend_scheme: first` prepends a word-boundary token to any text
    /// it is handed) breaks it — such a fingerprint is refused outright
    /// rather than served a path no render can reproduce.
    static func splitsEncoding(
        of render: [UInt8], marker: [UInt8], tokenizer: any Tokenizer
    ) -> Bool {
        let whole = tokenizer.encode(text: Self.text(render[...]), addSpecialTokens: false)
        for end in occurrenceEnds(of: marker, in: render) {
            let head = tokenizer.encode(text: Self.text(render[..<end]), addSpecialTokens: false)
            let tail = tokenizer.encode(text: Self.text(render[end...]), addSpecialTokens: false)
            guard head + tail == whole else { return false }
        }
        return true
    }

    /// The byte offsets just past every non-overlapping occurrence of
    /// `needle`, in order — the same boundaries the index hashes at.
    static func occurrenceEnds(of needle: [UInt8], in haystack: [UInt8]) -> [Int] {
        guard !needle.isEmpty, haystack.count >= needle.count else { return [] }
        var ends: [Int] = []
        var offset = 0
        let limit = haystack.count - needle.count
        while offset <= limit {
            if haystack[offset..<(offset + needle.count)].elementsEqual(needle) {
                ends.append(offset + needle.count)
                offset += needle.count
            } else {
                offset += 1
            }
        }
        return ends
    }

    private static func text(_ bytes: ArraySlice<UInt8>) -> String {
        // Render bytes are the template's own UTF-8: lossless by construction.
        // swiftlint:disable:next optional_data_string_conversion
        String(decoding: bytes, as: UTF8.self)
    }

    /// The byte offset just past the last occurrence of `needle`, or `nil`:
    /// the marker scan the derivation and the registration key share.
    static func lastOccurrenceEnd(of needle: [UInt8], in haystack: [UInt8]) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        var start = haystack.count - needle.count
        while start >= 0 {
            if haystack[start..<(start + needle.count)].elementsEqual(needle) {
                return start + needle.count
            }
            start -= 1
        }
        return nil
    }
}

/// A fingerprint's marker once derived: usable, or why the index never
/// registers or resolves for that model.
nonisolated enum EndOfTurnMarkerStatus: Equatable, Sendable {
    enum Unavailability: String, Equatable, Sendable {
        /// The template's assistant-turn tail is not one token.
        case noEndOfTurnMarker
        /// The marker is one token but not a hard encoding boundary: the
        /// standalone encode of the bytes after it differs from the
        /// in-context encode (see `EndOfTurnMarker.splitsEncoding`).
        case suffixEncodeUnstable
    }

    case available(EndOfTurnMarker)
    case unavailable(Unavailability)

    var marker: EndOfTurnMarker? {
        if case .available(let marker) = self { return marker }
        return nil
    }
}

// MARK: - Index

/// `@unchecked Sendable`: all mutable state is NSLock-guarded.
nonisolated final class EmittedPathIndex: @unchecked Sendable {

    nonisolated static let shared = EmittedPathIndex()

    /// Decision 7: 32 MB of ids in memory.
    static let defaultByteBudget = 32 << 20

    // MARK: Public surface

    struct Key: Hashable, Sendable {
        let fingerprint: String
        /// SHA-256 of the rendered bytes through the end-of-turn marker.
        let hash: [UInt8]
    }

    /// The running hash snapshotted at one end-of-turn marker.
    struct PrefixHash: Equatable, Sendable {
        let hash: [UInt8]
        /// Byte offset just past the marker: the prefix the hash covers.
        let end: Int
    }

    enum Registration: Equatable, Sendable {
        case inserted(evicted: Int)
        /// Same key seen again: last writer wins (decision 6).
        case replaced(previousLength: Int, evicted: Int)
        /// The path alone exceeds the byte budget; nothing stored.
        case rejectedTooLarge
    }

    enum MissReason: String, Equatable, Sendable {
        /// The render has no end-of-turn marker: a first request, or a
        /// template without one.
        case noMarker
        /// No entry at any marker — nothing was registered for this history.
        case noEntry
    }

    struct Resolution: Equatable, Sendable {
        /// The stored Emitted Path, ending with the end-of-turn id.
        let path: [Int]
        /// Byte offset just past the marker the path covers; the suffix to
        /// encode canonically starts here.
        let prefixEnd: Int
        /// How many markers back from the last one the hit was (0 = last).
        let markerDepth: Int
        /// Markers in the render.
        let markerCount: Int
    }

    enum Resolve: Equatable, Sendable {
        case hit(Resolution)
        case miss(MissReason)
    }

    struct Stats: Equatable, Sendable {
        var registrations = 0
        var overwrites = 0
        var evictions = 0
        var rejectedTooLarge = 0
        var fingerprintResets = 0
        var lookups = 0
        var resolves = 0
        var hits = 0
        /// `MissReason.rawValue` -> count.
        var missReasons: [String: Int] = [:]
        /// Marker depth of a hit -> count.
        var depthHistogram: [Int: Int] = [:]
        /// Registrations the fidelity check refused.
        var fidelityRejections = 0
        var entryCount = 0
        var idBytes = 0
    }

    init(byteBudget: Int = EmittedPathIndex.defaultByteBudget) {
        self.byteBudget = byteBudget
    }

    // MARK: Hashing

    /// One pass over `renderedBytes`: the running SHA-256 with a snapshot at
    /// the end of every occurrence of `marker`, in order.
    static func prefixHashes(renderedBytes: [UInt8], marker: [UInt8]) -> [PrefixHash] {
        guard !marker.isEmpty, renderedBytes.count >= marker.count else { return [] }
        var snapshots: [PrefixHash] = []
        var hasher = SHA256()
        var fed = 0
        renderedBytes.withUnsafeBytes { raw in
            let count = raw.count
            let first = marker[0]
            var offset = 0
            let limit = count - marker.count
            while offset <= limit {
                if raw[offset] == first, matches(marker, in: raw, at: offset) {
                    let end = offset + marker.count
                    hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: raw[fed..<end]))
                    fed = end
                    snapshots.append(PrefixHash(hash: Array(hasher.finalize()), end: end))
                    offset = end
                } else {
                    offset += 1
                }
            }
        }
        return snapshots
    }

    private static func matches(_ marker: [UInt8], in raw: UnsafeRawBufferPointer, at offset: Int)
        -> Bool
    {
        for (index, byte) in marker.enumerated() where raw[offset + index] != byte {
            return false
        }
        return true
    }

    /// The hash of one whole prefix, for callers that already know the
    /// boundary (registration hashes the stored render through its last
    /// marker).
    static func hash(of bytes: ArraySlice<UInt8>) -> [UInt8] {
        var hasher = SHA256()
        bytes.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        return Array(hasher.finalize())
    }

    // MARK: Register / lookup / resolve

    @discardableResult
    func register(fingerprint: String, hash: [UInt8], ids: [Int]) -> Registration {
        lock.withLock {
            adoptLocked(fingerprint: fingerprint)
            let cost = Self.cost(of: ids)
            guard cost <= byteBudget else {
                stats.rejectedTooLarge += 1
                return .rejectedTooLarge
            }
            let key = Key(fingerprint: fingerprint, hash: hash)
            let previous = entries[key]
            useCounter += 1
            entries[key] = Entry(ids: ids, lastUse: useCounter)
            idBytes += cost - (previous.map { Self.cost(of: $0.ids) } ?? 0)
            stats.registrations += 1
            let evicted = evictToBudgetLocked(keeping: key)
            if let previous {
                stats.overwrites += 1
                return .replaced(previousLength: previous.ids.count, evicted: evicted)
            }
            return .inserted(evicted: evicted)
        }
    }

    func lookup(fingerprint: String, hash: [UInt8]) -> [Int]? {
        lock.withLock {
            adoptLocked(fingerprint: fingerprint)
            stats.lookups += 1
            return findLocked(Key(fingerprint: fingerprint, hash: hash))
        }
    }

    /// Walks the markers in `renderedBytes` from the last one backwards and
    /// returns the deepest registered prefix (decision 4).
    func resolve(fingerprint: String, renderedBytes: [UInt8], marker: [UInt8]) -> Resolve {
        let hashes = Self.prefixHashes(renderedBytes: renderedBytes, marker: marker)
        return lock.withLock {
            adoptLocked(fingerprint: fingerprint)
            stats.resolves += 1
            guard !hashes.isEmpty else {
                return missLocked(.noMarker)
            }
            for (depth, snapshot) in hashes.reversed().enumerated() {
                guard let path = findLocked(Key(fingerprint: fingerprint, hash: snapshot.hash))
                else { continue }
                stats.hits += 1
                stats.depthHistogram[depth, default: 0] += 1
                return .hit(
                    Resolution(
                        path: path, prefixEnd: snapshot.end, markerDepth: depth,
                        markerCount: hashes.count))
            }
            return missLocked(.noEntry)
        }
    }

    // MARK: Marker memo

    /// The end-of-turn marker status for `fingerprint`: derived once by
    /// `derive` (the module's probe renders and the split check, run
    /// outside the lock) and remembered — an unavailable marker with its
    /// reason included — until the fingerprint changes.
    func endOfTurnMarker(
        fingerprint: String,
        derive: () -> EndOfTurnMarkerStatus
    ) -> EndOfTurnMarkerStatus {
        let memo: EndOfTurnMarkerStatus? = lock.withLock {
            adoptLocked(fingerprint: fingerprint)
            return marker
        }
        if let memo { return memo }
        let derived = derive()
        lock.withLock {
            guard activeFingerprint == fingerprint else { return }
            marker = derived
        }
        return derived
    }

    // MARK: Counters owned by the layers above

    func noteFidelityRejection() {
        lock.withLock { stats.fidelityRejections += 1 }
    }

    // MARK: Lifecycle and observability

    /// Drops every entry, the marker memo and the counters (model unload).
    func clear() {
        lock.withLock {
            entries.removeAll()
            idBytes = 0
            marker = nil
            activeFingerprint = nil
            stats = Stats()
        }
    }

    func statsSnapshot() -> Stats {
        lock.withLock { snapshotLocked() }
    }

    func logSummary(context: String) {
        let snapshot = statsSnapshot()
        Log.server.info("emitted-path-index [\(context)] \(Self.summary(of: snapshot))")
    }

    static func summary(of stats: Stats) -> String {
        let misses = stats.missReasons.sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }.joined(separator: ",")
        return
            "entries=\(stats.entryCount) idBytes=\(stats.idBytes)"
            + " registrations=\(stats.registrations) overwrites=\(stats.overwrites)"
            + " evictions=\(stats.evictions) rejectedTooLarge=\(stats.rejectedTooLarge)"
            + " resolves=\(stats.resolves) hits=\(stats.hits) misses=[\(misses)]"
            + " fidelityRejections=\(stats.fidelityRejections)"
    }

    // MARK: - Storage

    private struct Entry {
        let ids: [Int]
        var lastUse: UInt64
    }

    private let lock = NSLock()
    private let byteBudget: Int
    private var entries: [Key: Entry] = [:]
    private var idBytes = 0
    private var useCounter: UInt64 = 0
    private var activeFingerprint: String?
    /// Remembers an unusable marker with its reason too.
    private var marker: EndOfTurnMarkerStatus?
    private var stats = Stats()

    private static func cost(of ids: [Int]) -> Int {
        ids.count * MemoryLayout<Int>.size
    }

    /// Decision 7: the index is scoped to one model; a different fingerprint
    /// drops everything the previous one registered.
    private func adoptLocked(fingerprint: String) {
        guard activeFingerprint != fingerprint else { return }
        if activeFingerprint != nil {
            stats.fingerprintResets += 1
        }
        entries.removeAll()
        idBytes = 0
        marker = nil
        activeFingerprint = fingerprint
    }

    private func findLocked(_ key: Key) -> [Int]? {
        guard var entry = entries[key] else { return nil }
        useCounter += 1
        entry.lastUse = useCounter
        entries[key] = entry
        return entry.ids
    }

    private func missLocked(_ reason: MissReason) -> Resolve {
        stats.missReasons[reason.rawValue, default: 0] += 1
        return .miss(reason)
    }

    /// Evicts least-recently-used entries other than `keeping` until the
    /// budget holds; returns how many went.
    private func evictToBudgetLocked(keeping: Key) -> Int {
        guard idBytes > byteBudget else { return 0 }
        // Oldest first, ordered once: eviction runs on the post-EOS tail.
        let victims =
            entries
            .filter { $0.key != keeping }
            .sorted { $0.value.lastUse < $1.value.lastUse }
        var evicted = 0
        for victim in victims where idBytes > byteBudget {
            entries.removeValue(forKey: victim.key)
            idBytes -= Self.cost(of: victim.value.ids)
            evicted += 1
        }
        stats.evictions += evicted
        return evicted
    }

    private func snapshotLocked() -> Stats {
        var snapshot = stats
        snapshot.entryCount = entries.count
        snapshot.idBytes = idBytes
        return snapshot
    }
}
