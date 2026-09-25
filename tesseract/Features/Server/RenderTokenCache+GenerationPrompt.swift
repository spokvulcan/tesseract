//
//  RenderTokenCache+GenerationPrompt.swift
//  tesseract
//
//  The **Render+Token Cache**'s memo of **Generation Prompt** probes
//  (ADR-0070): the two one-message renders a render context costs, paid once
//  per model and render context and remembered until the cache is reset at
//  model unload. Keyed on the model fingerprint when it is known, and on the
//  tokenizer instance for the life of the load when it is not, so a request
//  never pays the probe once its context has been seen.
//

import Foundation
import MLXLMCommon

/// `@unchecked Sendable`: all mutable state is NSLock-guarded.
nonisolated final class GenerationPromptProbeMemo: @unchecked Sendable {

    /// Whose probe an entry is: the model fingerprint's, or, when the
    /// fingerprint is unknown, one tokenizer instance's. A value-type
    /// tokenizer has no instance to key on and is measured on every ask;
    /// production always knows the fingerprint. The tokenizer's type rides
    /// along with a fingerprint, so two kinds of tokenizer handed one
    /// fingerprint (test doubles) never share a probe.
    private enum Owner: Hashable {
        case fingerprint(String, tokenizerType: ObjectIdentifier)
        case tokenizer(ObjectIdentifier)
    }

    private struct Key: Hashable {
        let owner: Owner
        let contextDigest: String
    }

    private struct Entry {
        let probe: GenerationPrompt.Probe
        /// The tokenizer an instance-owned entry belongs to, held weakly: a
        /// freed tokenizer's identifier can be reused by the next one, which
        /// must not inherit this entry.
        weak var tokenizer: AnyObject?
        /// The unknown reasons already warned about under this key.
        var warned: Set<GenerationPrompt.Unknown> = []
    }

    private let lock = NSLock()
    private var entries: [Key: Entry] = [:]

    /// The probe for a model and render context: `measure` runs once,
    /// outside the lock, and its result is remembered until `clear()`.
    func probe(
        modelFingerprint: String?,
        tokenizer: any Tokenizer,
        contextDigest: String,
        measure: () -> GenerationPrompt.Probe
    ) -> GenerationPrompt.Probe {
        guard
            let (key, instance) = Self.key(
                modelFingerprint: modelFingerprint, tokenizer: tokenizer,
                contextDigest: contextDigest)
        else { return measure() }
        if let memo = lock.withLock({ liveEntryLocked(key, instance: instance)?.probe }) {
            return memo
        }
        let probe = measure()
        lock.withLock {
            if liveEntryLocked(key, instance: instance) == nil {
                entries[key] = Entry(probe: probe, tokenizer: instance)
            }
        }
        return probe
    }

    /// Whether an unknown Generation Prompt under this model and render
    /// context is being reported for `reason` the first time, so its
    /// warning lands once rather than on every request.
    func firstUnknown(
        modelFingerprint: String?,
        tokenizer: any Tokenizer,
        contextDigest: String,
        reason: GenerationPrompt.Unknown
    ) -> Bool {
        guard
            let (key, instance) = Self.key(
                modelFingerprint: modelFingerprint, tokenizer: tokenizer,
                contextDigest: contextDigest)
        else { return true }
        return lock.withLock {
            guard var entry = liveEntryLocked(key, instance: instance) else { return true }
            let first = entry.warned.insert(reason).inserted
            entries[key] = entry
            return first
        }
    }

    /// Drop every probe (model unload).
    func clear() {
        lock.withLock { entries.removeAll() }
    }

    private static func key(
        modelFingerprint: String?, tokenizer: any Tokenizer, contextDigest: String
    ) -> (key: Key, instance: AnyObject?)? {
        if let modelFingerprint {
            let owner = Owner.fingerprint(
                modelFingerprint, tokenizerType: ObjectIdentifier(type(of: tokenizer)))
            return (Key(owner: owner, contextDigest: contextDigest), nil)
        }
        guard type(of: tokenizer) is AnyClass else { return nil }
        let instance = tokenizer as AnyObject
        return (
            Key(owner: .tokenizer(ObjectIdentifier(instance)), contextDigest: contextDigest),
            instance
        )
    }

    /// The entry under `key`, dropped when it belonged to a tokenizer
    /// instance other than `instance`.
    private func liveEntryLocked(_ key: Key, instance: AnyObject?) -> Entry? {
        guard let entry = entries[key] else { return nil }
        if let instance, entry.tokenizer !== instance {
            entries[key] = nil
            return nil
        }
        return entry
    }
}
