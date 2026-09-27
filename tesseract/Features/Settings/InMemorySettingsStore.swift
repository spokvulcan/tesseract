//
//  InMemorySettingsStore.swift
//  tesseract
//
//  A hermetic, in-memory Settings Store Adapter for tests — a dictionary, not a
//  mock, and a *peer implementation* of `UserDefaultsSettingsStore`. Sharing no
//  global state, it runs hermetically and in parallel. A never-written key
//  returns the passed default; `setOptional(nil)` deletes the key. It lives in
//  the app, not the test target, because the test host's own container runs
//  on it too, so no test run reads or changes the owner's settings (ADR-0073).
//
//  It also records the keys written via `set`/`setOptional`, so facade tests can
//  pin the hydration≠mutation boundary: constructing from already-valid values
//  performs *zero* writes, while a stale-value migration performs *exactly* the
//  key(s) it normalizes. Call `resetWriteRecording()` to scope an assertion to a
//  single phase (e.g. construction).
//

import Foundation

@MainActor
final class InMemorySettingsStore: SettingsStore {
    private var storage: [String: Any] = [:]

    /// Keys written via `set`/`setOptional`, in order.
    private(set) var writes: [String] = []

    func resetWriteRecording() { writes = [] }

    func bool(for key: String, default def: Bool) -> Bool { storage[key] as? Bool ?? def }
    func int(for key: String, default def: Int) -> Int { storage[key] as? Int ?? def }
    func double(for key: String, default def: Double) -> Double { storage[key] as? Double ?? def }
    func string(for key: String, default def: String) -> String { storage[key] as? String ?? def }
    func optionalString(for key: String) -> String? { storage[key] as? String }
    func optionalInt(for key: String) -> Int? { storage[key] as? Int }

    func set<V>(_ value: V, for key: String) {
        storage[key] = value
        writes.append(key)
    }

    func setOptional(_ value: String?, for key: String) {
        if let value {
            storage[key] = value
        } else {
            storage.removeValue(forKey: key)
        }
        writes.append(key)
    }

    func setOptional(_ value: Int?, for key: String) {
        if let value {
            storage[key] = value
        } else {
            storage.removeValue(forKey: key)
        }
        writes.append(key)
    }

    func removeAll(withPrefix prefix: String) {
        for key in storage.keys.filter({ $0.hasPrefix(prefix) }) {
            storage.removeValue(forKey: key)
            writes.append(key)
        }
    }
}
