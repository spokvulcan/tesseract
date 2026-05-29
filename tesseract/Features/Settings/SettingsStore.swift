//
//  SettingsStore.swift
//  tesseract
//

import Foundation

/// The seam between *what a setting means* and *where its bytes live* — a typed
/// key-value persistence port with **default-on-read** semantics: the default
/// travels with every read, so there is no `register(defaults:)` step. Sits
/// *below* the `@Observable` Settings Facade (`SettingsManager`), never above it
/// — making it the module interface would break the SwiftUI `$settings.foo`
/// binding sites and per-property Observation (see ADR-0002). Two adapters
/// satisfy it — `UserDefaultsSettingsStore` (app) and `InMemorySettingsStore`
/// (tests). The store moves bytes and never learns what a setting means.
protocol SettingsStore {
    func bool(for key: String, default: Bool) -> Bool
    func int(for key: String, default: Int) -> Int
    func double(for key: String, default: Double) -> Double
    func string(for key: String, default: String) -> String
    func optionalString(for key: String) -> String?
    func set<V>(_ value: V, for key: String)
    /// Writing `nil` removes the key (so a later read returns the default).
    func setOptional(_ value: String?, for key: String)
}

/// The single immutable declaration of one persisted setting — its key, its one
/// canonical default, and its codec to a stored primitive. The sole source of
/// truth for that setting's default, consumed by both initial load and
/// `resetToDefaults`. Construct via the type-specific factories below so the
/// codec (identity for `Bool`/`Int`/`Double`/`String`; `nil`⇒remove for
/// `String?`) is bound once.
struct Setting<Value> {
    let key: String
    let `default`: Value

    private let _load: (any SettingsStore) -> Value
    private let _write: (Value, any SettingsStore) -> Void

    /// Default-on-read: the persisted value, or `default` if the key was never written.
    func load(from store: any SettingsStore) -> Value { _load(store) }

    /// Forwarded from the Settings Facade's `didSet`.
    func write(_ value: Value, to store: any SettingsStore) { _write(value, store) }

    fileprivate init(
        key: String,
        default: Value,
        load: @escaping (any SettingsStore) -> Value,
        write: @escaping (Value, any SettingsStore) -> Void
    ) {
        self.key = key
        self.default = `default`
        self._load = load
        self._write = write
    }
}

extension Setting where Value == Bool {
    static func bool(_ key: String, default def: Bool) -> Setting<Bool> {
        Setting(key: key, default: def,
                load: { $0.bool(for: key, default: def) },
                write: { value, store in store.set(value, for: key) })
    }
}

extension Setting where Value == Int {
    static func int(_ key: String, default def: Int) -> Setting<Int> {
        Setting(key: key, default: def,
                load: { $0.int(for: key, default: def) },
                write: { value, store in store.set(value, for: key) })
    }
}

extension Setting where Value == Double {
    static func double(_ key: String, default def: Double) -> Setting<Double> {
        Setting(key: key, default: def,
                load: { $0.double(for: key, default: def) },
                write: { value, store in store.set(value, for: key) })
    }
}

extension Setting where Value == String {
    static func string(_ key: String, default def: String) -> Setting<String> {
        Setting(key: key, default: def,
                load: { $0.string(for: key, default: def) },
                write: { value, store in store.set(value, for: key) })
    }
}

extension Setting where Value == String? {
    static func optionalString(_ key: String) -> Setting<String?> {
        Setting(key: key, default: nil,
                load: { $0.optionalString(for: key) },
                write: { value, store in store.setOptional(value, for: key) })
    }
}
