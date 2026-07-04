//
//  UserDefaultsSettingsStore.swift
//  tesseract
//

import Foundation

/// The production Settings Store Adapter — the only production Swift code that
/// calls `UserDefaults` (the privacy manifest still declares the API).
///
/// Owns **default-on-read**. Because there is no `register(defaults:)`, a missing
/// key must be detected with `object(forKey:) == nil` and the *passed* default
/// returned. It must NOT lean on `bool`/`integer`/`double(forKey:)`, which coerce
/// a missing key to `false`/`0` and would silently flip every unset true-default
/// (`showInDock`, `prefixCacheSSDEnabled`, `webAccessEnabled`, …) on a fresh
/// install. This is the load-bearing risk of the whole seam; `SettingsStoreTests`
/// pins it.
///
/// `init(defaults:)` accepts an injected suite so adapter tests run against a
/// throwaway `UserDefaults(suiteName:)` and clean up with `removePersistentDomain`.
struct UserDefaultsSettingsStore: SettingsStore {
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    func bool(for key: String, default def: Bool) -> Bool {
        guard defaults.object(forKey: key) != nil else { return def }
        return defaults.bool(forKey: key)
    }

    func int(for key: String, default def: Int) -> Int {
        guard defaults.object(forKey: key) != nil else { return def }
        return defaults.integer(forKey: key)
    }

    func double(for key: String, default def: Double) -> Double {
        guard defaults.object(forKey: key) != nil else { return def }
        return defaults.double(forKey: key)
    }

    func string(for key: String, default def: String) -> String {
        defaults.string(forKey: key) ?? def
    }

    func optionalString(for key: String) -> String? {
        defaults.string(forKey: key)
    }

    func optionalInt(for key: String) -> Int? {
        guard defaults.object(forKey: key) != nil else { return nil }
        return defaults.integer(forKey: key)
    }

    func set<V>(_ value: V, for key: String) {
        defaults.set(value, forKey: key)
    }

    func setOptional(_ value: String?, for key: String) {
        if let value {
            defaults.set(value, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }

    func setOptional(_ value: Int?, for key: String) {
        if let value {
            defaults.set(value, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }

    func removeAll(withPrefix prefix: String) {
        // `dictionaryRepresentation` snapshots the keys, so removing while
        // iterating the filtered copy is safe. Global-domain keys never carry
        // our app-specific prefixes, so the sweep stays scoped to our settings.
        for key in defaults.dictionaryRepresentation().keys where key.hasPrefix(prefix) {
            defaults.removeObject(forKey: key)
        }
    }
}
