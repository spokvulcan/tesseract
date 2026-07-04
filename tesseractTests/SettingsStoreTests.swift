//
//  SettingsStoreTests.swift
//  tesseractTests
//
//  Both Settings Store Adapters, tested through the `SettingsStore` interface.
//  The in-memory adapter cannot catch a bug in the production adapter, so the
//  same behavioural contract runs against both (peer-implementation conformance).
//  The `UserDefaultsSettingsStore` cases run against an injected throwaway suite
//  and tear it down with `removePersistentDomain`.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct SettingsStoreTests {

    // MARK: - Helpers

    /// Run `body` against a throwaway `UserDefaults` suite, cleaned up after.
    private func withThrowawayDefaults(_ body: (UserDefaults) -> Void) {
        let suiteName = "SettingsStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        body(defaults)
    }

    /// The behavioural contract every Settings Store Adapter must satisfy.
    private func assertStoreContract(_ store: any SettingsStore) {
        // Default-on-read for never-written keys.
        #expect(store.bool(for: "c.bool", default: true) == true)
        #expect(store.int(for: "c.int", default: 5) == 5)
        #expect(store.double(for: "c.double", default: 1.5) == 1.5)
        #expect(store.string(for: "c.string", default: "fallback") == "fallback")
        #expect(store.optionalString(for: "c.opt") == nil)

        // Round-trip each primitive.
        store.set(false, for: "c.bool")
        #expect(store.bool(for: "c.bool", default: true) == false)
        store.set(7, for: "c.int")
        #expect(store.int(for: "c.int", default: 0) == 7)
        store.set(2.5, for: "c.double")
        #expect(store.double(for: "c.double", default: 0) == 2.5)
        store.set("hi", for: "c.string")
        #expect(store.string(for: "c.string", default: "x") == "hi")

        // Optional round-trip, then nil removes the key (read-back returns the
        // default — nil here — not an empty string).
        store.setOptional("path", for: "c.opt")
        #expect(store.optionalString(for: "c.opt") == "path")
        store.setOptional(nil as String?, for: "c.opt")
        #expect(store.optionalString(for: "c.opt") == nil)

        // Optional-int round-trip, then nil removes the key.
        #expect(store.optionalInt(for: "c.optInt") == nil)
        store.setOptional(9, for: "c.optInt")
        #expect(store.optionalInt(for: "c.optInt") == 9)
        store.setOptional(nil as Int?, for: "c.optInt")
        #expect(store.optionalInt(for: "c.optInt") == nil)
    }

    // MARK: - Peer-implementation conformance

    @Test
    func inMemoryAdapterSatisfiesContract() {
        assertStoreContract(InMemorySettingsStore())
    }

    @Test
    func userDefaultsAdapterSatisfiesContract() {
        withThrowawayDefaults { defaults in
            assertStoreContract(UserDefaultsSettingsStore(defaults: defaults))
        }
    }

    // MARK: - register(defaults:)-removal regression guard

    @Test
    func userDefaultsReturnsPassedDefaultForUnsetKeys() {
        // The load-bearing risk: with `register(defaults:)` gone, an unset
        // true-default must NOT collapse to `false` via `bool(forKey:)`, nor an
        // unset numeric default to `0` via `integer`/`double(forKey:)`, nor an
        // unset string to "". The adapter must detect the missing key and return
        // the *passed* default instead.
        withThrowawayDefaults { defaults in
            let store = UserDefaultsSettingsStore(defaults: defaults)
            #expect(store.bool(for: "showInDock", default: true) == true)
            #expect(store.int(for: "serverPort", default: 8321) == 8321)
            #expect(store.double(for: "ttsTopP", default: 0.8) == 0.8)
            #expect(store.string(for: "ttsLanguage", default: "English") == "English")
        }
    }
}
