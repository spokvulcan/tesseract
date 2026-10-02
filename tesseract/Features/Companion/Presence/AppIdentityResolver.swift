//
//  AppIdentityResolver.swift
//  tesseract
//
//  Who an app is, from what macOS knows about it: its bundle id, where it is
//  installed, and what its Info.plist declares. A banner names its app only
//  by display name, so the name is matched against the running apps. Read
//  once per app and cached.
//

import AppKit
import Foundation

@MainActor
final class AppIdentityResolver {

    private var byName: [String: AppIdentity] = [:]
    private var byPath: [String: AppIdentity] = [:]

    /// The identity of a running app.
    func identity(of app: NSRunningApplication) -> AppIdentity {
        let name = app.localizedName ?? app.bundleIdentifier ?? ""
        guard let url = app.bundleURL else {
            return AppIdentity(name: name, bundleID: app.bundleIdentifier)
        }
        if let cached = byPath[url.path] { return cached }
        let info = Bundle(url: url)?.infoDictionary ?? [:]
        let identity = AppIdentity(
            name: name, bundleID: app.bundleIdentifier, bundlePath: url.path,
            category: info["LSApplicationCategoryType"] as? String,
            supportsGameMode: info["GCSupportsGameMode"] as? Bool ?? false)
        byPath[url.path] = identity
        return identity
    }

    /// The identity behind a banner's app name, when that app is running.
    func identity(named name: String) -> AppIdentity? {
        let key = name.lowercased()
        if let cached = byName[key] { return cached }
        guard
            let app = NSWorkspace.shared.runningApplications.first(where: {
                $0.localizedName?.lowercased() == key
            })
        else { return nil }
        let identity = identity(of: app)
        byName[key] = identity
        return identity
    }
}
