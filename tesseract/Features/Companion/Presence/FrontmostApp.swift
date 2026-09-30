//
//  FrontmostApp.swift
//  tesseract
//
//  Which app is in front, and since when a terminal was: the facts the
//  seen ledger, "where you were" and the coding-agent rule read. Observed
//  from NSWorkspace, nothing more.
//

import AppKit
import IOKit.ps
import Foundation

@MainActor
final class FrontmostAppTracker {

    private(set) var name: String?
    private(set) var bundleID: String?
    /// When a terminal was last in front (kept current while one is).
    private var terminalLeftAt: Date?
    private var observer: NSObjectProtocol?

    /// Called on every activation with the app's display name and bundle id.
    var onActivate: ((String, String?) -> Void)?

    func start() {
        guard observer == nil else { return }
        record(NSWorkspace.shared.frontmostApplication)
        observer = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] note in
            let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            let name = app?.localizedName
            let bundleID = app?.bundleIdentifier
            MainActor.assumeIsolated {
                self?.activated(name: name, bundleID: bundleID)
            }
        }
    }

    /// When a terminal was last in front: now, while one is.
    func lastTerminalFrontAt(now: Date = Date()) -> Date? {
        TerminalApps.contains(bundleID) ? now : terminalLeftAt
    }

    private func record(_ app: NSRunningApplication?) {
        name = app?.localizedName
        bundleID = app?.bundleIdentifier
    }

    private func activated(name: String?, bundleID: String?) {
        if TerminalApps.contains(self.bundleID), !TerminalApps.contains(bundleID) {
            terminalLeftAt = Date()
        }
        self.name = name
        self.bundleID = bundleID
        if let name { onActivate?(name, bundleID) }
    }
}

/// Power source and thermal state, for the governor.
@MainActor
final class PowerMonitor {

    var onChange: (() -> Void)?
    private var observers: [NSObjectProtocol] = []
    private var pollTask: Task<Void, Never>?

    var state: PowerState {
        let thermal: PowerState.Thermal =
            switch ProcessInfo.processInfo.thermalState {
            case .nominal: .nominal
            case .fair: .fair
            case .serious: .serious
            case .critical: .critical
            @unknown default: .nominal
            }
        let (onAC, battery) = PowerSource.current()
        return PowerState(onACPower: onAC, batteryPercent: battery, thermal: thermal)
    }

    func start() {
        guard observers.isEmpty else { return }
        observers.append(
            NotificationCenter.default.addObserver(
                forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.onChange?() }
            })
        observers.append(
            NotificationCenter.default.addObserver(
                forName: .NSProcessInfoPowerStateDidChange, object: nil, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.onChange?() }
            })
        // Plugging in has no notification here; the clock tick re-reads it,
        // and a slow poll catches the change between ticks.
        var last = state.onACPower
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(120))
                guard let self else { return }
                let now = self.state.onACPower
                if now != last {
                    last = now
                    self.onChange?()
                }
            }
        }
    }
}

/// The power source, read through IOKit.
nonisolated enum PowerSource {
    /// On AC power (a desktop reads AC), and the battery's charge when there
    /// is a battery.
    static func current() -> (onAC: Bool, batteryPercent: Int?) {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue() else { return (true, nil) }
        let providing = IOPSGetProvidingPowerSourceType(info)?.takeRetainedValue() as String?
        let onAC = providing != kIOPSBatteryPowerValue
        guard let list = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef] else {
            return (onAC, nil)
        }
        for source in list {
            guard
                let description = IOPSGetPowerSourceDescription(info, source)?.takeUnretainedValue()
                    as? [String: Any],
                let current = description[kIOPSCurrentCapacityKey] as? Int,
                let maximum = description[kIOPSMaxCapacityKey] as? Int, maximum > 0
            else { continue }
            return (onAC, current * 100 / maximum)
        }
        return (onAC, nil)
    }
}
