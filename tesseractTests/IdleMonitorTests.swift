//
//  IdleMonitorTests.swift
//  tesseractTests
//

import AppKit
import Foundation
import Testing

@testable import Tesseract_Agent

@Suite("Idle monitor")
@MainActor
struct IdleMonitorTests {

    @Test("Wake hops onto MainActor instead of claiming notification delivery is isolated")
    func wakeHopsOntoMainActor() async {
        let workspace = NotificationCenter()
        let monitor = IdleMonitor(
            pollInterval: .seconds(60),
            returnPollInterval: .seconds(60),
            workspaceNotificationCenter: workspace,
            secondsSinceLastEvent: { IdleMonitor.idleThreshold + 1 })

        monitor.poll()
        #expect(monitor.isIdle)

        var isPostingWake = false
        var returnedWhilePostingWake = false

        await confirmation("wake delivered") { returned in
            monitor.onReturn = {
                returnedWhilePostingWake = isPostingWake
                returned()
            }
            monitor.start()

            isPostingWake = true
            workspace.post(name: NSWorkspace.didWakeNotification, object: nil)
            isPostingWake = false

            for _ in 0..<10 where monitor.isIdle {
                await Task.yield()
            }
        }

        #expect(!returnedWhilePostingWake)
        #expect(!monitor.isIdle)
        monitor.stop()
    }

    @Test("Sleep is never a return: idle stays idle, and a Mac shut mid-work is away")
    func sleepIsAnAbsence() async {
        // Away for an hour when the Mac goes to sleep: nothing returns.
        let workspace = NotificationCenter()
        let idle = IdleMonitor(
            pollInterval: .seconds(60), returnPollInterval: .seconds(60),
            workspaceNotificationCenter: workspace,
            secondsSinceLastEvent: { 3600 })
        idle.poll()
        #expect(idle.isIdle)
        var returns = 0
        idle.onReturn = { returns += 1 }
        idle.start()
        workspace.post(name: NSWorkspace.willSleepNotification, object: nil)
        for _ in 0..<10 { await Task.yield() }
        #expect(returns == 0)
        #expect(idle.isIdle)
        idle.stop()

        // At the Mac (input a few seconds ago) when the lid closes: away.
        let busy = IdleMonitor(
            pollInterval: .seconds(60), returnPollInterval: .seconds(60),
            workspaceNotificationCenter: workspace,
            secondsSinceLastEvent: { 5 })
        busy.poll()
        #expect(!busy.isIdle)
        var left = 0
        busy.onIdle = { left += 1 }
        busy.onReturn = { returns += 1 }
        busy.start()
        workspace.post(name: NSWorkspace.willSleepNotification, object: nil)
        for _ in 0..<10 where !busy.isIdle { await Task.yield() }
        #expect(left == 1)
        #expect(busy.isIdle)
        #expect(returns == 0)
        #expect(busy.awaySince.map { Date().timeIntervalSince($0) >= 5 } == true)
        busy.stop()
    }

    @Test("A queued notification cannot call back after the monitor stops")
    func queuedNotificationIsIgnoredAfterStop() async {
        let workspace = NotificationCenter()
        let monitor = IdleMonitor(
            pollInterval: .seconds(60),
            returnPollInterval: .seconds(60),
            workspaceNotificationCenter: workspace,
            secondsSinceLastEvent: { IdleMonitor.idleThreshold + 1 })

        monitor.poll()
        var returnCount = 0
        monitor.onReturn = { returnCount += 1 }
        monitor.start()

        workspace.post(name: NSWorkspace.didWakeNotification, object: nil)
        monitor.stop()

        for _ in 0..<10 {
            await Task.yield()
        }

        #expect(returnCount == 0)
        #expect(!monitor.isIdle)
    }
}
