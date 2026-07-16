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
