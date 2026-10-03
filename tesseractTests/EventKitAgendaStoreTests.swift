//
//  EventKitAgendaStoreTests.swift
//  tesseractTests
//
//  The one part of the EventKit adapter a test can run without the real store
//  (ADR-0073): the completion the Reminders fetch hands EventKit, which calls
//  it on its own queue. Built inside the `@MainActor` store it inherited that
//  isolation, and every launch that fetched Reminders trapped on EventKit's
//  queue. The trap itself only fires where an Objective-C API calls the
//  closure, so this pins the shape that prevents it: a nonisolated builder
//  whose closure delivers from a background queue.
//

import EventKit
import Foundation
import Testing

@testable import Tesseract_Agent

struct EventKitAgendaStoreTests {

    @Test func theRemindersCompletionDeliversFromEventKitsQueue() async {
        let delivered = await withCheckedContinuation { continuation in
            let completion = EventKitAgendaStore.remindersCompletion { reminders in
                continuation.resume(returning: (reminders, Thread.isMainThread))
            }
            DispatchQueue.global(qos: .utility).async { completion(nil) }
        }
        #expect(delivered.0.isEmpty)
        #expect(!delivered.1)
    }
}
