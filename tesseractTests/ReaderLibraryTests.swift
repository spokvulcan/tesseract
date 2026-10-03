//
//  ReaderLibraryTests.swift
//  tesseractTests
//
//  The **Library** (#515): texts added, newest first, removed, each with
//  its own Bookmark, and all of it still there after a relaunch.
//

import Foundation
import Testing

@testable import Tesseract_Agent

@MainActor
struct ReaderLibraryTests {

    @Test func newestTextsComeFirst() {
        let library = ReaderLibrary(directory: makeTempDir("library"))
        let now = Date.now
        let older = library.add("An older article.", added: now.addingTimeInterval(-60))
        let newer = library.add("A newer article.", added: now)
        let oldest = library.add("The oldest one.", added: now.addingTimeInterval(-3_600))
        #expect(library.entries.map(\.id) == [newer.id, older.id, oldest.id])
    }

    @Test func eachTextKeepsItsOwnBookmark() {
        let library = ReaderLibrary(directory: makeTempDir("library"))
        let first = library.add("First text, read halfway.")
        let second = library.add("Second text, not started.")
        library.store(for: first.id).save(bookmark: 10, length: first.length)

        library.refreshProgress()
        #expect(library.progress[first.id] == 10 / Double(first.length))
        #expect(library.progress[second.id] == 0)
        #expect(library.store(for: second.id).load().text == "Second text, not started.")
        #expect(library.store(for: first.id).load().bookmark == 10)
    }

    @Test func removingATextForgetsItAndItsFiles() {
        let directory = makeTempDir("library")
        let library = ReaderLibrary(directory: directory)
        let kept = library.add("Kept.")
        let removed = library.add("Removed.")
        library.remove(removed.id)

        #expect(library.entries.map(\.id) == [kept.id])
        #expect(library.store(for: removed.id).load().text.isEmpty)
        #expect(ReaderLibrary(directory: directory).entries.map(\.id) == [kept.id])
    }

    @Test func theLibraryOutlivesARelaunch() {
        let directory = makeTempDir("library")
        let first = ReaderLibrary(directory: directory)
        #expect(first.isNew)
        let entry = first.add("Persisted text.\nSecond line.", title: "My article")

        let reopened = ReaderLibrary(directory: directory)
        #expect(!reopened.isNew)
        #expect(reopened.entries == [entry])
        #expect(reopened.entries.first?.title == "My article")
    }

    @Test func aTextWithoutATitleIsNamedByItsFirstLine() {
        #expect(
            ReaderLibrary.title(for: "\n\n  The Harbor at Night  \nIt was late.")
                == "The Harbor at Night")
        let long = String(repeating: "word ", count: 40)
        let title = ReaderLibrary.title(for: long)
        #expect(title.hasSuffix("…"))
        #expect(title.count <= 81)
        #expect(ReaderLibrary.title(for: "   \n  ") == "Untitled")
    }

    @Test func eachTextRemembersItsLanguage() {
        let library = ReaderLibrary(directory: makeTempDir("library"))
        let german = library.add(
            "Der Hafen lag still in der Nacht, und die Boote schaukelten leise auf dem Wasser.")
        let english = library.add(
            "The harbor lay still in the night, and the boats rocked quietly on the water.")
        #expect(german.language == "German")
        #expect(english.language == "English")
    }
}
