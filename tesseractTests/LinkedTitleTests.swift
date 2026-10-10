//
//  LinkedTitleTests.swift
//  tesseractTests
//
//  A title as Today shows it: each web link in it short (its host and path,
//  without "www.", the query or a closing slash) and one click from opening;
//  the words around it stay as written.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct LinkedTitleTests {

    static func url(_ string: String) -> URL { URL(string: string)! }

    @Test func aLinkReadsAsItsHostAndPath() {
        #expect(
            LinkedTitle.runs("Research https://github.com/lithos-ai/lithos-metal and test") == [
                .text("Research "),
                .link(
                    "github.com/lithos-ai/lithos-metal",
                    Self.url("https://github.com/lithos-ai/lithos-metal")),
                .text(" and test"),
            ])
    }

    @Test func wwwTheQueryAndAClosingSlashGo() {
        #expect(
            LinkedTitle.runs("https://www.example.org/team/open-roles/?ref=mail read later")
                == [
                    .link(
                        "example.org/team/open-roles",
                        Self.url("https://www.example.org/team/open-roles/?ref=mail")),
                    .text(" read later"),
                ])
        #expect(
            LinkedTitle.runs("http://www.example.org/")
                == [.link("example.org", Self.url("http://www.example.org/"))])
    }

    @Test func punctuationAfterALinkBelongsToTheSentence() {
        #expect(
            LinkedTitle.runs("Check https://en.vedur.is/weather/forecasts/aurora, then go.") == [
                .text("Check "),
                .link(
                    "en.vedur.is/weather/forecasts/aurora",
                    Self.url("https://en.vedur.is/weather/forecasts/aurora")),
                .text(", then go."),
            ])
    }

    @Test func aLongLinkIsCutShort() throws {
        let runs = LinkedTitle.runs(
            "https://docs.google.com/document/d/1aBcD3fGhIjKlMnOpQrStUvWxYz/edit")
        guard case .link(let shown, let url) = try #require(runs.first) else {
            Issue.record("Expected a link, got \(runs)")
            return
        }
        #expect(shown == "docs.google.com/document/d/1aBcD3fGhIjK…")
        #expect(shown.count == LinkedTitle.maxLinkLength)
        // The click still opens the whole link.
        #expect(
            url == Self.url("https://docs.google.com/document/d/1aBcD3fGhIjKlMnOpQrStUvWxYz/edit"))
    }

    @Test func aTitleWithoutALinkIsAsWritten() {
        #expect(LinkedTitle.runs("Pay rent") == [.text("Pay rent")])
        #expect(
            LinkedTitle.runs("Mail anna@example.com about www") == [
                .text("Mail anna@example.com about www")
            ])
        #expect(LinkedTitle.runs("").isEmpty)
    }

    @Test func theTextShowsTheShortLinkAndOpensTheWholeOne() {
        let text = LinkedTitle.attributed("Read https://www.example.org/a/b/ now")
        #expect(String(text.characters) == "Read example.org/a/b now")
        let links = text.runs.compactMap { $0.link }
        #expect(links == [Self.url("https://www.example.org/a/b/")])
    }

    @Test func aBracketTheLinkOpenedIsTheLinks() {
        #expect(
            LinkedTitle.runs("(see https://en.wikipedia.org/wiki/Aurora_(astronomy))") == [
                .text("(see "),
                .link(
                    "en.wikipedia.org/wiki/Aurora_(astronomy)",
                    Self.url("https://en.wikipedia.org/wiki/Aurora_(astronomy)")),
                .text(")"),
            ])
    }
}
