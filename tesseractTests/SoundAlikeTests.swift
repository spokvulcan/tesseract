//
//  SoundAlikeTests.swift
//  tesseractTests
//
//  "Sounds like" (PRD #612): the phonetic key's folding rules, the owner's
//  real mishearings scoring as alike, clearly different words scoring as
//  different, number words, empty input, and the spelling tie-break.
//

import Testing

@testable import Tesseract_Agent

struct SoundAlikeTests {

    // MARK: - Key

    @Test func keyLowercasesKeepsTheFirstLetterAndDropsLaterVowels() {
        #expect(SoundAlike.key("Claude") == "kld")
        #expect(SoundAlike.key("cloud") == "kld")
        #expect(SoundAlike.key("apple") == "apl")
        #expect(SoundAlike.key("Echo") == "ekh")
    }

    @Test(arguments: [
        ("phone", "fn"),
        ("whale", "wl"),
        ("back", "bk"),
        ("city", "st"),
        ("cat", "kt"),
        ("quiz", "ks"),
        ("box", "bks"),
        ("zone", "sn"),
    ])
    func keyFoldsCommonSpellings(_ word: String, expected: String) {
        #expect(SoundAlike.key(word) == expected)
    }

    @Test func keyCollapsesDoubledLetters() {
        #expect(SoundAlike.key("Tesseract") == "tsrkt")
        #expect(SoundAlike.key("book") == "bk")
        #expect(SoundAlike.key("apple") == SoundAlike.key("aple"))
    }

    @Test func keyJoinsASplitWordLikeTheWholeOne() {
        #expect(SoundAlike.key("work tree") == SoundAlike.key("worktree"))
        #expect(SoundAlike.key("test flight") == SoundAlike.key("TestFlight"))
        #expect(SoundAlike.key("D-flash") == "dflsh")
    }

    @Test func keyWritesNumberWordsAsDigits() {
        #expect(SoundAlike.key("D flash two") == "dflsh2")
        #expect(SoundAlike.key("DFlash2") == "dflsh2")
        #expect(SoundAlike.key("one two") == "12")
        #expect(SoundAlike.key("version three") == SoundAlike.key("version 3"))
    }

    @Test func numberWordsOnlyFoldAsWholeWords() {
        // "someone" holds "one" but is not the number.
        #expect(SoundAlike.key("someone") == "smn")
        #expect(SoundAlike.key("twenty") == "twnt")
    }

    @Test func emptyAndPunctuationOnlyInputHasNoKeyAndNoSimilarity() {
        #expect(SoundAlike.key("") == "")
        #expect(SoundAlike.key("   ") == "")
        #expect(SoundAlike.key("...") == "")
        #expect(SoundAlike.similarity("", "Claude") == 0)
        #expect(SoundAlike.similarity("Claude", "\u{2014}") == 0)
        #expect(!SoundAlike.soundsAlike("", ""))
    }

    // MARK: - Similarity

    @Test(arguments: [
        ("Claude", "Cloud"),
        ("claude", "cloud"),
        ("Tesseract", "SRACT"),
        ("Tesseract", "SRAX"),
        ("Tesseract", "TSRAC"),
        ("worktree", "work tree"),
        ("a PR", "APR"),
        ("DFlash2", "D flash two"),
        ("KV cache", "KiwiCache"),
        ("TestFlight", "test flight"),
    ])
    func theOwnersRealMishearingsSoundAlike(_ meant: String, heard: String) {
        #expect(SoundAlike.soundsAlike(meant, heard))
        #expect(SoundAlike.soundsAlike(heard, meant))
    }

    @Test func identicalKeysScoreOne() {
        #expect(SoundAlike.similarity("Claude", "cloud") == 1)
        #expect(SoundAlike.similarity("worktree", "work tree") == 1)
        #expect(SoundAlike.similarity("DFlash2", "D flash two") == 1)
    }

    @Test(arguments: [
        ("backend", "Pi agent"),
        ("Claude", "the"),
        ("hello", "world"),
        ("cat", "dog"),
        ("Tesseract", "banana"),
    ])
    func clearlyDifferentWordsDoNotSoundAlike(_ a: String, _ b: String) {
        #expect(!SoundAlike.soundsAlike(a, b))
        #expect(SoundAlike.similarity(a, b) < SoundAlike.threshold)
    }

    @Test func similarityIsSymmetricAndBounded() {
        let pairs = [("Tesseract", "SRAX"), ("KV cache", "KiwiCache"), ("backend", "Pi agent")]
        for (a, b) in pairs {
            let score = SoundAlike.similarity(a, b)
            #expect(score == SoundAlike.similarity(b, a))
            #expect(score >= 0 && score <= 1)
        }
    }

    // MARK: - Spelling

    @Test func spellingBreaksTheTieBetweenWordsThatSoundTheSame() {
        // "Cloud" and "clod" both key as "kld", like "Claude": spelling
        // tells which is closer.
        #expect(SoundAlike.similarity("Cloud", "Claude") == SoundAlike.similarity("clod", "Claude"))
        #expect(SoundAlike.spelling("Cloud", "Claude") > SoundAlike.spelling("clod", "Claude"))
    }

    @Test func spellingIgnoresCaseSpacesAndPunctuation() {
        #expect(SoundAlike.spelling("Claude", "claude") == 1)
        #expect(SoundAlike.spelling("work tree", "worktree") == 1)
        #expect(SoundAlike.spelling("CLAUDE.md", "claudemd") == 1)
        #expect(SoundAlike.spelling("", "") == 1)
        #expect(SoundAlike.spelling("", "abc") == 0)
    }

    @Test func levenshteinCountsEdits() {
        #expect(SoundAlike.levenshtein(Array("kitten"), Array("sitting")) == 3)
        #expect(SoundAlike.levenshtein(Array(""), Array("abc")) == 3)
        #expect(SoundAlike.levenshtein(Array("abc"), Array("")) == 3)
        #expect(SoundAlike.levenshtein(Array("same"), Array("same")) == 0)
    }

    @Test func otherScriptsFoldToLatinBeforeKeying() {
        #expect(SoundAlike.soundsAlike("Клод", "Claude"))
        #expect(SoundAlike.key("café") == SoundAlike.key("cafe"))
    }
}
