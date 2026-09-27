//
//  VoiceDesignTests.swift
//  tesseractTests
//
//  Designing a voice (pure `VoiceDesign`) and the voices the Speech page
//  offers (`VoiceLibrary` over an in-memory settings store and a scratch
//  Pinned Voice store).
//

import Foundation
import Testing
import TesseractSpeech

@testable import Tesseract_Agent

struct VoiceDesignTests {

    private func pick(_ labels: [String: String], in design: inout VoiceDesign) {
        for trait in design.traits {
            guard let label = labels[trait.id],
                let option = trait.options.first(where: { $0.label == label })
            else { continue }
            design.toggle(option, for: trait)
        }
    }

    @Test func traitsWriteQwensCompactShapeWhoFirst() {
        var design = VoiceDesign(language: "English")
        pick(
            [
                "gender": "Female", "age": "Middle-aged", "pitch": "Low", "texture": "Husky",
                "mood": "Calm", "pace": "Slow", "accent": "American",
            ], in: &design)

        #expect(
            design.description
                == "Female, middle-aged, low pitch. A slightly husky voice, calm and steady, speaking fluently at a slow, unhurried pace with clear articulation. General American accent."
        )
        #expect(design.suggestedName == "Husky middle-aged woman")
    }

    @Test func aFewTraitsStillReadAsSentences() {
        var design = VoiceDesign()
        pick(["gender": "Male", "texture": "Bright"], in: &design)
        #expect(
            design.description
                == "Male. A bright, clear voice speaking fluently with clear articulation.")
    }

    @Test func aCheerfulMoodRulesOutLaughter() {
        var design = VoiceDesign()
        pick(["mood": "Cheerful"], in: &design)
        #expect(design.description.contains("without laughing"))
    }

    @Test func pickingAChosenOptionAgainClearsIt() {
        var design = VoiceDesign()
        pick(["gender": "Female"], in: &design)
        pick(["gender": "Female"], in: &design)
        #expect(design.isEmpty)
    }

    @Test func writingByHandTakesOverFromTheTraitsAndBack() {
        var design = VoiceDesign()
        pick(["gender": "Male"], in: &design)
        design.setDescription("A gravelly old sea captain.")
        #expect(design.description == "A gravelly old sea captain.")
        #expect(design.traits.allSatisfy { design.choice(for: $0) == nil })

        pick(["gender": "Female"], in: &design)
        #expect(design.description.hasPrefix("Female."))
    }

    @Test func theReferenceLineFollowsTheLanguageUnlessEdited() {
        var design = VoiceDesign(language: "English")
        design.setLanguage("German")
        #expect(design.referenceLine == VoiceDesign.referenceLine(for: "German"))

        design.referenceLine = "Mein eigener Satz."
        design.setLanguage("French")
        #expect(design.referenceLine == "Mein eigener Satz.")
    }

    @Test func accentIsOfferedOnlyInEnglishAndDroppedOutsideIt() {
        var design = VoiceDesign(language: "English")
        #expect(design.traits.contains { $0.id == "accent" })
        pick(["accent": "British"], in: &design)
        #expect(design.description.contains("British accent"))

        design.setLanguage("Japanese")
        #expect(!design.traits.contains { $0.id == "accent" })
        #expect(!design.description.contains("accent"))
    }

    @Test func shuffleAlwaysNamesWhoIsSpeakingAndTheTimbre() throws {
        for _ in 0..<20 {
            var design = VoiceDesign(language: "English")
            design.shuffle()
            for id in ["gender", "age", "pitch", "texture"] {
                let trait = try #require(design.traits.first { $0.id == id })
                #expect(design.choice(for: trait) != nil, "\(id) is always picked")
            }
        }
    }

    @MainActor
    @Test func everyLanguageHasTwoSentencesToBeBornReading() {
        for language in TTSLanguage.allCases {
            let line = VoiceDesign.referenceLine(for: language.rawValue)
            #expect(!line.isEmpty)
            let enders = line.filter { ".!?。！？".contains($0) }
            #expect(enders.count == 2, "\(language.rawValue): \(line)")
        }
    }

    @Test func shortNamesForDesignedAndWrittenDescriptions() {
        #expect(
            VoiceDesign.shortName(
                for: "Female, middle-aged, low pitch. A slightly husky voice, calm and steady.")
                == "Female, middle-aged, low pitch")
        #expect(
            VoiceDesign.shortName(for: "A warm, raspy older female voice with a British accent.")
                == "Warm, raspy older")
        #expect(VoiceDesign.shortName(for: "") == "Custom voice")
    }

    @Test func theSummaryTellsNearIdenticalDesignsApart() {
        let american =
            "A calm, breathy older female voice with an American accent and a slow, unhurried pace."
        let british =
            "A calm, breathy older female voice with a British accent and a slow, unhurried pace."
        #expect(VoiceDesign.summary(of: american) == "Female · American accent · slow")
        #expect(VoiceDesign.summary(of: british) == "Female · British accent · slow")
        #expect(VoiceDesign.summary(of: "Something plain.").isEmpty)
    }
}

@MainActor
struct VoiceLibraryTests {

    private let model = ModelDefinition.textToSpeechModelSpec

    private func makeLibrary(
        source: TTSVoiceSource = .designed
    ) -> (VoiceLibrary, SettingsManager, PinnedVoiceStore) {
        let settings = SettingsManager(store: InMemorySettingsStore())
        let pinned = PinnedVoiceStore(directory: makeTempDir("voice-library"))
        return (
            VoiceLibrary(settings: settings, pinnedVoices: pinned, source: source), settings, pinned
        )
    }

    private func pin(_ description: String, in store: PinnedVoiceStore) {
        store.save(
            PinnedVoice(
                modelFingerprint: model.fingerprint, voiceDescription: description,
                language: "English",
                referenceText: "Hello there.", codeFrames: [[1, 2, 3], [4, 5, 6]]))
    }

    @Test func yourVoicesAreSavedOnesThenDesignsNeverNamed() {
        let (library, _, pinned) = makeLibrary()
        let builtIn = VoiceLibrary.designedBuiltIns[0].description
        pin(builtIn, in: pinned)
        pin("A husky older male voice with a slow pace.", in: pinned)
        pin("A crisp young female voice.", in: pinned)
        library.save(name: "Captain", description: "A crisp young female voice.")
        library.refresh()

        #expect(library.yourVoices.map(\.name) == ["Captain", "Husky older male"])
        #expect(library.yourVoices.map(\.kind) == [.saved, .unnamed])
        #expect(
            library.yourVoices[1].detail == "Male · slow", "a built-in is never listed as yours")
    }

    @Test func savingADescriptionAgainReplacesItsEarlierSave() {
        let (library, settings, _) = makeLibrary()
        library.save(name: "First", description: "A calm voice.")
        library.save(name: "Second", description: " A calm voice. ")
        #expect(settings.savedVoices.map(\.name) == ["Second"])
    }

    @Test func renamingAnUnnamedDesignSavesIt() {
        let (library, settings, pinned) = makeLibrary()
        pin("A husky older male voice.", in: pinned)
        library.refresh()
        let voice = library.yourVoices[0]
        library.rename(voice, to: "Grandpa")
        #expect(settings.savedVoices.map(\.name) == ["Grandpa"])
        #expect(library.name(for: voice.description) == "Grandpa")
    }

    @Test func deletingTheVoiceInUseFallsBackToABuiltIn() {
        let (library, settings, pinned) = makeLibrary()
        let description = "A husky older male voice."
        pin(description, in: pinned)
        library.save(name: "Grandpa", description: description)
        library.refresh()
        let voice = library.yourVoices[0]
        library.select(voice)

        library.delete(voice)

        #expect(library.yourVoices.isEmpty)
        #expect(pinned.voice(description: description, language: "English", model: model) == nil)
        #expect(settings.ttsVoiceDescription == VoiceLibrary.designedBuiltIns[0].description)
    }

    /// A checkpoint that can't design voices offers its own speakers only.
    @Test func aPresetCheckpointListsItsSpeakersAndNoDesigns() {
        let (library, settings, pinned) = makeLibrary(source: .presets(["vivian", "ryan"]))
        pin("A husky older male voice.", in: pinned)
        library.refresh()
        library.save(name: "Nope", description: "A calm voice.")

        #expect(!library.source.supportsVoiceDesign)
        #expect(library.yourVoices.isEmpty)
        #expect(library.builtIn.map(\.name) == ["Vivian", "Ryan"])
        #expect(settings.savedVoices.isEmpty)
    }
}
