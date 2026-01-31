//
//  whisper_on_deviceTests.swift
//  whisper-on-deviceTests
//
//  Created by Bohdan Ivanchenko on 31.01.2026.
//

import Testing
@testable import whisper_on_device

@MainActor
struct TranscriptionPostProcessorTests {

    @Test func processTrimsWhitespace() async throws {
        let processor = TranscriptionPostProcessor()
        let result = processor.process("  Hello World  ")
        #expect(result == "Hello World")
    }

    @Test func processRemovesDuplicateSpaces() async throws {
        let processor = TranscriptionPostProcessor()
        let result = processor.process("Hello    World")
        #expect(result == "Hello World")
    }

    @Test func processCapitalizesFirstLetter() async throws {
        let processor = TranscriptionPostProcessor()
        let result = processor.process("hello world")
        #expect(result == "Hello world")
    }

    @Test func processCapitalizesAfterPunctuation() async throws {
        let processor = TranscriptionPostProcessor()
        let result = processor.process("Hello. world")
        #expect(result == "Hello. World")
    }

    @Test func processCapitalizesStandaloneI() async throws {
        let processor = TranscriptionPostProcessor()
        let result = processor.process("i think i can")
        #expect(result == "I think I can")
    }

    @Test func processRemovesRepeatedWords() async throws {
        let processor = TranscriptionPostProcessor()
        let result = processor.process("the the cat")
        #expect(result == "The cat")
    }

    @Test func processHandlesEmptyString() async throws {
        let processor = TranscriptionPostProcessor()
        let result = processor.process("")
        #expect(result == "")
    }
}

@MainActor
struct AudioConverterTests {

    @Test func normalizeHandlesEmptyArray() async throws {
        let result = AudioConverter.normalize([])
        #expect(result.isEmpty)
    }

    @Test func calculateRMSHandlesEmptyArray() async throws {
        let result = AudioConverter.calculateRMS([])
        #expect(result == 0)
    }

    @Test func hasClippingDetectsClipping() async throws {
        let samples: [Float] = [0.5, 0.99, 0.3]
        #expect(AudioConverter.hasClipping(samples))
    }

    @Test func hasClippingReturnsFalseForNormalAudio() async throws {
        let samples: [Float] = [0.5, 0.3, 0.2]
        #expect(!AudioConverter.hasClipping(samples))
    }
}

@MainActor
struct KeyComboTests {

    @Test func f5PresetIsCorrect() async throws {
        let combo = KeyCombo.f5
        #expect(combo.keyCode == 96)
        #expect(combo.modifiers == 0)
    }

    @Test func displayStringShowsCorrectFormat() async throws {
        let combo = KeyCombo.f5
        #expect(combo.displayString == "F5")
    }
}

@MainActor
struct DictationStateTests {

    @Test func idleStateIsNotActive() async throws {
        let state = DictationState.idle
        #expect(!state.isActive)
    }

    @Test func recordingStateIsActive() async throws {
        let state = DictationState.recording
        #expect(state.isActive)
    }

    @Test func processingStateIsActive() async throws {
        let state = DictationState.processing
        #expect(state.isActive)
    }

    @Test func errorStateIsNotActive() async throws {
        let state = DictationState.error("Test error")
        #expect(!state.isActive)
    }
}

@MainActor
struct WhisperModelTests {

    @Test func allCasesExist() async throws {
        #expect(WhisperModel.allCases.count == 4)
    }

    @Test func baseSizeIsCorrect() async throws {
        #expect(WhisperModel.base.sizeGB == 0.145)
    }

    @Test func displayNameIsCorrect() async throws {
        #expect(WhisperModel.tiny.displayName == "Tiny")
        #expect(WhisperModel.base.displayName == "Base")
        #expect(WhisperModel.small.displayName == "Small")
        #expect(WhisperModel.medium.displayName == "Medium")
    }
}
