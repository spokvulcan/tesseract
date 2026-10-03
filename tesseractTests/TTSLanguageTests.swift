//
//  TTSLanguageTests.swift
//  tesseractTests
//
//  A text's language, as the voice reads it: one of its ten.
//

import Testing

@testable import Tesseract_Agent

struct TTSLanguageTests {

    @Test(arguments: [
        (
            "The harbor lay still in the night, and the boats rocked on the water.",
            TTSLanguage.english
        ),
        ("Der Hafen lag still in der Nacht, und die Boote schaukelten auf dem Wasser.", .german),
        ("Гавань тихо лежала в ночи, и лодки покачивались на воде.", .russian),
        ("Le port était calme dans la nuit, et les bateaux se balançaient sur l'eau.", .french),
        ("El puerto estaba tranquilo en la noche y los barcos se mecían en el agua.", .spanish),
        ("港は夜の中で静かに横たわり、船は水の上で揺れていた。", .japanese),
        ("항구는 밤에 고요히 누워 있었고, 배들은 물 위에서 흔들렸다.", .korean),
        ("港口在夜色中静静地躺着，船只在水面上轻轻摇晃。", .chinese),
    ])
    func aTextIsReadInItsLanguage(text: String, language: TTSLanguage) {
        #expect(TTSLanguage.detected(in: text) == language)
    }

    @Test func nothingToJudgeByIsNoLanguage() {
        #expect(TTSLanguage.detected(in: "") == nil)
        #expect(TTSLanguage.detected(in: "  123 \n ") == nil)
    }
}
