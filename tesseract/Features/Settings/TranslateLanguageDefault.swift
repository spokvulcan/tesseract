//
//  TranslateLanguageDefault.swift
//  tesseract
//

import Foundation

/// Derives the default target language for the `translate` skill (PRD #174):
/// the English display name of the first non-English macOS preferred language,
/// or "English" when every preferred language is English. Pure over an
/// injected language list so tests need no machine state; the catalogue default
/// feeds it `Locale.preferredLanguages` once per launch.
nonisolated enum TranslateLanguageDefault {

    /// English display name of the first non-English language identifier
    /// (e.g. `["uk-UA", "en-US"]` → "Ukrainian"), or "English".
    ///
    /// Names come from the ``SupportedLanguage`` catalogue first so the derived
    /// default matches the Settings picker's options exactly; `Locale` is the
    /// fallback for codes outside the catalogue.
    static func derive(from preferredLanguages: [String]) -> String {
        let english = Locale(identifier: "en_US")
        for identifier in preferredLanguages {
            let language = Locale(identifier: identifier).language
            guard let code = language.languageCode?.identifier, code != "en" else { continue }
            if let catalogued = SupportedLanguage.language(forCode: code) {
                return catalogued.name
            }
            if let name = english.localizedString(forLanguageCode: code) {
                return name
            }
        }
        return "English"
    }

    /// The launch-time system derivation.
    static func systemDefault() -> String {
        derive(from: Locale.preferredLanguages)
    }
}
