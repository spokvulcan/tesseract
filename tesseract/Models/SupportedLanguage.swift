//
//  SupportedLanguage.swift
//  tesseract
//

import Foundation

/// Represents a language supported by the Whisper model for transcription.
struct SupportedLanguage: Identifiable, Hashable {
    let code: String       // ISO 639-1 code (e.g., "en")
    let name: String       // Display name (e.g., "English")
    let flag: String       // Emoji flag (e.g., "🇬🇧")
    let nativeName: String? // Optional native name

    var id: String { code }

    var displayName: String {
        "\(flag) \(name)"
    }

    // MARK: - Special Values

    static let auto = SupportedLanguage(
        code: "auto",
        name: "Auto-detect",
        flag: "🌐",
        nativeName: nil
    )

    // MARK: - All Supported Languages (99 languages from Whisper Large V3 Turbo)

    static let all: [SupportedLanguage] = [
        auto,
        SupportedLanguage(code: "af", name: "Afrikaans", flag: "🇿🇦", nativeName: "Afrikaans"),
        SupportedLanguage(code: "am", name: "Amharic", flag: "🇪🇹", nativeName: "አማርኛ"),
        SupportedLanguage(code: "ar", name: "Arabic", flag: "🇸🇦", nativeName: "العربية"),
        SupportedLanguage(code: "as", name: "Assamese", flag: "🇮🇳", nativeName: "অসমীয়া"),
        SupportedLanguage(code: "az", name: "Azerbaijani", flag: "🇦🇿", nativeName: "Azərbaycan"),
        SupportedLanguage(code: "ba", name: "Bashkir", flag: "🇷🇺", nativeName: "Башҡорт"),
        SupportedLanguage(code: "be", name: "Belarusian", flag: "🇧🇾", nativeName: "Беларуская"),
        SupportedLanguage(code: "bg", name: "Bulgarian", flag: "🇧🇬", nativeName: "Български"),
        SupportedLanguage(code: "bn", name: "Bengali", flag: "🇧🇩", nativeName: "বাংলা"),
        SupportedLanguage(code: "bo", name: "Tibetan", flag: "🏔️", nativeName: "བོད་སྐད"),
        SupportedLanguage(code: "br", name: "Breton", flag: "🇫🇷", nativeName: "Brezhoneg"),
        SupportedLanguage(code: "bs", name: "Bosnian", flag: "🇧🇦", nativeName: "Bosanski"),
        SupportedLanguage(code: "ca", name: "Catalan", flag: "🇪🇸", nativeName: "Català"),
        SupportedLanguage(code: "cs", name: "Czech", flag: "🇨🇿", nativeName: "Čeština"),
        SupportedLanguage(code: "cy", name: "Welsh", flag: "🏴󠁧󠁢󠁷󠁬󠁳󠁿", nativeName: "Cymraeg"),
        SupportedLanguage(code: "da", name: "Danish", flag: "🇩🇰", nativeName: "Dansk"),
        SupportedLanguage(code: "de", name: "German", flag: "🇩🇪", nativeName: "Deutsch"),
        SupportedLanguage(code: "el", name: "Greek", flag: "🇬🇷", nativeName: "Ελληνικά"),
        SupportedLanguage(code: "en", name: "English", flag: "🇬🇧", nativeName: "English"),
        SupportedLanguage(code: "es", name: "Spanish", flag: "🇪🇸", nativeName: "Español"),
        SupportedLanguage(code: "et", name: "Estonian", flag: "🇪🇪", nativeName: "Eesti"),
        SupportedLanguage(code: "eu", name: "Basque", flag: "🇪🇸", nativeName: "Euskara"),
        SupportedLanguage(code: "fa", name: "Persian", flag: "🇮🇷", nativeName: "فارسی"),
        SupportedLanguage(code: "fi", name: "Finnish", flag: "🇫🇮", nativeName: "Suomi"),
        SupportedLanguage(code: "fo", name: "Faroese", flag: "🇫🇴", nativeName: "Føroyskt"),
        SupportedLanguage(code: "fr", name: "French", flag: "🇫🇷", nativeName: "Français"),
        SupportedLanguage(code: "gl", name: "Galician", flag: "🇪🇸", nativeName: "Galego"),
        SupportedLanguage(code: "gu", name: "Gujarati", flag: "🇮🇳", nativeName: "ગુજરાતી"),
        SupportedLanguage(code: "ha", name: "Hausa", flag: "🇳🇬", nativeName: "Hausa"),
        SupportedLanguage(code: "haw", name: "Hawaiian", flag: "🌺", nativeName: "ʻŌlelo Hawaiʻi"),
        SupportedLanguage(code: "he", name: "Hebrew", flag: "🇮🇱", nativeName: "עברית"),
        SupportedLanguage(code: "hi", name: "Hindi", flag: "🇮🇳", nativeName: "हिन्दी"),
        SupportedLanguage(code: "hr", name: "Croatian", flag: "🇭🇷", nativeName: "Hrvatski"),
        SupportedLanguage(code: "ht", name: "Haitian", flag: "🇭🇹", nativeName: "Kreyòl Ayisyen"),
        SupportedLanguage(code: "hu", name: "Hungarian", flag: "🇭🇺", nativeName: "Magyar"),
        SupportedLanguage(code: "hy", name: "Armenian", flag: "🇦🇲", nativeName: "Հայերեն"),
        SupportedLanguage(code: "id", name: "Indonesian", flag: "🇮🇩", nativeName: "Bahasa Indonesia"),
        SupportedLanguage(code: "is", name: "Icelandic", flag: "🇮🇸", nativeName: "Íslenska"),
        SupportedLanguage(code: "it", name: "Italian", flag: "🇮🇹", nativeName: "Italiano"),
        SupportedLanguage(code: "ja", name: "Japanese", flag: "🇯🇵", nativeName: "日本語"),
        SupportedLanguage(code: "jw", name: "Javanese", flag: "🇮🇩", nativeName: "Basa Jawa"),
        SupportedLanguage(code: "ka", name: "Georgian", flag: "🇬🇪", nativeName: "ქართული"),
        SupportedLanguage(code: "kk", name: "Kazakh", flag: "🇰🇿", nativeName: "Қазақша"),
        SupportedLanguage(code: "km", name: "Khmer", flag: "🇰🇭", nativeName: "ខ្មែរ"),
        SupportedLanguage(code: "kn", name: "Kannada", flag: "🇮🇳", nativeName: "ಕನ್ನಡ"),
        SupportedLanguage(code: "ko", name: "Korean", flag: "🇰🇷", nativeName: "한국어"),
        SupportedLanguage(code: "la", name: "Latin", flag: "🏛️", nativeName: "Latina"),
        SupportedLanguage(code: "lb", name: "Luxembourgish", flag: "🇱🇺", nativeName: "Lëtzebuergesch"),
        SupportedLanguage(code: "ln", name: "Lingala", flag: "🇨🇩", nativeName: "Lingála"),
        SupportedLanguage(code: "lo", name: "Lao", flag: "🇱🇦", nativeName: "ລາວ"),
        SupportedLanguage(code: "lt", name: "Lithuanian", flag: "🇱🇹", nativeName: "Lietuvių"),
        SupportedLanguage(code: "lv", name: "Latvian", flag: "🇱🇻", nativeName: "Latviešu"),
        SupportedLanguage(code: "mg", name: "Malagasy", flag: "🇲🇬", nativeName: "Malagasy"),
        SupportedLanguage(code: "mi", name: "Maori", flag: "🇳🇿", nativeName: "Te Reo Māori"),
        SupportedLanguage(code: "mk", name: "Macedonian", flag: "🇲🇰", nativeName: "Македонски"),
        SupportedLanguage(code: "ml", name: "Malayalam", flag: "🇮🇳", nativeName: "മലയാളം"),
        SupportedLanguage(code: "mn", name: "Mongolian", flag: "🇲🇳", nativeName: "Монгол"),
        SupportedLanguage(code: "mr", name: "Marathi", flag: "🇮🇳", nativeName: "मराठी"),
        SupportedLanguage(code: "ms", name: "Malay", flag: "🇲🇾", nativeName: "Bahasa Melayu"),
        SupportedLanguage(code: "mt", name: "Maltese", flag: "🇲🇹", nativeName: "Malti"),
        SupportedLanguage(code: "my", name: "Myanmar", flag: "🇲🇲", nativeName: "မြန်မာဘာသာ"),
        SupportedLanguage(code: "ne", name: "Nepali", flag: "🇳🇵", nativeName: "नेपाली"),
        SupportedLanguage(code: "nl", name: "Dutch", flag: "🇳🇱", nativeName: "Nederlands"),
        SupportedLanguage(code: "nn", name: "Norwegian Nynorsk", flag: "🇳🇴", nativeName: "Nynorsk"),
        SupportedLanguage(code: "no", name: "Norwegian", flag: "🇳🇴", nativeName: "Norsk"),
        SupportedLanguage(code: "oc", name: "Occitan", flag: "🇫🇷", nativeName: "Occitan"),
        SupportedLanguage(code: "pa", name: "Punjabi", flag: "🇮🇳", nativeName: "ਪੰਜਾਬੀ"),
        SupportedLanguage(code: "pl", name: "Polish", flag: "🇵🇱", nativeName: "Polski"),
        SupportedLanguage(code: "ps", name: "Pashto", flag: "🇦🇫", nativeName: "پښتو"),
        SupportedLanguage(code: "pt", name: "Portuguese", flag: "🇵🇹", nativeName: "Português"),
        SupportedLanguage(code: "ro", name: "Romanian", flag: "🇷🇴", nativeName: "Română"),
        SupportedLanguage(code: "ru", name: "Russian", flag: "🇷🇺", nativeName: "Русский"),
        SupportedLanguage(code: "sa", name: "Sanskrit", flag: "🇮🇳", nativeName: "संस्कृतम्"),
        SupportedLanguage(code: "sd", name: "Sindhi", flag: "🇵🇰", nativeName: "سنڌي"),
        SupportedLanguage(code: "si", name: "Sinhala", flag: "🇱🇰", nativeName: "සිංහල"),
        SupportedLanguage(code: "sk", name: "Slovak", flag: "🇸🇰", nativeName: "Slovenčina"),
        SupportedLanguage(code: "sl", name: "Slovenian", flag: "🇸🇮", nativeName: "Slovenščina"),
        SupportedLanguage(code: "sn", name: "Shona", flag: "🇿🇼", nativeName: "ChiShona"),
        SupportedLanguage(code: "so", name: "Somali", flag: "🇸🇴", nativeName: "Soomaali"),
        SupportedLanguage(code: "sq", name: "Albanian", flag: "🇦🇱", nativeName: "Shqip"),
        SupportedLanguage(code: "sr", name: "Serbian", flag: "🇷🇸", nativeName: "Српски"),
        SupportedLanguage(code: "su", name: "Sundanese", flag: "🇮🇩", nativeName: "Basa Sunda"),
        SupportedLanguage(code: "sv", name: "Swedish", flag: "🇸🇪", nativeName: "Svenska"),
        SupportedLanguage(code: "sw", name: "Swahili", flag: "🇰🇪", nativeName: "Kiswahili"),
        SupportedLanguage(code: "ta", name: "Tamil", flag: "🇮🇳", nativeName: "தமிழ்"),
        SupportedLanguage(code: "te", name: "Telugu", flag: "🇮🇳", nativeName: "తెలుగు"),
        SupportedLanguage(code: "tg", name: "Tajik", flag: "🇹🇯", nativeName: "Тоҷикӣ"),
        SupportedLanguage(code: "th", name: "Thai", flag: "🇹🇭", nativeName: "ไทย"),
        SupportedLanguage(code: "tk", name: "Turkmen", flag: "🇹🇲", nativeName: "Türkmen"),
        SupportedLanguage(code: "tl", name: "Tagalog", flag: "🇵🇭", nativeName: "Tagalog"),
        SupportedLanguage(code: "tr", name: "Turkish", flag: "🇹🇷", nativeName: "Türkçe"),
        SupportedLanguage(code: "tt", name: "Tatar", flag: "🇷🇺", nativeName: "Татарча"),
        SupportedLanguage(code: "uk", name: "Ukrainian", flag: "🇺🇦", nativeName: "Українська"),
        SupportedLanguage(code: "ur", name: "Urdu", flag: "🇵🇰", nativeName: "اردو"),
        SupportedLanguage(code: "uz", name: "Uzbek", flag: "🇺🇿", nativeName: "O'zbek"),
        SupportedLanguage(code: "vi", name: "Vietnamese", flag: "🇻🇳", nativeName: "Tiếng Việt"),
        SupportedLanguage(code: "yi", name: "Yiddish", flag: "🕎", nativeName: "ייִדיש"),
        SupportedLanguage(code: "yo", name: "Yoruba", flag: "🇳🇬", nativeName: "Yorùbá"),
        SupportedLanguage(code: "yue", name: "Cantonese", flag: "🇭🇰", nativeName: "廣東話"),
        SupportedLanguage(code: "zh", name: "Chinese", flag: "🇨🇳", nativeName: "中文"),
    ]

    // MARK: - Lookup

    /// Find a language by its code
    static func language(forCode code: String) -> SupportedLanguage? {
        all.first { $0.code == code }
    }

    /// Common languages shown at the top of pickers
    static let common: [SupportedLanguage] = [
        language(forCode: "en")!,
        language(forCode: "es")!,
        language(forCode: "fr")!,
        language(forCode: "de")!,
        language(forCode: "it")!,
        language(forCode: "pt")!,
        language(forCode: "ru")!,
        language(forCode: "ja")!,
        language(forCode: "zh")!,
        language(forCode: "ko")!,
        language(forCode: "ar")!,
        language(forCode: "hi")!,
    ]
}
