//
//  Logging.swift
//  tesseract
//

import os

enum Log {
    static let audio = Logger(subsystem: "com.tesseract.app", category: "audio")
    static let transcription = Logger(subsystem: "com.tesseract.app", category: "transcription")
    static let general = Logger(subsystem: "com.tesseract.app", category: "general")
}
