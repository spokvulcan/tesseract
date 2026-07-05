//
//  FormatStyles.swift
//  tesseract
//

import Foundation

extension FormatStyle where Self == FloatingPointFormatStyle<Double>.Percent {
    /// "68%" — the whole-number percent shared by download/progress UI
    /// (onboarding tour, agent input strip).
    static var wholePercent: Self {
        .percent.precision(.fractionLength(0))
    }
}
