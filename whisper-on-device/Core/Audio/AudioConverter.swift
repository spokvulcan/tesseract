//
//  AudioConverter.swift
//  whisper-on-device
//

import Foundation
import AVFoundation
import Accelerate

enum AudioConverter {
    static let whisperSampleRate: Double = 16000

    /// Convert AVAudioPCMBuffer to Float array suitable for WhisperKit
    static func convertToWhisperFormat(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData?[0] else {
            return []
        }

        let frameCount = Int(buffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData, count: frameCount))

        // Resample if needed
        if buffer.format.sampleRate != whisperSampleRate {
            return resample(samples, from: buffer.format.sampleRate, to: whisperSampleRate)
        }

        return samples
    }

    /// Resample audio from one sample rate to another using linear interpolation
    static func resample(_ samples: [Float], from sourceSampleRate: Double, to targetSampleRate: Double) -> [Float] {
        guard sourceSampleRate != targetSampleRate else { return samples }
        guard !samples.isEmpty else { return [] }

        let ratio = targetSampleRate / sourceSampleRate
        let outputLength = Int(Double(samples.count) * ratio)

        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)

        // Linear interpolation resampling
        for i in 0..<outputLength {
            let position = Double(i) / ratio
            let index = Int(position)
            let fraction = Float(position - Double(index))

            if index + 1 < samples.count {
                output[i] = samples[index] * (1 - fraction) + samples[index + 1] * fraction
            } else if index < samples.count {
                output[i] = samples[index]
            }
        }

        return output
    }

    /// Normalize audio samples to [-1, 1] range
    static func normalize(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return [] }

        var maxValue: Float = 0
        vDSP_maxmgv(samples, 1, &maxValue, vDSP_Length(samples.count))

        guard maxValue > 0 else { return samples }

        var output = [Float](repeating: 0, count: samples.count)
        var scale = 1.0 / maxValue
        vDSP_vsmul(samples, 1, &scale, &output, 1, vDSP_Length(samples.count))

        return output
    }

    /// Calculate RMS (root mean square) of audio samples
    static func calculateRMS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }

        var rms: Float = 0
        vDSP_rmsqv(samples, 1, &rms, vDSP_Length(samples.count))
        return rms
    }

    /// Convert RMS to decibels
    static func rmsToDecibels(_ rms: Float) -> Float {
        guard rms > 0 else { return -Float.infinity }
        return 20 * log10(rms)
    }

    /// Check if audio contains clipping (samples at max value)
    static func hasClipping(_ samples: [Float], threshold: Float = 0.99) -> Bool {
        samples.contains { abs($0) >= threshold }
    }

    /// Trim silence from beginning and end of audio
    static func trimSilence(
        _ samples: [Float],
        sampleRate: Double,
        threshold: Float = 0.01,
        minDuration: TimeInterval = 0.1
    ) -> [Float] {
        guard !samples.isEmpty else { return [] }

        let minSamples = Int(minDuration * sampleRate)

        // Find first non-silent sample
        var startIndex = 0
        for (index, sample) in samples.enumerated() {
            if abs(sample) > threshold {
                startIndex = max(0, index - minSamples)
                break
            }
        }

        // Find last non-silent sample
        var endIndex = samples.count - 1
        for index in stride(from: samples.count - 1, through: 0, by: -1) {
            if abs(samples[index]) > threshold {
                endIndex = min(samples.count - 1, index + minSamples)
                break
            }
        }

        guard startIndex < endIndex else { return [] }

        return Array(samples[startIndex...endIndex])
    }
}
