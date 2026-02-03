//
//  WaveformVisualizer.swift
//  tesseract
//

import SwiftUI
import Combine

// MARK: - Audio Level Buffer

@MainActor
final class AudioLevelBuffer: ObservableObject {
    @Published private(set) var levels: [Float]
    private let capacity: Int

    init(capacity: Int = 40) {
        self.capacity = capacity
        self.levels = Array(repeating: 0, count: capacity)
    }

    func push(_ level: Float) {
        levels.removeFirst()
        levels.append(level)
    }

    func reset() {
        levels = Array(repeating: 0, count: capacity)
    }
}

// MARK: - Waveform Canvas

struct WaveformCanvas: View {
    let levels: [Float]
    let color: Color

    private let barSpacing: CGFloat = 2
    private let cornerRadius: CGFloat = 2

    var body: some View {
        Canvas { context, size in
            let barCount = levels.count
            guard barCount > 0 else { return }

            let totalSpacing = barSpacing * CGFloat(barCount - 1)
            let barWidth = (size.width - totalSpacing) / CGFloat(barCount)
            let centerY = size.height / 2
            let maxBarHeight = size.height / 2 - 2  // Leave small padding

            for (index, level) in levels.enumerated() {
                let barHeight = max(2, CGFloat(level) * maxBarHeight)
                let x = CGFloat(index) * (barWidth + barSpacing)

                // Draw mirrored bar (extends up and down from center)
                let rect = CGRect(
                    x: x,
                    y: centerY - barHeight,
                    width: barWidth,
                    height: barHeight * 2
                )

                let path = Path(roundedRect: rect, cornerRadius: cornerRadius)
                context.fill(path, with: .color(color))
            }
        }
    }
}

// MARK: - Waveform Visualizer

struct WaveformVisualizer: View {
    @ObservedObject var audioCapture: AudioCaptureEngine
    let state: DictationState

    @StateObject private var buffer = AudioLevelBuffer()
    @State private var timer: Timer?

    private let updateInterval: TimeInterval = 0.05  // 20Hz

    var body: some View {
        WaveformCanvas(levels: buffer.levels, color: waveformColor)
            .frame(height: 60)
            .opacity(waveformOpacity)
            .animation(.easeInOut(duration: 0.2), value: state)
            .onChange(of: state) { oldState, newState in
                handleStateChange(from: oldState, to: newState)
            }
            .onAppear {
                if state == .recording {
                    startSampling()
                }
            }
            .onDisappear {
                stopSampling()
            }
    }

    private var waveformColor: Color {
        switch state {
        case .recording:
            return .red
        case .processing:
            return .orange
        case .listening:
            return .yellow
        case .idle:
            return .secondary
        case .error:
            return .red
        }
    }

    private var waveformOpacity: Double {
        switch state {
        case .recording:
            return 1.0
        case .processing:
            return 0.7
        case .listening:
            return 0.5
        case .idle:
            return 0.25
        case .error:
            return 0.4
        }
    }

    private func handleStateChange(from oldState: DictationState, to newState: DictationState) {
        if newState == .recording && oldState != .recording {
            buffer.reset()
            startSampling()
        } else if newState != .recording {
            stopSampling()
        }
    }

    private func startSampling() {
        stopSampling()
        timer = Timer.scheduledTimer(withTimeInterval: updateInterval, repeats: true) { _ in
            Task { @MainActor in
                buffer.push(audioCapture.audioLevel)
            }
        }
    }

    private func stopSampling() {
        timer?.invalidate()
        timer = nil
    }
}

#Preview {
    VStack {
        WaveformCanvas(
            levels: (0..<40).map { _ in Float.random(in: 0.1...0.8) },
            color: .red
        )
        .frame(height: 60)
        .padding()
    }
    .frame(width: 400)
}
