//
//  DictationStatusView.swift
//  tesseract
//

import SwiftUI

struct StatusIndicator: View {
    let state: DictationState

    // Fixed height: status line (18) + spacing (2) + detail line (16) = 36
    private let totalHeight: CGFloat = 36

    var body: some View {
        VStack(spacing: 2) {
            // Status line - centered with dot
            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                    .accessibilityHidden(true)

                Text(statusTitle)
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            .frame(height: 18)
            .animation(.easeInOut(duration: 0.2), value: state)

            // Error detail - always reserves space
            Text(statusDetail ?? " ")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(height: 16)
                .opacity(statusDetail != nil ? 1 : 0)
        }
        .frame(height: totalHeight)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Status: \(statusTitle)")
        .accessibilityHint(statusDetail ?? "")
    }

    private var statusTitle: String {
        switch state {
        case .error:
            return "Error"
        default:
            return state.statusText
        }
    }

    private var statusDetail: String? {
        switch state {
        case .error(let message):
            return message
        default:
            return nil
        }
    }

    private var statusColor: Color {
        switch state {
        case .idle:
            return .green
        case .listening:
            return .yellow
        case .recording:
            return .red
        case .processing:
            return .orange
        case .error:
            return .red
        }
    }
}
