//
//  DictationStatusView.swift
//  tesseract
//

import SwiftUI

struct StatusIndicator: View {
    enum Badge {
        case dot(Color)
        case spinner
    }

    let badge: Badge
    let title: String
    let detail: String?

    init(badge: Badge, title: String, detail: String?) {
        self.badge = badge
        self.title = title
        self.detail = detail
    }

    init(state: DictationState) {
        switch state {
        case .error(let message):
            self.init(badge: .dot(.red), title: "Error", detail: message)
        case .idle:
            self.init(badge: .dot(.green), title: state.statusText, detail: nil)
        case .listening:
            self.init(badge: .dot(.yellow), title: state.statusText, detail: nil)
        case .recording:
            self.init(badge: .dot(.red), title: state.statusText, detail: nil)
        case .processing:
            self.init(badge: .dot(.orange), title: state.statusText, detail: nil)
        }
    }

    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 6) {
                switch badge {
                case .dot(let color):
                    Circle()
                        .fill(color)
                        .frame(width: 8, height: 8)
                        .accessibilityHidden(true)
                case .spinner:
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityHidden(true)
                }

                Text(title)
                    .font(.system(size: DictationPageStyle.bodySize, weight: .medium))
            }
            .frame(height: 20)

            // Detail line always reserves space so the button doesn't jump.
            Text(detail ?? " ")
                .font(.system(size: DictationPageStyle.bodySize))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(height: 20)
                .opacity(detail != nil ? 1 : 0)
                .help(detail ?? "")
        }
        .frame(height: 44)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Status: \(title)")
        .accessibilityHint(detail ?? "")
    }
}
