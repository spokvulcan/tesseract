//
//  ShareViewController.swift
//  tesseract-share
//
//  "Read in Tesseract" in the share sheet (#515): takes the shared page, PDF,
//  file or text, leaves its text in the app group's inbox, and closes. It
//  never runs the voice (an extension gets far less memory than an app); the
//  app adds the text to the Library and opens it when it next comes to the
//  front.
//

import SwiftUI
import UIKit

final class ShareViewController: UIViewController {
    private let status = ShareStatus()

    override func viewDidLoad() {
        super.viewDidLoad()
        let host = UIHostingController(rootView: ShareStatusView(status: status))
        addChild(host)
        host.view.frame = view.bounds
        host.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        host.view.backgroundColor = .clear
        view.addSubview(host.view)
        host.didMove(toParent: self)
        Task { await take() }
    }

    private func take() async {
        let items = extensionContext?.inputItems.compactMap { $0 as? NSExtensionItem } ?? []
        do {
            let incoming = try await SharedContent.text(from: items)
            guard let inbox = LibraryInbox.shared() else {
                throw SharedContent.Failure.nothingToRead
            }
            try inbox.drop(incoming)
            status.phase = .added(incoming.title)
            try? await Task.sleep(for: .seconds(1.2))
            extensionContext?.completeRequest(returningItems: nil)
        } catch {
            status.phase = .failed
            try? await Task.sleep(for: .seconds(2.5))
            extensionContext?.cancelRequest(withError: error)
        }
    }
}

@Observable @MainActor
private final class ShareStatus {
    enum Phase: Equatable {
        case reading
        case added(String?)
        case failed
    }
    var phase = Phase.reading
}

private struct ShareStatusView: View {
    let status: ShareStatus

    var body: some View {
        VStack(spacing: 12) {
            switch status.phase {
            case .reading:
                ProgressView()
                Text("Adding to your Library…")
            case .added(let title):
                Image(systemName: "checkmark.circle.fill")
                    .font(.largeTitle)
                    .foregroundStyle(.green)
                Text(title.map { "Added “\($0)”" } ?? "Added to your Library")
                    .multilineTextAlignment(.center)
                Text("Open Tesseract to listen.")
                    .foregroundStyle(.secondary)
            case .failed:
                Image(systemName: "text.badge.xmark")
                    .font(.largeTitle)
                    .foregroundStyle(.secondary)
                Text("There's nothing here to read.")
                Text("Tesseract reads web pages from Safari, PDFs and text.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .font(.headline)
        .padding(28)
        .frame(maxWidth: 320)
        .background(.regularMaterial, in: .rect(cornerRadius: 24))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
