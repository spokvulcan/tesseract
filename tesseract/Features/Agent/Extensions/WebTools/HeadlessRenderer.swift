import Foundation
import WebKit

// MARK: - HeadlessRendererError

nonisolated enum HeadlessRendererError: LocalizedError, Sendable {
    case timeout
    case navigationFailed(String)
    case emptyResult

    var errorDescription: String? {
        switch self {
        case .timeout: "Page render timed out"
        case .navigationFailed(let msg): "Navigation failed: \(msg)"
        case .emptyResult: "Rendered page produced no HTML"
        }
    }
}

// MARK: - HeadlessRenderer

/// Renders JavaScript-heavy pages using macOS 26 WebPage API.
/// Single reusable instance with nonPersistent data store for privacy.
@MainActor
final class HeadlessRenderer {
    static let shared = HeadlessRenderer()

    private let page: WebPage

    private init() {
        var config = WebPage.Configuration()
        config.websiteDataStore = .nonPersistent()
        config.applicationNameForUserAgent = "TesseractAgent/1.0"
        page = WebPage(configuration: config)
    }

    /// Render a URL with full JavaScript execution and return the rendered HTML.
    /// Waits for navigation to finish, then allows time for SPA hydration.
    func render(url: URL, timeout: Duration = .seconds(20)) async throws -> String {
        let navigationEvents = page.load(URLRequest(url: url))

        // Create a timeout task that we cancel on success
        let timeoutTask = Task { @MainActor in
            try await Task.sleep(for: timeout)
        }

        defer { timeoutTask.cancel() }

        do {
            for try await event in navigationEvents {
                if event == .finished {
                    // Allow time for SPA framework hydration (React, Vue, Next.js)
                    try await Task.sleep(for: .milliseconds(1500))

                    // Extract fully rendered DOM
                    guard let html = try await page.callJavaScript(
                        "document.documentElement.outerHTML"
                    ) as? String, !html.isEmpty else {
                        throw HeadlessRendererError.emptyResult
                    }
                    return html
                }
            }
            // Navigation sequence ended without reaching .finished
            throw HeadlessRendererError.emptyResult
        } catch is CancellationError {
            throw HeadlessRendererError.timeout
        } catch let error as HeadlessRendererError {
            throw error
        } catch let error as WebPage.NavigationError {
            throw HeadlessRendererError.navigationFailed(mapNavigationError(error))
        }
    }

    private func mapNavigationError(_ error: WebPage.NavigationError) -> String {
        switch error {
        case .failedProvisionalNavigation(let underlying):
            underlying.localizedDescription
        case .pageClosed:
            "Page closed"
        case .webContentProcessTerminated:
            "WebContent process terminated"
        case .invalidURL:
            "Invalid URL"
        @unknown default:
            error.localizedDescription
        }
    }
}
