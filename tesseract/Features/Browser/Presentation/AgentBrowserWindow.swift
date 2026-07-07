import SwiftUI
import WebKit

// MARK: - AgentBrowserWindow

/// The user's own window onto the **Agent Profile** (ADR-0026, User Story #1):
/// an address bar plus the live `WebView`, opened deliberately from Settings so
/// the user can log into the sites they choose to grant agents. Distinct from
/// the bare, agent-mirroring Browser Session windows — this one carries
/// navigation chrome because a human drives it. Cookies established here persist
/// in the shared profile, so every agent Browser Session is logged in afterwards.
struct AgentBrowserWindow: View {

    let page: WebPage

    @State private var address: String = ""
    /// While the user is editing the field, live `page.url` updates must not
    /// overwrite what they are typing.
    @FocusState private var addressFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            WebView(page)
        }
        .frame(minWidth: 520, minHeight: 400)
        .onAppear {
            if let url = page.url { address = url.absoluteString }
        }
        .onChange(of: page.url) { _, url in
            if !addressFocused, let url { address = url.absoluteString }
        }
    }

    private var toolbar: some View {
        HStack(spacing: 8) {
            Button(action: goBack) {
                Image(systemName: "chevron.left")
            }
            .buttonStyle(.borderless)
            .disabled(page.backForwardList.backList.isEmpty)
            .help("Back")

            Button(action: reload) {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.borderless)
            .help("Reload")

            TextField("Search or enter address", text: $address)
                .textFieldStyle(.roundedBorder)
                .focused($addressFocused)
                .onSubmit(navigate)
        }
        .padding(8)
    }

    private func navigate() {
        guard let url = BrowserURL.normalized(from: address) else { return }
        _ = page.load(URLRequest(url: url))
    }

    private func reload() {
        if let url = page.url {
            _ = page.load(URLRequest(url: url))
        } else {
            navigate()
        }
    }

    private func goBack() {
        guard let previous = page.backForwardList.backList.last else { return }
        _ = page.load(previous)
    }
}
