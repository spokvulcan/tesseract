import Foundation
import Testing
@testable import Tesseract_Agent

// SSRF coverage for the anonymous web path. This policy previously ran through
// the standalone web_fetch tool; with that tool removed (ADR-0028) it is tested
// directly against ``WebAddressGuard``, the single home for it (applied by
// `browser.fetch` via ``EphemeralPageReader/read(url:)``). `browser.search` is
// intentionally exempt — its engine host is a trusted constant.

@MainActor
struct WebAddressGuardTests {

    private enum Verdict: Equatable { case ok, invalid, priv }

    private func verdict(_ urlString: String) -> Verdict {
        guard let url = URL(string: urlString) else { return .invalid }
        do {
            try WebAddressGuard.validate(url)
            return .ok
        } catch WebAddressGuard.GuardError.privateAddress {
            return .priv
        } catch {
            return .invalid
        }
    }

    // MARK: - Private / internal addresses rejected

    @Test func blocksLoopbackIPv4() { #expect(verdict("http://127.0.0.1/") == .priv) }
    @Test func blocksLoopbackIPv6() { #expect(verdict("http://[::1]/") == .priv) }
    @Test func blocksAllZeros() { #expect(verdict("http://0.0.0.0/") == .priv) }
    @Test func blocksLocalhostName() { #expect(verdict("http://localhost:8080/admin") == .priv) }
    @Test func blocks10Range() { #expect(verdict("http://10.0.0.1/internal") == .priv) }
    @Test func blocks192168Range() { #expect(verdict("http://192.168.1.1/config") == .priv) }
    @Test func blocks172PrivateRange() { #expect(verdict("http://172.16.0.1/") == .priv) }
    @Test func blocks169LinkLocal() { #expect(verdict("http://169.254.1.1/") == .priv) }
    @Test func blocksDotLocal() { #expect(verdict("http://myserver.local/") == .priv) }
    @Test func blocksDotInternal() { #expect(verdict("http://db.internal/") == .priv) }

    // MARK: - Malformed / unsafe URLs rejected

    @Test func rejectsFTPScheme() { #expect(verdict("ftp://files.example.com/doc") == .invalid) }
    @Test func rejectsFileScheme() { #expect(verdict("file:///etc/passwd") == .invalid) }
    @Test func rejectsEmbeddedCredentials() {
        #expect(verdict("https://user:pass@example.com/") == .invalid)
    }

    // MARK: - Public addresses pass

    @Test func allowsPublicDomain() { #expect(verdict("https://example.com/path") == .ok) }
    @Test func allowsPublicIP() { #expect(verdict("https://93.184.216.34/") == .ok) }
    // A public 172 address outside the private 16–31 block is allowed.
    @Test func allowsPublic172() { #expect(verdict("http://172.32.0.1/") == .ok) }
}
