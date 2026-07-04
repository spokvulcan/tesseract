//
//  ServerDashboardView.swift
//  tesseract
//

import SwiftUI

/// Server dashboard, "telemetry face, console under ⌘`":
/// - system toolbar carries server status, endpoint, Stop, and the console
///   toggle (the functional layer — Liquid Glass belongs there and only there)
/// - the content layer is plain: a hero band of live numbers over the
///   streaming transcript
/// - the full request log lives in a slide-up console drawer (⌘`)
struct ServerDashboardView: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(HTTPServer.self) private var httpServer
    @Environment(ServerGenerationLog.self) private var log

    @AppStorage("server.console.open") private var isConsoleOpen = false
    @State private var justCopiedEndpoint = false

    var body: some View {
        VStack(spacing: 0) {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                ServerHeroBand(trace: heroTrace, now: context.date)
            }

            Divider()

            transcript
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            if isConsoleOpen {
                ServerConsoleDrawer(onClose: { isConsoleOpen = false })
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.spring(response: 0.32, dampingFraction: 0.86), value: isConsoleOpen)
        .navigationTitle("Dashboard")
        .toolbar {
            ToolbarItem(placement: .navigation) {
                HStack(spacing: Theme.Spacing.md) {
                    statusItem
                    endpointItem
                }
                .padding(.horizontal, Theme.Spacing.xs)
            }

            ToolbarItemGroup(placement: .primaryAction) {
                if let cancellable = cancellableTrace {
                    Button {
                        log.requestCancel(traceID: cancellable.id)
                    } label: {
                        Label {
                            Text("Stop")
                        } icon: {
                            Image(systemName: "stop.fill")
                                .foregroundStyle(.red)
                        }
                    }
                    .keyboardShortcut(".", modifiers: .command)
                    .help("Stop the current generation (⌘.)")
                }

                Button {
                    isConsoleOpen.toggle()
                } label: {
                    Label("Console", systemImage: "square.bottomthird.inset.filled")
                }
                .keyboardShortcut("`", modifiers: .command)
                .help(isConsoleOpen ? "Hide console (⌘`)" : "Show console (⌘`)")

                Menu {
                    ShareLink(item: heroTrace?.concatenatedText ?? "") {
                        Label("Share Output", systemImage: "square.and.arrow.up")
                    }
                    .disabled((heroTrace?.concatenatedText ?? "").isEmpty)

                    Button {
                        revealRawRequests()
                    } label: {
                        Label("Reveal Raw Requests", systemImage: "doc.text.magnifyingglass")
                    }

                    Divider()

                    Button(role: .destructive) {
                        log.clear()
                    } label: {
                        Label("Clear Requests", systemImage: "trash")
                    }
                    .disabled(log.traces.isEmpty)
                } label: {
                    Label("More", systemImage: "ellipsis.circle")
                }
            }
        }
    }

    // MARK: - Content

    @ViewBuilder
    private var transcript: some View {
        if let heroTrace {
            StreamingSpanListView(trace: heroTrace)
        } else {
            ContentUnavailableView {
                Label("No Requests Yet", systemImage: "waveform")
            } description: {
                Text("Requests to /v1/chat/completions stream in here, token by token.")
            }
        }
    }

    /// The request on stage: the explicit selection when valid, else the
    /// newest trace. New requests select themselves on arrival.
    private var heroTrace: RequestTrace? {
        if let id = log.selectedTraceID,
            let selected = log.traces.first(where: { $0.id == id })
        {
            return selected
        }
        return log.traces.last
    }

    private var cancellableTrace: RequestTrace? {
        log.traces.last { $0.isActive && $0.isCancellable && !$0.cancelRequested }
    }

    // MARK: - Toolbar items

    private var runState: ServerRunState {
        ServerRunState(
            enabled: settings.isServerEnabled,
            isRunning: httpServer.isRunning,
            isStarting: httpServer.isStarting,
            lastStartError: httpServer.lastStartError
        )
    }

    private var statusItem: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(runState.dotColor)
                .frame(width: 7, height: 7)
            Text(runState.displayLabel)
                .font(.caption.weight(.semibold))
        }
        .help(runState.failureMessage ?? runState.displayLabel)
    }

    private var endpointItem: some View {
        HStack(spacing: 4) {
            Text(serverEndpointURL(port: settings.serverPort))
                .font(.caption.monospaced())
                .foregroundStyle(runState == .running ? .secondary : .tertiary)
                .textSelection(.enabled)

            Button {
                copyEndpoint()
            } label: {
                Image(systemName: justCopiedEndpoint ? "checkmark" : "document.on.document")
                    .contentTransition(.symbolEffect(.replace))
                    .font(.caption)
            }
            .buttonStyle(.borderless)
            .help("Copy Endpoint")
        }
    }

    private func copyEndpoint() {
        copyServerEndpointToPasteboard(port: settings.serverPort)
        justCopiedEndpoint = true
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(1.5))
            justCopiedEndpoint = false
        }
    }

    private func revealRawRequests() {
        NSWorkspace.shared.open(HTTPRequestLogger.shared.directoryURL)
    }
}
