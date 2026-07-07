import SwiftUI

// MARK: - MCPServersSection

/// Settings UI for the agent's MCP **client** servers (PRD #190): the configured
/// servers with their live connection state and tool lists, a per-server
/// enable/disable switch, and add/remove with an explicit consent step. Embeds
/// as a `Section` in the Server configuration `Form`, next to the Browser Access
/// (server) controls so all MCP configuration lives together.
struct MCPServersSection: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(MCPClientManager.self) private var manager

    @State private var isAdding = false

    var body: some View {
        @Bindable var settings = settings

        Section {
            if settings.mcpServers.isEmpty {
                Text("No MCP servers added.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach($settings.mcpServers) { $config in
                    MCPServerRow(
                        config: $config,
                        connection: manager.connection(id: config.id),
                        onRemove: { settings.mcpServers.removeAll { $0.id == config.id } })
                }
            }
            Button {
                isAdding = true
            } label: {
                Label("Add MCP Server…", systemImage: "plus")
            }
        } header: {
            Text("MCP Servers")
        } footer: {
            Text(
                "Connect the agent to Model Context Protocol servers over HTTP; their tools appear "
                    + "to the agent alongside the built-ins. Only add servers you trust — their "
                    + "tools run inside your agent."
            )
        }
        .sheet(isPresented: $isAdding) {
            MCPAddServerSheet { newServer in
                settings.mcpServers.append(newServer)
            }
        }
    }
}

// MARK: - MCPServerRow

private struct MCPServerRow: View {
    @Binding var config: MCPServerConfig
    let connection: MCPServerConnection?
    let onRemove: () -> Void

    @State private var showingTools = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(config.name)
                    Text(config.url)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
                stateBadge
                Toggle("", isOn: $config.enabled)
                    .labelsHidden()
                    .help(config.enabled ? "Disable this server" : "Enable this server")
                Button(role: .destructive, action: onRemove) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Remove server")
            }
            if config.enabled { detail }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder private var stateBadge: some View {
        switch connection?.state {
        case .connected:
            Label("\(connection?.tools.count ?? 0)", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.caption)
                .help("Connected — \(connection?.tools.count ?? 0) tools")
        case .connecting, .idle, .none:
            Image(systemName: "circle.dotted")
                .foregroundStyle(.secondary)
                .help("Connecting…")
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .help("Connection failed")
        }
    }

    @ViewBuilder private var detail: some View {
        switch connection?.state {
        case .connected:
            if let tools = connection?.tools, !tools.isEmpty {
                DisclosureGroup(isExpanded: $showingTools) {
                    ForEach(tools, id: \.name) { tool in
                        Text(tool.name)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                } label: {
                    Text("\(tools.count) tools").font(.caption).foregroundStyle(.secondary)
                }
            }
        case .failed(let message):
            Text(message)
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        default:
            EmptyView()
        }
    }
}

// MARK: - MCPAddServerSheet

/// The explicit consent step (US #3): the user names a server, gives its URL and
/// any protected headers, and confirms — nothing connects until they add it.
private struct MCPAddServerSheet: View {
    @Environment(\.dismiss) private var dismiss
    let onAdd: (MCPServerConfig) -> Void

    @State private var name = ""
    @State private var url = ""
    @State private var headers: [HeaderField] = []

    private struct HeaderField: Identifiable {
        let id = UUID()
        var key = ""
        var value = ""
    }

    private var trimmedName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var trimmedURL: String { url.trimmingCharacters(in: .whitespacesAndNewlines) }

    private var isValid: Bool {
        guard !trimmedName.isEmpty else { return false }
        guard let scheme = URL(string: trimmedURL)?.scheme?.lowercased() else { return false }
        return scheme == "http" || scheme == "https"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Add MCP Server")
                .font(.headline)
                .padding([.top, .horizontal])

            Form {
                Section {
                    TextField("Name", text: $name)
                    TextField("URL (https://…/mcp)", text: $url)
                } footer: {
                    Text(
                        "Adding a server lets its tools run inside your agent. Only add servers "
                            + "you trust.")
                }

                Section("Custom Headers (optional)") {
                    ForEach($headers) { $header in
                        HStack {
                            TextField("Header", text: $header.key)
                            TextField("Value", text: $header.value)
                        }
                    }
                    Button("Add Header") { headers.append(HeaderField()) }
                        .font(.caption)
                }
            }
            .formStyle(.grouped)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                Button("Add & Enable") {
                    onAdd(makeConfig())
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!isValid)
            }
            .padding()
        }
        .frame(width: 440)
    }

    private func makeConfig() -> MCPServerConfig {
        var headerDict: [String: String] = [:]
        for header in headers {
            let key = header.key.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else { continue }
            headerDict[key] = header.value
        }
        return MCPServerConfig(
            name: trimmedName, url: trimmedURL, enabled: true, headers: headerDict,
            transport: .http)
    }
}
