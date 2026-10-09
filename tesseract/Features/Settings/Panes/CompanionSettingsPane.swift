//
//  CompanionSettingsPane.swift
//  tesseract
//
//  The Companion pane: Jarvis on or off, when his moments come, quiet hours,
//  nudges, and which Reminders lists are the owner's Areas. Everything lives
//  in the one Settings store; the Areas map is JSON in one setting.
//

import ServiceManagement
import SwiftUI

struct CompanionSettingsPane: View {
    @Environment(SettingsManager.self) private var settings
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject private var container: DependencyContainer
    @State private var showingLaunchAtLoginAsk = false

    var body: some View {
        @Bindable var settings = settings
        Form {
            Section {
                Toggle("Companion", isOn: $settings.companionHeartbeatEnabled)
                Toggle("Speak Urgent Things Aloud", isOn: $settings.companionSpeaks)
                    .disabled(!settings.companionHeartbeatEnabled)
                Button("Open Profile…") { openWindow(id: WindowID.profile) }
                // Never a silent login-item flip: the toggle reads and writes
                // the real SMAppService state.
                Toggle(
                    "Launch at Login",
                    isOn: Binding(
                        get: { SMAppService.mainApp.status == .enabled },
                        set: { wanted in
                            do {
                                if wanted {
                                    try SMAppService.mainApp.register()
                                } else {
                                    try SMAppService.mainApp.unregister()
                                }
                            } catch {
                                Log.companion.error("Launch-at-login change failed: \(error)")
                            }
                        }))
            } header: {
                Text("Jarvis")
            } footer: {
                Text(
                    "Jarvis helps you run your day from the Today page: a plan when you first sit down, a card when you come back, a wrap-up in the evening, and a nudge before each event. He runs on the agent model; your reminders and events live in Reminders and Calendar."
                )
            }

            Section {
                Stepper(
                    "Morning Plan from \(hour(settings.companionMorningStartHour))",
                    value: $settings.companionMorningStartHour, in: 3...11)
                Stepper(
                    "…until \(hour(settings.companionMorningEndHour))",
                    value: $settings.companionMorningEndHour,
                    in: (settings.companionMorningStartHour + 1)...16)
                DatePicker(
                    "Evening Wrap-up", selection: minutesBinding($settings.companionEveningMinutes),
                    displayedComponents: .hourAndMinute)
                Stepper(
                    "Welcome-back card after \(settings.companionBreakpointAwayMinutes) min away",
                    value: $settings.companionBreakpointAwayMinutes, in: 5...120, step: 5)
                Stepper(
                    "Nudge \(settings.companionNudgeLeadMinutes) min before events",
                    value: $settings.companionNudgeLeadMinutes, in: 0...60, step: 5)
            } header: {
                Text("Your Day")
            }

            Section {
                DatePicker(
                    "Quiet from", selection: minutesBinding($settings.companionQuietStartMinutes),
                    displayedComponents: .hourAndMinute)
                DatePicker(
                    "Until", selection: minutesBinding($settings.companionQuietEndMinutes),
                    displayedComponents: .hourAndMinute)
                Toggle("Say When Tomorrow Starts as They Begin", isOn: $settings.companionWindDown)
                    .disabled(!settings.companionHeartbeatEnabled)
            } header: {
                Text("Quiet Hours")
            } footer: {
                Text(
                    "Jarvis's own banners, panels and voice stop in quiet hours. If you're still at the Mac as they begin, one banner says when tomorrow starts. Your reminders and event nudges still fire."
                )
            }

            AreasSection()

            CodingAgentsSection()

            NotificationRulesSection()

            Section {
                Stepper(
                    "Compact past \(settings.companionThreadCeilingTokens / 1000)k tokens",
                    value: $settings.companionThreadCeilingTokens, in: 20_000...200_000,
                    step: 10_000)
            } header: {
                Text("Day Thread")
            } footer: {
                Text(
                    "Each day is one conversation that grows through the day and starts fresh at 04:00. Past this size it is summarised to stay sharp."
                )
            }
        }
        .formStyle(.grouped)
        .onChange(of: settings.companionHeartbeatEnabled) { _, enabled in
            if enabled, !settings.companionLaunchAtLoginAsked {
                settings.companionLaunchAtLoginAsked = true
                showingLaunchAtLoginAsk = true
            }
        }
        .alert("Keep Jarvis running?", isPresented: $showingLaunchAtLoginAsk) {
            Button("Launch at Login") {
                do { try SMAppService.mainApp.register() } catch {
                    Log.companion.error("Launch-at-login register failed: \(error)")
                }
            }
            Button("Not Now", role: .cancel) {}
        } message: {
            Text(
                "The Companion only runs while Tesseract is open. Start it at login so it keeps up with your day — change this anytime with Launch at Login."
            )
        }
    }

    private func hour(_ value: Int) -> String { String(format: "%02d:00", value) }

    /// A time-of-day picker over minutes after midnight.
    private func minutesBinding(_ minutes: Binding<Int>) -> Binding<Date> {
        Binding(
            get: {
                Calendar.current.date(
                    byAdding: .minute, value: minutes.wrappedValue,
                    to: Calendar.current.startOfDay(for: Date())) ?? Date()
            },
            set: { date in
                let parts = Calendar.current.dateComponents([.hour, .minute], from: date)
                minutes.wrappedValue = (parts.hour ?? 0) * 60 + (parts.minute ?? 0)
            })
    }
}

/// Which Reminders lists are Areas, what they're called, and which is the
/// Inbox; and the calendar new events go to.
private struct AreasSection: View {
    @Environment(SettingsManager.self) private var settings
    @EnvironmentObject private var container: DependencyContainer

    var body: some View {
        let agenda = container.agenda
        let lists = agenda.store.reminderLists()
        let calendars = agenda.store.eventCalendars().filter(\.isWritable)
        var map = AreaMap(json: settings.companionAreasJSON)
        Section {
            if !agenda.access.canUseReminders {
                HStack {
                    Text("Tesseract needs access to Reminders to map your Areas.")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Allow Access") { Task { await agenda.requestAccessIfNeeded() } }
                }
            } else {
                ForEach(lists) { list in
                    let isArea =
                        map.entries.isEmpty || map.entries.contains { $0.listID == list.id }
                    HStack {
                        Toggle(
                            list.title,
                            isOn: Binding(
                                get: { isArea },
                                set: { on in
                                    if map.entries.isEmpty {
                                        map.entries = lists.map {
                                            .init(listID: $0.id, name: $0.title)
                                        }
                                    }
                                    if on {
                                        if !map.entries.contains(where: { $0.listID == list.id }) {
                                            map.entries.append(
                                                .init(listID: list.id, name: list.title))
                                        }
                                    } else {
                                        map.entries.removeAll { $0.listID == list.id }
                                    }
                                    settings.companionAreasJSON = map.json
                                }))
                        Spacer()
                        if isArea {
                            TextField(
                                "Area name",
                                text: Binding(
                                    get: {
                                        map.entries.first { $0.listID == list.id }?.name
                                            ?? list.title
                                    },
                                    set: { name in
                                        if map.entries.isEmpty {
                                            map.entries = lists.map {
                                                .init(listID: $0.id, name: $0.title)
                                            }
                                        }
                                        if let index = map.entries.firstIndex(where: {
                                            $0.listID == list.id
                                        }) {
                                            map.entries[index].name = name
                                        }
                                        settings.companionAreasJSON = map.json
                                    })
                            )
                            .frame(width: 160)
                        }
                    }
                }
                Picker(
                    "Inbox",
                    selection: Binding(
                        get: { map.inbox(in: lists)?.id ?? "" },
                        set: { id in
                            map.inboxListID = id
                            settings.companionAreasJSON = map.json
                        })
                ) {
                    ForEach(lists) { Text($0.title).tag($0.id) }
                }
            }
            if agenda.access.canUseCalendar {
                Picker(
                    "New Events Go To",
                    selection: Binding(
                        get: { settings.companionDefaultCalendarID ?? "" },
                        set: { settings.companionDefaultCalendarID = $0.isEmpty ? nil : $0 })
                ) {
                    Text("Default Calendar").tag("")
                    ForEach(calendars) { Text($0.title).tag($0.id) }
                }
            }
        } header: {
            Text("Areas")
        } footer: {
            Text(
                "Areas are the parts of your life a day spans — each one a Reminders list. Captures without a home land in the Inbox."
            )
        }
    }
}

/// Claude Code's hooks: install or remove, and the one-liner for a terminal.
private struct CodingAgentsSection: View {
    @Environment(SettingsManager.self) private var settings
    @State private var installed = ClaudeCodeHooks.installed()
    @State private var problem: String?

    var body: some View {
        let port = Int(HTTPServer.clampedPort(settings.serverPort))
        Section {
            HStack {
                Text("Claude Code")
                Spacer()
                Text(installed ? "Connected" : "Not connected").foregroundStyle(.secondary)
                if installed {
                    Button("Disconnect") { change { try ClaudeCodeHooks.uninstall() } }
                } else {
                    Button("Connect") {
                        change {
                            settings.isServerEnabled = true
                            try ClaudeCodeHooks.install(port: port)
                        }
                    }
                }
            }
            Text(ClaudeCodeHooks.oneLiner(port: port))
                .font(.system(.callout, design: .monospaced))
                .textSelection(.enabled)
                .foregroundStyle(.secondary)
            if let problem {
                Text(problem).foregroundStyle(.red)
            }
        } header: {
            Text("Coding Agents")
        } footer: {
            Text(
                "Claude Code's hooks tell Jarvis when an agent is waiting on you or has finished. Waiting agents show in Today; if you've been out of the terminal for two minutes, Jarvis says so once. Connecting writes to ~/.claude/settings.json (with a backup) and turns on the local server, which only listens on this Mac. Or run the line above in a terminal."
            )
        }
    }

    private func change(_ action: () throws -> Void) {
        do {
            try action()
            problem = nil
        } catch {
            problem = "Couldn't update ~/.claude/settings.json: \(error.localizedDescription)"
        }
        installed = ClaudeCodeHooks.installed()
    }
}

/// The owner's notification rules, set by talking to Jarvis.
private struct NotificationRulesSection: View {
    @Environment(SettingsManager.self) private var settings

    var body: some View {
        let rules = TriageRules.decode(settings.companionTriageRulesJSON)
        Section {
            if rules.isEmpty {
                Text("No rules yet.").foregroundStyle(.secondary)
            }
            ForEach(rules) { rule in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(rule.phrase.isEmpty ? rule.action.rawValue : rule.phrase)
                        Text(summary(rule)).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Remove") {
                        settings.companionTriageRulesJSON = TriageRules.encode(
                            rules.filter { $0.id != rule.id })
                    }
                }
            }
        } header: {
            Text("Notification Rules")
        } footer: {
            Text(
                "Tell Jarvis in any chat — \"never tell me about CI passing\" — and it becomes a rule here. Rules apply before Jarvis sees a notification."
            )
        }
    }

    private func summary(_ rule: TriageRule) -> String {
        var parts = [rule.action.rawValue]
        if let app = rule.app { parts.append("app: \(app)") }
        if let sender = rule.sender { parts.append("from: \(sender)") }
        if !rule.keywords.isEmpty {
            parts.append("words: \(rule.keywords.joined(separator: ", "))")
        }
        return parts.joined(separator: " · ")
    }
}
