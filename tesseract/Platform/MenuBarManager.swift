//
//  MenuBarManager.swift
//  tesseract
//
//  The status-bar surface (map #211, ticket #248): a state-reflecting
//  animated glyph plus a menu rebuilt fresh on every open, aligned with the
//  app's feature set — Dictation (with the language switch), Agent (talk,
//  Appshot, the Translate skill's target), Speech, and the Server (start/stop
//  plus model & cache management).
//
//  Two update paths, deliberately different:
//  - The **icon** is pushed: App Bindings feeds dictation and speech state in
//    (`updateState(from:)` / `updateState(fromSpeech:)`), and the glyph
//    animates via SF Symbol effects — GPU-cheap, static at rest per the HIG,
//    and Reduce Motion is honored by the framework.
//  - The **menu** is pulled: `menuNeedsUpdate` rebuilds every item at open
//    time from injected closures, so titles, checkmarks, sizes, and enabled
//    states are always current with zero standing subscriptions.
//

import AppKit
import Foundation
import Observation

@MainActor
final class MenuBarManager: NSObject {

    /// What the status glyph reflects. Dictation wins over speech: it is
    /// the acute, push-to-talk interaction.
    enum Activity: Equatable {
        case idle
        case listening
        case processing
        case speaking
    }

    // MARK: - Dependencies

    let settings: SettingsManager
    weak var coordinator: DictationCoordinator?
    weak var history: TranscriptionHistory?
    weak var speechCoordinator: SpeechCoordinator?

    // Window-management callbacks (attached by the app delegate).
    var onShowMainWindow: (() -> Void)?
    var onShowSettings: (() -> Void)?
    var onTalkToAgent: (() -> Void)?
    var onQuit: (() -> Void)?

    // Feature callbacks (wired by the container).
    var onTakeAppshot: (() -> Void)?
    var onOffloadModel: (() -> Void)?
    var onClearMemoryCache: (() -> Void)?
    var onClearDiskCache: (() -> Void)?

    // Live-state pulls (wired by the container), read at menu-open time.
    var serverStatus: (() -> (isRunning: Bool, port: Int))?
    var isModelLoaded: (() -> Bool)?
    var residentCacheBytes: (() -> Int?)?
    var diskCacheBytes: (() -> Int)?

    // MARK: - State

    private var statusItem: NSStatusItem?
    private var iconView: NSImageView?
    private var settingsObservationTask: Task<Void, Never>?

    private var dictationActivity: Activity = .idle
    private var speechActivity: Activity = .idle
    private var appliedActivity: Activity = .idle

    private var activity: Activity {
        dictationActivity != .idle ? dictationActivity : speechActivity
    }

    init(settings: SettingsManager) {
        self.settings = settings
    }

    // MARK: - Lifecycle

    func setupMenuBar() {
        settingsObservationTask = Task { @MainActor [weak self] in
            guard let self else { return }
            for await visible in Observations({ self.settings.showInMenuBar }) {
                self.setMenuBarVisible(visible)
            }
        }
    }

    func teardownMenuBar() {
        settingsObservationTask?.cancel()
        settingsObservationTask = nil
        removeStatusItem()
    }

    // MARK: - State pushes (App Bindings)

    func updateState(from phase: DictationFeed.Phase) {
        switch phase {
        case .idle, .error:
            dictationActivity = .idle
        case .recording:
            dictationActivity = .listening
        case .processing, .proofreading:
            dictationActivity = .processing
        }
        applyActivityToIcon()
    }

    func updateState(fromSpeech speechState: SpeechState) {
        switch speechState {
        case .idle, .capturingText, .paused, .error:
            speechActivity = .idle
        case .generating, .streaming, .streamingLongForm, .playing:
            speechActivity = .speaking
        }
        applyActivityToIcon()
    }

    // MARK: - Status item

    private func setMenuBarVisible(_ isVisible: Bool) {
        if isVisible {
            guard statusItem == nil else { return }
            createStatusItem()
        } else {
            removeStatusItem()
        }
    }

    private func removeStatusItem() {
        if let item = statusItem {
            NSStatusBar.system.removeStatusItem(item)
        }
        statusItem = nil
        iconView = nil
    }

    private func createStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem = item

        if let button = item.button {
            // The glyph lives in an embedded image view, not `button.image`:
            // `addSymbolEffect` is public on `NSImageView` only. The view is
            // click-through so the button keeps owning the menu.
            let icon = ClickThroughImageView()
            icon.translatesAutoresizingMaskIntoConstraints = false
            icon.imageScaling = .scaleNone
            button.addSubview(icon)
            NSLayoutConstraint.activate([
                icon.centerXAnchor.constraint(equalTo: button.centerXAnchor),
                icon.centerYAnchor.constraint(equalTo: button.centerYAnchor),
            ])
            iconView = icon
        }

        let menu = NSMenu()
        menu.delegate = self
        menu.autoenablesItems = false
        item.menu = menu

        appliedActivity = .idle
        setIcon(for: .idle)
    }

    // MARK: - Icon

    private func applyActivityToIcon() {
        let next = activity
        guard next != appliedActivity else { return }
        appliedActivity = next
        setIcon(for: next)
    }

    private func setIcon(for activity: Activity) {
        guard let iconView else { return }

        let symbolName: String
        let description: String
        switch activity {
        case .idle:
            symbolName = "waveform"
            description = "Tesseract Agent"
        case .listening:
            symbolName = "waveform"
            description = "Tesseract Agent — recording"
        case .processing:
            symbolName = "waveform"
            description = "Tesseract Agent — transcribing"
        case .speaking:
            symbolName = "speaker.wave.3"
            description = "Tesseract Agent — speaking"
        }

        iconView.removeAllSymbolEffects()
        let config = NSImage.SymbolConfiguration(pointSize: 16, weight: .medium)
        if let image = NSImage(
            systemSymbolName: symbolName, accessibilityDescription: description)?
            .withSymbolConfiguration(config)
        {
            image.isTemplate = true
            iconView.image = image
        }

        // Animate transient activity only; idle stays static (HIG). Symbol
        // effects honor Reduce Motion on their own.
        switch activity {
        case .idle:
            break
        case .listening, .speaking:
            iconView.addSymbolEffect(.variableColor.iterative, options: .repeating)
        case .processing:
            iconView.addSymbolEffect(.pulse, options: .repeating)
        }
    }

    // MARK: - Actions

    @objc private func toggleDictation() {
        coordinator?.toggleRecording()
    }

    @objc private func copyLastTranscription() {
        history?.copyLatestToPasteboard()
    }

    @objc private func selectDictationLanguage(_ sender: NSMenuItem) {
        guard let code = sender.representedObject as? String else { return }
        settings.language = code
        settings.recordRecentDictationLanguage(code)
    }

    @objc private func talkToAgent() {
        onTalkToAgent?()
    }

    @objc private func takeAppshot() {
        onTakeAppshot?()
    }

    @objc private func selectTranslateTarget(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String else { return }
        settings.translateTargetLanguage = name
    }

    @objc private func speakSelectedText() {
        speechCoordinator?.onHotkeyPressed()
    }

    @objc private func toggleServer() {
        // App Bindings owns the start/stop rule; the menu only flips intent.
        settings.isServerEnabled.toggle()
    }

    @objc private func offloadModel() {
        onOffloadModel?()
    }

    @objc private func clearMemoryCache() {
        onClearMemoryCache?()
    }

    @objc private func clearDiskCache() {
        let bytes = diskCacheBytes?() ?? 0
        let alert = NSAlert()
        alert.messageText = "Clear the disk cache?"
        alert.informativeText =
            "Frees \(Self.formatBytes(bytes)) of cached prompt snapshots. "
            + "The model is offloaded first, so the next request starts cold."
        alert.alertStyle = .warning
        let clear = alert.addButton(withTitle: "Clear Disk Cache")
        clear.hasDestructiveAction = true
        alert.addButton(withTitle: "Cancel")
        NSApp.activate(ignoringOtherApps: true)
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        onClearDiskCache?()
    }

    @objc private func showMainWindow() {
        onShowMainWindow?()
    }

    @objc private func showSettings() {
        onShowSettings?()
    }

    @objc private func quit() {
        onQuit?()
    }
}

// MARK: - Menu construction

extension MenuBarManager: NSMenuDelegate {

    /// The whole menu is rebuilt on every open: state is read once, here,
    /// and nothing can go stale between opens.
    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        addDictationSection(to: menu)
        addAgentSection(to: menu)
        addSpeechSection(to: menu)
        addServerSection(to: menu)
        addAppSection(to: menu)
    }

    // MARK: Sections

    private func addDictationSection(to menu: NSMenu) {
        menu.addItem(.sectionHeader(title: "Dictation"))

        let isActive = coordinator?.state.isActive ?? false
        let toggle = actionItem(
            title: isActive ? "Stop Dictation" : "Start Dictation",
            symbol: isActive ? "mic.fill" : "mic",
            action: #selector(toggleDictation),
            badge: settings.hotkey.displayString
        )
        menu.addItem(toggle)

        let language = actionItem(
            title: "Language", symbol: "globe", action: nil, badge: nil)
        language.submenu = makeLanguageSubmenu()
        menu.addItem(language)

        let copy = actionItem(
            title: "Copy Last Transcription",
            symbol: "doc.on.doc",
            action: #selector(copyLastTranscription),
            badge: nil
        )
        copy.isEnabled = !(history?.entries.isEmpty ?? true)
        menu.addItem(copy)
    }

    private func addAgentSection(to menu: NSMenu) {
        menu.addItem(.sectionHeader(title: "Agent"))

        menu.addItem(
            actionItem(
                title: "Talk to Tesseract",
                symbol: "waveform.and.mic",
                action: #selector(talkToAgent),
                badge: settings.agentHotkey.displayString
            ))
        menu.addItem(
            actionItem(
                title: "Take Appshot",
                symbol: "macwindow",
                action: #selector(takeAppshot),
                badge: settings.appshotHotkey.displayString
            ))

        let translate = actionItem(
            title: "Translate To", symbol: "translate", action: nil, badge: nil)
        translate.submenu = makeTranslateSubmenu()
        menu.addItem(translate)
    }

    private func addSpeechSection(to menu: NSMenu) {
        menu.addItem(.sectionHeader(title: "Speech"))
        menu.addItem(
            actionItem(
                title: "Speak Selected Text",
                symbol: "speaker.wave.2",
                action: #selector(speakSelectedText),
                badge: settings.ttsHotkey.displayString
            ))
    }

    private func addServerSection(to menu: NSMenu) {
        menu.addItem(.sectionHeader(title: "Server"))

        // Title follows the *intent* being toggled (`isServerEnabled`);
        // the badge reports the runtime truth. When they disagree (enabled
        // but not yet listening) the item still toggles the right thing.
        let status = serverStatus?() ?? (isRunning: false, port: 0)
        let toggle = actionItem(
            title: settings.isServerEnabled ? "Stop Server" : "Start Server",
            symbol: "power",
            action: #selector(toggleServer),
            badge: status.isRunning ? "port \(status.port)" : nil
        )
        menu.addItem(toggle)

        let management = actionItem(
            title: "Model & Cache", symbol: "memorychip", action: nil, badge: nil)
        management.submenu = makeModelAndCacheSubmenu()
        menu.addItem(management)
    }

    private func addAppSection(to menu: NSMenu) {
        menu.addItem(.separator())
        menu.addItem(
            actionItem(
                title: "Open Tesseract Agent", symbol: nil,
                action: #selector(showMainWindow), badge: nil))
        let settingsItem = actionItem(
            title: "Settings…", symbol: nil, action: #selector(showSettings), badge: nil)
        settingsItem.keyEquivalent = ","
        settingsItem.keyEquivalentModifierMask = .command
        menu.addItem(settingsItem)
        menu.addItem(.separator())
        menu.addItem(
            actionItem(
                title: "Quit Tesseract Agent", symbol: nil, action: #selector(quit), badge: nil))
    }

    // MARK: Submenus

    /// Auto-detect, the pinned working set (current selection, recents,
    /// system-preferred), then the full catalogue one level deeper — 99
    /// languages never flood the first level.
    private func makeLanguageSubmenu() -> NSMenu {
        let submenu = NSMenu()
        submenu.autoenablesItems = false
        let current = settings.language

        let auto = languageItem(SupportedLanguage.auto, current: current)
        submenu.addItem(auto)
        submenu.addItem(.separator())

        let pinnedCodes = MenuBarLanguagePins.pinnedCodes(
            current: current,
            recents: settings.recentDictationLanguages
                .split(separator: ",").map(String.init),
            preferredLanguageIdentifiers: Locale.preferredLanguages
        )
        for code in pinnedCodes {
            guard let language = SupportedLanguage.language(forCode: code) else { continue }
            submenu.addItem(languageItem(language, current: current))
        }
        if !pinnedCodes.isEmpty {
            submenu.addItem(.separator())
        }

        let more = NSMenuItem(title: "All Languages", action: nil, keyEquivalent: "")
        let all = NSMenu()
        all.autoenablesItems = false
        for language in SupportedLanguage.all where language.code != SupportedLanguage.auto.code {
            all.addItem(languageItem(language, current: current))
        }
        more.submenu = all
        submenu.addItem(more)
        return submenu
    }

    private func languageItem(_ language: SupportedLanguage, current: String) -> NSMenuItem {
        let item = NSMenuItem(
            title: language.displayName,
            action: #selector(selectDictationLanguage),
            keyEquivalent: ""
        )
        item.target = self
        item.representedObject = language.code
        item.state = language.code == current ? .on : .off
        return item
    }

    /// The Translate skill's default target — same option list as the Agent
    /// settings pane, current selection and system-preferred names pinned,
    /// the rest one level deeper.
    private func makeTranslateSubmenu() -> NSMenu {
        let submenu = NSMenu()
        submenu.autoenablesItems = false
        let current = settings.translateTargetLanguage
        let options = SupportedLanguage.translateTargetOptions(current: current)

        let pinnedNames = MenuBarLanguagePins.pinnedTranslateNames(
            current: current,
            preferredLanguageIdentifiers: Locale.preferredLanguages,
            options: options
        )
        for name in pinnedNames {
            submenu.addItem(translateItem(name, current: current))
        }
        submenu.addItem(.separator())

        let more = NSMenuItem(title: "All Languages", action: nil, keyEquivalent: "")
        let all = NSMenu()
        all.autoenablesItems = false
        for name in options {
            all.addItem(translateItem(name, current: current))
        }
        more.submenu = all
        submenu.addItem(more)
        return submenu
    }

    private func translateItem(_ name: String, current: String) -> NSMenuItem {
        let item = NSMenuItem(
            title: name,
            action: #selector(selectTranslateTarget),
            keyEquivalent: ""
        )
        item.target = self
        item.representedObject = name
        item.state = name == current ? .on : .off
        return item
    }

    private func makeModelAndCacheSubmenu() -> NSMenu {
        let submenu = NSMenu()
        submenu.autoenablesItems = false

        let offload = actionItem(
            title: "Offload Model",
            symbol: "memorychip",
            action: #selector(offloadModel),
            badge: nil
        )
        offload.isEnabled = isModelLoaded?() ?? false
        submenu.addItem(offload)
        submenu.addItem(.separator())

        let ramBytes = residentCacheBytes.flatMap { $0() }
        let clearRAM = actionItem(
            title: "Clear Memory Cache",
            symbol: "bolt.horizontal",
            action: #selector(clearMemoryCache),
            badge: ramBytes.map(Self.formatBytes)
        )
        clearRAM.isEnabled = (ramBytes ?? 0) > 0
        submenu.addItem(clearRAM)

        let diskBytes = diskCacheBytes?() ?? 0
        let clearDisk = actionItem(
            title: "Clear Disk Cache",
            symbol: "internaldrive",
            action: #selector(clearDiskCache),
            badge: diskBytes > 0 ? Self.formatBytes(diskBytes) : nil
        )
        clearDisk.isEnabled = diskBytes > 0
        submenu.addItem(clearDisk)
        return submenu
    }

    // MARK: Item helpers

    private func actionItem(
        title: String,
        symbol: String?,
        action: Selector?,
        badge: String?
    ) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
        item.target = action == nil ? nil : self
        if let symbol {
            item.image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)
        }
        if let badge {
            item.badge = NSMenuItemBadge(string: badge)
        }
        return item
    }

    private static func formatBytes(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}

// MARK: - Click-through image view

/// Hosts the animated status glyph without stealing the status button's
/// clicks — the button keeps owning menu presentation.
private final class ClickThroughImageView: NSImageView {
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}

// MARK: - Pinned-language derivation

/// Pure derivations for the menu's pinned language entries — kept
/// nonisolated and value-in/value-out so they are directly testable.
nonisolated enum MenuBarLanguagePins {

    /// Dictation pins: the current selection first, then recents (newest
    /// first), then the system's preferred languages — deduplicated,
    /// filtered to the Whisper catalogue, `auto` excluded (it has a
    /// permanent slot), capped at six.
    static func pinnedCodes(
        current: String,
        recents: [String],
        preferredLanguageIdentifiers: [String]
    ) -> [String] {
        let preferred = preferredLanguageIdentifiers.compactMap {
            Locale(identifier: $0).language.languageCode?.identifier
        }
        var seen: Set<String> = []
        var pins: [String] = []
        for code in [current] + recents + preferred {
            guard code != SupportedLanguage.auto.code,
                SupportedLanguage.language(forCode: code) != nil,
                seen.insert(code).inserted
            else { continue }
            pins.append(code)
            if pins.count == 6 { break }
        }
        return pins
    }

    /// Translate pins: the current target first, then the system's
    /// preferred languages plus English, as display names filtered to the
    /// available option list, capped at six. (The Translate skill stores
    /// display names, not codes — PRD #174.)
    static func pinnedTranslateNames(
        current: String,
        preferredLanguageIdentifiers: [String],
        options: [String]
    ) -> [String] {
        let preferredNames = preferredLanguageIdentifiers.compactMap { identifier -> String? in
            guard let code = Locale(identifier: identifier).language.languageCode?.identifier
            else { return nil }
            return SupportedLanguage.language(forCode: code)?.name
        }
        var seen: Set<String> = []
        var pins: [String] = []
        for name in [current] + preferredNames + ["English"] {
            guard options.contains(name), seen.insert(name).inserted else { continue }
            pins.append(name)
            if pins.count == 6 { break }
        }
        return pins
    }
}
