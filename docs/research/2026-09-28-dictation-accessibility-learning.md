# Learning from edits made after insertion: Accessibility, and what to use instead

**Date:** 2026-09-28
**Question:** After Tesseract pastes a take, should it watch the target app's text through the macOS Accessibility API and learn when the owner fixes a word in place? What does that cost in privacy, reliability and performance, and what are the alternatives?

**Method.** Apple SDK headers and docs; source code of WebKit, Chromium, Electron, VS Code, iTerm2, Ghostty, Warp, kitty and the JetBrains Runtime; product docs and privacy pages; issue trackers. Seven probes ran on this Mac (macOS 27.0) against test windows the probes created, a throwaway Chrome profile, and, read-only and without printing any text, the owner's running apps. The probe code and the full notes stay outside the repository.

---

## Verdict

Don't make watching the main way the app learns. Where the owner dictates most, it can't see anything; where it can see, it reads far more than the dictated words; and many edits are rewrites, not corrections. Keep it as an opt-in extra for apps that expose their text well, producing candidates rather than lessons, and never in terminals or secure fields. The fix shortcut, plus a Claude Code hook, gives more and cleaner signal for this owner.

## Where it can see the text

| Target | What Accessibility gives | Verdict |
|---|---|---|
| Claude Code in a terminal (Ghostty, Terminal, iTerm2) | The whole scrollback as one read-only string; Ghostty reports no caret and posts no change notifications ([Ghostty source](https://github.com/ghostty-org/ghostty/blob/main/macos/Sources/Ghostty/Surface%20View/SurfaceView_AppKit.swift)). Terminal.app's value was measured at 3 million characters and 30 ms per read ([Ghostty issue 9932](https://github.com/ghostty-org/ghostty/issues/9932)). Long pastes become a `[Pasted text]` chip ([Claude Code docs](https://code.claude.com/docs/en/terminal-config#paste-large-content)) | No |
| Claude desktop, Slack, VS Code (Electron), Chrome | Nothing until a client asks; asking switches the whole app into screen-reader mode until it quits ([Electron source](https://github.com/electron/electron/blob/main/shell/browser/mac/electron_application.mm)), and flips VS Code into Screen Reader Optimized mode ([VS Code issue 279644](https://github.com/microsoft/vscode/issues/279644)). In a probe Chrome grew about 165 MB | Only if the owner opts an app in |
| Safari, Mail, WKWebView apps | The removed and inserted strings arrive in the change notification (a probe saw `cloud.md` deleted and `CLAUDE.md` typed as one event) ([WebKit](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/accessibility/mac/AXObjectCacheMac.mm)) | Yes, the best target |
| Native AppKit fields (TextEdit, Notes, Messages) | The caret before and after the paste; changes arrive without details and have to be diffed | Yes |
| Google Docs, Zed, Warp, kitty | No usable text | No |
| Password fields | `AXSecureTextField`; WebKit still leaks the length | Never read |

Wispr Flow, which ships edit watching, was measured by a competitor catching about 87% of edits in TextEdit, 69% in Chrome and 29% in a terminal (anecdotal, [EnviousWispr issue 996](https://github.com/saurabhav88/EnviousWispr/issues/996)); another project got it working in 9 of 13 apps ([PR 3054](https://github.com/saurabhav88/EnviousWispr/pull/3054)).

## Precision

Every shipping learner ended up filtering: one contiguous change, a few words, close in sound or spelling, no case-only changes, no common words, often a judge model, and an undo. They still learned words like "what" and "Pattern" ([OpenWhispr PR 2178](https://github.com/OpenWhispr/openwhispr/pull/2178), [VoiceInk issue 637](https://github.com/Beingpax/VoiceInk/issues/637)). Nuance warns that teaching Dragon through content edits makes recognition worse ([Dragon for Mac help](https://www.nuance.com/products/help/dragon/dragon-for-mac/enx/Content/Correction/AboutCorrection.htm)).

## Privacy and cost

- The Accessibility grant Tesseract already holds covers all of it, so the owner sees no new prompt; the same grant reads any text in any app ([Apple Support](https://support.apple.com/guide/mac-help/allow-accessibility-apps-to-access-your-mac-mh43185/mac)). A watcher sees the whole field and, when a message is sent, its full text.
- Reads take tens of microseconds, but a hung app blocks each call for 1.5 s by default, so reads belong off the main thread with short timeouts ([AXUIElementSetMessagingTimeout](https://developer.apple.com/documentation/applicationservices/1459345-axuielementsetmessagingtimeout)).
- Never set `AXEnhancedUserInterface`: it breaks window managers and blurred the Claude desktop composer in another dictation app ([OpenWhispr PR 1116](https://github.com/OpenWhispr/openwhispr/pull/1116)).
- The norms the careful products share: off by default, an allowlist of apps, skip secure fields, a visible sign while watching, keep only the learned pair, and an undo for every lesson.

## Alternatives

- **A fix shortcut for the last take.** Tesseract knows what it inserted, so it can select it again (through Accessibility where the field is editable) or, in a terminal, backspace over it and type the fix, as long as nothing was typed since. The fix is an explicit owner signal.
- **Select, then shortcut.** Reads only the selection, only when asked. Weak in Claude Code, whose input has no keyboard selection.
- **A Claude Code `UserPromptSubmit` hook.** It receives the prompt as submitted, with pasted text expanded ([hooks docs](https://code.claude.com/docs/en/hooks)); Tesseract can diff it against what it inserted. It works in every terminal, installing it is the consent, and one dictation project chose it over reading the terminal ([localvoxtral issue 520](https://github.com/T0mSIlver/localvoxtral/issues/520)). The submitted prompt mixes fixes with rewrites, so the same filters apply.
- **Saying it again** is weak alone: respoken corrections are misrecognized more often than first attempts, 44% against 16% ([Levow 1998](https://aclanthology.org/P98-1122/)). An explicit "retry the last take" makes the intent clear.
