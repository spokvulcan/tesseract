# Tesseract Agent Distribution Checklist

Tracks what needs to be done to publish Tesseract Agent on the Mac App Store.

---

## Current State

| Setting | Value |
|---------|-------|
| App Name (display) | Tesseract Agent (`CFBundleDisplayName`) |
| Bundle Identifier | `app.tesseract.agent` |
| Version | 1.0 |
| Build | 1 |
| Development Team | Configured (`5RBTC2MNY8`) |
| Code Signing | Automatic (Apple Development) |
| Minimum macOS | 26 (Tahoe) |
| Category | Productivity |

### Entitlements (per-configuration)

| Configuration | File | Extra |
|---|---|---|
| Debug | `tesseract/tesseract.entitlements` | `/private/tmp/tesseract-debug/` write access |
| Release | `tesseract/tesseractRelease.entitlements` | Clean — no temporary exceptions |

Both share:
- `com.apple.security.app-sandbox` — sandboxed
- `com.apple.security.device.audio-input` — microphone access
- `com.apple.security.network.client` — outbound network for model downloads

### Privacy Manifest (`tesseract/PrivacyInfo.xcprivacy`)

- `NSPrivacyTracking` = false
- `NSPrivacyCollectedDataTypes` = none (all processing is on-device)
- `NSPrivacyAccessedAPITypes`:
  - `NSPrivacyAccessedAPICategoryUserDefaults` — reason `CA92.1` (app-own `@AppStorage` preferences)

---

## Already Done

- [x] Bundle identifier set
- [x] Development team configured
- [x] Entitlements configured (sandbox, microphone, network)
- [x] Separate Release entitlements (no debug exceptions)
- [x] Privacy manifest (`PrivacyInfo.xcprivacy`)
- [x] Privacy usage description (`NSMicrophoneUsageDescription`)
- [x] `CFBundleDisplayName` set to "Tesseract Agent" (Debug + Release)
- [x] App icon added (all required macOS sizes)
- [x] App category set (`public.app-category.productivity`)
- [x] Version/build numbers set (1.0 / 1)
- [x] App Store metadata drafted (`APP_STORE_METADATA.md`)
- [x] All user-facing strings updated to "Tesseract Agent"
- [x] Onboarding wizard updated for current feature set

---

## Remaining Steps

### 1. Code Signing for App Store

Current signing identity is "Apple Development" (local testing only).

- [ ] Create an **Apple Distribution** certificate at developer.apple.com
- [ ] Create a **Mac App Store** provisioning profile
- [ ] Xcode Automatic signing will pick these up once they exist

### 2. App Store Connect Setup

- [ ] Create the app record (bundle ID `app.tesseract.agent`, SKU, name "Tesseract Agent")
- [ ] Configure pricing and availability
- [ ] Complete age rating questionnaire
- [ ] Complete App Privacy questionnaire (no data collected — strong story)
- [ ] Complete Export Compliance (no non-standard encryption)

### 3. Metadata and Assets

Draft copy is in `APP_STORE_METADATA.md`. Still needed:

- [ ] Support URL
- [ ] Privacy Policy URL
- [ ] Marketing URL (optional)
- [ ] Screenshots (at least one set)
- [ ] App preview video (optional)

### 4. Archive and Upload

```bash
# Archive with Release configuration
xcodebuild archive \
  -project tesseract.xcodeproj \
  -scheme tesseract \
  -archivePath build/Tesseract Agent.xcarchive

# Or use Xcode: Product → Archive → Distribute App → App Store Connect
```

- [ ] Archive with Release configuration
- [ ] Upload from Xcode Organizer to App Store Connect
- [ ] Confirm the build appears in App Store Connect

### 5. TestFlight (Recommended)

- [ ] Run an internal TestFlight build on a clean Mac (no Xcode)
- [ ] Verify all permission prompts appear (Microphone, Accessibility)
- [ ] Verify model downloads work from sandbox
- [ ] Verify push-to-talk dictation works
- [ ] Verify TTS playback works
- [ ] Verify text injection into various apps (Notes, TextEdit, browser)
- [ ] Verify menu bar icon and quit menu
- [ ] Verify settings persistence across launches

### 6. App Review Submission

- [ ] Add review notes explaining permissions and hotkey flow (draft in `APP_STORE_METADATA.md`)
- [ ] Submit for review

---

## Direct Distribution (Alternative)

If App Store review is problematic (Accessibility permission usage can get scrutiny), distribute directly:

### 1. Developer ID Signing

- Create "Developer ID Application" certificate (requires paid Apple Developer Program)

### 2. Notarization

```bash
# 1. Archive
xcodebuild archive \
  -project tesseract.xcodeproj \
  -scheme tesseract \
  -archivePath build/Tesseract Agent.xcarchive

# 2. Export
xcodebuild -exportArchive \
  -archivePath build/Tesseract Agent.xcarchive \
  -exportPath build/export \
  -exportOptionsPlist ExportOptions.plist

# 3. Notarize
xcrun notarytool submit build/export/Tesseract Agent.app.zip \
  --apple-id YOUR_APPLE_ID \
  --team-id 5RBTC2MNY8 \
  --password APP_SPECIFIC_PASSWORD \
  --wait

# 4. Staple
xcrun stapler staple build/export/Tesseract Agent.app
```

### 3. Create DMG

```bash
# Using create-dmg (brew install create-dmg)
create-dmg \
  --volname "Tesseract Agent" \
  --volicon "path/to/icon.icns" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "Tesseract Agent.app" 150 190 \
  --app-drop-link 450 190 \
  "Tesseract Agent-1.0.dmg" \
  "build/export/"
```

---

## Version Checklist (Before Each Release)

- [ ] Update `MARKETING_VERSION` in Xcode (e.g., 1.0 → 1.1)
- [ ] Increment `CURRENT_PROJECT_VERSION` (build number)
- [ ] Test all features end-to-end
- [ ] Archive with Release configuration
- [ ] Upload to App Store Connect
- [ ] Submit for review

**Direct distribution only:**
- [ ] Notarize
- [ ] Create DMG
- [ ] Test DMG installation on clean Mac

---

## Resources

- [Apple Developer Program](https://developer.apple.com/programs/)
- [App Store Product Page](https://developer.apple.com/app-store/product-page/)
- [Notarizing macOS Software](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [App Sandbox Documentation](https://developer.apple.com/documentation/security/app_sandbox)
- [Privacy Manifest Files](https://developer.apple.com/documentation/bundleresources/privacy_manifest_files)
- [Creating a DMG](https://github.com/create-dmg/create-dmg)
