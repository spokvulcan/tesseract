# WhisperOnDevice Distribution Checklist

This document tracks what needs to be done to prepare the application for distribution.

---

## Current State

| Setting | Value |
|---------|-------|
| Bundle Identifier | `tesseract.whisper-on-device` |
| Version | 1.0 |
| Build | 1 |
| Development Team | Configured |
| Code Signing | Automatic (Apple Development) |
| Minimum macOS | 26 (Tahoe) |
| Entitlements | App Sandbox, Audio Input (Microphone) |
| Privacy Usage Strings | NSMicrophoneUsageDescription |

---

## Entitlements and Privacy Notes

- The app does not currently use Apple Events, so no Apple Events usage string or automation entitlements are included.
- The app does not use outbound networking or user-selected file access. If those are added later, update entitlements accordingly.

---

## Already Done

- [x] Bundle identifier set
- [x] Development team configured
- [x] Entitlements configured (sandbox, microphone)
- [x] Privacy description set (microphone)
- [x] Version/build numbers set
- [x] Whisper model bundled
- [x] App icon added (all required sizes)
- [x] App category set (`public.app-category.productivity`)

---

## Needs to Be Done

### Required for Mac App Store

#### 1. Code Signing for App Store

Current signing identity is "Apple Development" which only works for local testing.

- Create an **Apple Distribution** certificate
- Create a **Mac App Store** provisioning profile
- Update target signing to use Apple Distribution (Automatic is OK once the cert/profile exist)

#### 2. App Store Connect Setup

- Create the app record (bundle ID, SKU, name, primary language)
- Configure pricing, availability, and age rating
- Complete App Privacy (data collection/usage) and Export Compliance

#### 3. App Store Metadata and Assets

- Fill in description, subtitle, keywords, and promotional text
- Provide support URL and privacy policy URL
- Add screenshots and app preview if desired

Draft copy lives in `APP_STORE_METADATA.md`.

#### 4. Archive and Upload

- Archive with **Release** configuration
- Upload from Xcode Organizer to App Store Connect
- Confirm the build appears in App Store Connect

#### 5. TestFlight (Recommended)

- Run an internal TestFlight build to verify entitlements, permissions, and onboarding

#### 6. App Review Submission

- Provide review notes (permissions + hotkey flow)
- Submit once metadata and build are complete

#### 7. Pre-Submission Testing

- [ ] Test on a Mac without Xcode installed
- [ ] Verify Accessibility permission prompt appears
- [ ] Verify Microphone permission prompt appears
- [ ] Verify model loads correctly from bundle
- [ ] Test push-to-talk recording flow
- [ ] Test text injection into various apps (Notes, TextEdit, browser)
- [ ] Test menu bar functionality
- [ ] Test settings persistence

### Direct Distribution (Optional)

#### 1. Developer ID Signing

- Create "Developer ID Application" certificate
- Requires paid Apple Developer Program

#### 2. Notarization

Required for direct distribution. Apple must notarize the app for it to run on other Macs without Gatekeeper warnings.

```bash
# 1. Archive the app
xcodebuild archive \
  -project whisper-on-device.xcodeproj \
  -scheme whisper-on-device \
  -archivePath build/WhisperOnDevice.xcarchive

# 2. Export the archive
xcodebuild -exportArchive \
  -archivePath build/WhisperOnDevice.xcarchive \
  -exportPath build/export \
  -exportOptionsPlist ExportOptions.plist

# 3. Notarize
xcrun notarytool submit build/export/WhisperOnDevice.app.zip \
  --apple-id YOUR_APPLE_ID \
  --team-id YOUR_TEAM_ID \
  --password APP_SPECIFIC_PASSWORD \
  --wait

# 4. Staple the notarization ticket
xcrun stapler staple build/export/WhisperOnDevice.app
```

#### 3. Create Distribution Package

**DMG (recommended for direct distribution):**
```bash
# Using create-dmg (brew install create-dmg)
create-dmg \
  --volname "WhisperOnDevice" \
  --volicon "path/to/icon.icns" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "WhisperOnDevice.app" 150 190 \
  --app-drop-link 450 190 \
  "WhisperOnDevice-1.0.dmg" \
  "build/export/"
```

---

### Optional / Nice-to-Have

#### App Display Name

Current product name: `whisper-on-device`

Consider changing to: `WhisperOnDevice` or `Whisper`

Set in Xcode: Target → General → Display Name

#### Hardened Runtime

Already enabled via sandbox, but verify these options:
- Allow Unsigned Executable Memory: NO
- Allow DYLD Environment Variables: NO
- Disable Library Validation: NO (unless needed for plugins)

---

## Distribution Channels

### Option A: Direct Distribution (Recommended for MVP)

1. Sign with Developer ID
2. Notarize with Apple
3. Distribute via website/GitHub Releases as DMG

**Pros:** Full control, no review process, can update anytime
**Cons:** Users see "downloaded from internet" warning on first launch

### Option B: Mac App Store

1. Sign with Apple Distribution certificate
2. Upload build to App Store Connect
3. Submit for App Store review

**Pros:** Trusted source, automatic updates, wider reach
**Cons:** Review delays, revenue share (15-30%), stricter sandboxing

---

## Version Checklist (Before Each Release)

- [ ] Update `MARKETING_VERSION` (e.g., 1.0 → 1.1)
- [ ] Update `CURRENT_PROJECT_VERSION` (increment build number)
- [ ] Test all features
- [ ] Archive with Release configuration
- [ ] Upload build to App Store Connect
- [ ] Submit for review

**Direct distribution only:**
- [ ] Notarize
- [ ] Create DMG
- [ ] Test DMG installation on clean Mac

---

## Resources

- [Apple Developer Program](https://developer.apple.com/programs/)
- [Notarizing macOS Software](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Creating a DMG](https://github.com/create-dmg/create-dmg)
- [App Sandbox Documentation](https://developer.apple.com/documentation/security/app_sandbox)
