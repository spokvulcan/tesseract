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

---

## Already Done

- [x] Bundle identifier set
- [x] Development team configured
- [x] Entitlements configured (sandbox, microphone, network, file access)
- [x] Privacy descriptions set (microphone, Apple Events)
- [x] Version/build numbers set
- [x] Whisper model bundled

---

## Needs to Be Done

### Required

#### 1. App Icon
The icon asset catalog is empty - no images exist.

**Required sizes (1x and 2x for each):**
- 16x16
- 32x32
- 128x128
- 256x256
- 512x512

**Total: 10 PNG files**

Location: `whisper-on-device/Assets.xcassets/AppIcon.appiconset/`

#### 2. Code Signing for Distribution

Current signing identity is "Apple Development" which only works for local testing.

**For direct distribution (outside App Store):**
- Need "Developer ID Application" certificate
- Requires paid Apple Developer Program ($99/year)
- Generate at: https://developer.apple.com/account/resources/certificates

**For Mac App Store:**
- Need "Apple Distribution" certificate
- Subject to App Store review process
- More restrictive sandboxing requirements

#### 3. Notarization

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

#### 4. Create Distribution Package

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

#### 5. Testing on Clean Machine

- [ ] Test on a Mac without Xcode installed
- [ ] Verify Accessibility permission prompt appears
- [ ] Verify Microphone permission prompt appears
- [ ] Verify model loads correctly from bundle
- [ ] Test push-to-talk recording flow
- [ ] Test text injection into various apps (Notes, TextEdit, browser)
- [ ] Test menu bar functionality
- [ ] Test settings persistence

---

### Optional / Nice-to-Have

#### App Display Name
Current product name: `whisper-on-device`

Consider changing to: `WhisperOnDevice` or `Whisper`

Set in Xcode: Target → General → Display Name

#### App Category

Set `LSApplicationCategoryType` in build settings for Finder/App Store categorization.

Suggested values:
- `public.app-category.productivity`
- `public.app-category.utilities`

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
2. Submit for App Store review
3. Distribute via Mac App Store

**Pros:** Trusted source, automatic updates, wider reach
**Cons:** Review delays, revenue share (15-30%), stricter sandboxing

---

## Version Checklist (Before Each Release)

- [ ] Update `MARKETING_VERSION` (e.g., 1.0 → 1.1)
- [ ] Update `CURRENT_PROJECT_VERSION` (increment build number)
- [ ] Test all features
- [ ] Archive with Release configuration
- [ ] Notarize
- [ ] Create DMG
- [ ] Test DMG installation on clean Mac
- [ ] Update release notes
- [ ] Tag release in git

---

## Resources

- [Apple Developer Program](https://developer.apple.com/programs/)
- [Notarizing macOS Software](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Creating a DMG](https://github.com/create-dmg/create-dmg)
- [App Sandbox Documentation](https://developer.apple.com/documentation/security/app_sandbox)
