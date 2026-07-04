# Changelog

## [1.1.0](https://github.com/spokvulcan/tesseract/compare/v1.0.0...v1.1.0) (2026-07-04)


### Features

* **server:** Leaf Home Guarantee — the newest turn is never lost ([#148](https://github.com/spokvulcan/tesseract/issues/148)) ([7d68d13](https://github.com/spokvulcan/tesseract/commit/7d68d13cde41a88be075c9416f34c9f5adda4456))
* **server:** Prompt Cache telemetry face — hero band, full-bleed tree, events drawer ([adb3b9d](https://github.com/spokvulcan/tesseract/commit/adb3b9d08eb66c934cfd39bf47b08465650f557c))
* **server:** stream tool-call arguments incrementally (Argument Transcoder) ([#154](https://github.com/spokvulcan/tesseract/issues/154)) ([f809854](https://github.com/spokvulcan/tesseract/commit/f80985468b58a9e9739aa8a34599686217152405))
* **server:** telemetry dashboard with console drawer, in-app cancel, live metrics ([9c4d065](https://github.com/spokvulcan/tesseract/commit/9c4d06535909485e854b846c6d77bf47ca595fa4))


### Bug Fixes

* **server:** prompt-cache tree filters contract the tree; Empty hidden by default ([fa405e2](https://github.com/spokvulcan/tesseract/commit/fa405e21899ead6768d5add3d53a9d75a19c6a39))


### Code Refactoring

* **agent:** one commit step for tool results + loop-level integration suite ([#142](https://github.com/spokvulcan/tesseract/issues/142)) ([dd176b9](https://github.com/spokvulcan/tesseract/commit/dd176b901f42860c3e1b7d428cea67e26d9cf90d))
* **models:** one download lifecycle, Model Fetching seam, truthful cancel ([#146](https://github.com/spokvulcan/tesseract/issues/146)) ([38f5de3](https://github.com/spokvulcan/tesseract/commit/38f5de37a3c4695768732f47d4952c8e43837141))
* **server:** drop status/endpoint widget from dashboard toolbar ([90b60fc](https://github.com/spokvulcan/tesseract/commit/90b60fcd9b98039cf89514c06be47abe9982889e))
* **speech:** one owned notch teardown, no polling; hermetic Word Tracker tests ([b301bca](https://github.com/spokvulcan/tesseract/commit/b301bca03159eccc7b3464fad3991605141e8a6c)), closes [#140](https://github.com/spokvulcan/tesseract/issues/140)


### Documentation

* **adr:** dynamic budget ceilings + recoverable eviction (prefix-cache grilling) ([9e6857e](https://github.com/spokvulcan/tesseract/commit/9e6857e27dfb1fcceeea3f8680f3130446ab0a68))
* **readme:** redesign README as a compact marketing page ([2e7f835](https://github.com/spokvulcan/tesseract/commit/2e7f83529e0c643dcd1d1864e552482808001f0a))


### Continuous Integration

* **release:** drop the release-as pin and cut redundant per-release CI runs ([#145](https://github.com/spokvulcan/tesseract/issues/145)) ([3a3070e](https://github.com/spokvulcan/tesseract/commit/3a3070eb0a2529d234ff18dc89131b3b89ab7eed))
* **release:** make every conventional commit type release-worthy ([0d51f68](https://github.com/spokvulcan/tesseract/commit/0d51f6894a7e0cdc16e702f7d9f1f11c1062dce5))

## 1.0.0 (2026-07-03)


### Features

* **release:** add Release Please pipeline with signed, notarized DMG builds ([#143](https://github.com/spokvulcan/tesseract/issues/143)) ([6b86e47](https://github.com/spokvulcan/tesseract/commit/6b86e4783370120d5d86c3adc628ddb19da59080))
