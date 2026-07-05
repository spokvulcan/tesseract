# Changelog

## [1.2.0](https://github.com/spokvulcan/tesseract/compare/v1.1.0...v1.2.0) (2026-07-05)


### Features

* **agent:** Appshots — double-⌘ frontmost-window capture into the composer ([#172](https://github.com/spokvulcan/tesseract/issues/172)) ([bc30e61](https://github.com/spokvulcan/tesseract/commit/bc30e612e0caebccc800e21abdd9d18a7e74b2fa))
* **agent:** calm progress banner while the selected model downloads ([b7c1cd1](https://github.com/spokvulcan/tesseract/commit/b7c1cd1d1edb5efc4651775b9cb1c364a994ac6c))
* **onboarding:** replace the setup sheet with the six-chapter Welcome Tour ([7c163b3](https://github.com/spokvulcan/tesseract/commit/7c163b3db82fe7856043a3fada616de313481771))
* **server:** cache-miss attribution + multi-process-safe telemetry writers ([#160](https://github.com/spokvulcan/tesseract/issues/160)) ([ad87ac0](https://github.com/spokvulcan/tesseract/commit/ad87ac0af4633ea07502c64ac5eb997e85f7fb61))
* **server:** dynamic budget ceilings + uniform eviction (PRD [#149](https://github.com/spokvulcan/tesseract/issues/149)) ([#155](https://github.com/spokvulcan/tesseract/issues/155)) ([c2357ee](https://github.com/spokvulcan/tesseract/commit/c2357ee5925ddb2c5dbd43aaff0ca41f21662632))
* **server:** SSD tier of record — endurance counters, adaptive write eagerness, stale-partition GC, cache panel v1 (PRD [#150](https://github.com/spokvulcan/tesseract/issues/150)) ([#157](https://github.com/spokvulcan/tesseract/issues/157)) ([0429f4b](https://github.com/spokvulcan/tesseract/commit/0429f4b35db448176ef57bca42ef791ed1fad4fc))


### Bug Fixes

* **agent:** unify chat and composer text at 16pt across both render modes ([838c960](https://github.com/spokvulcan/tesseract/commit/838c96046207cf0aff70e3a3b664a42334571f22))
* composer image gestures + dictation into our own composer ([#169](https://github.com/spokvulcan/tesseract/issues/169)) ([e969156](https://github.com/spokvulcan/tesseract/commit/e969156ca73552ff034e347c660927e7f5f8fd23))
* **deps:** revert swift-readability to d4f0824 — db890d6 breaks extraction ([08f4638](https://github.com/spokvulcan/tesseract/commit/08f463802eec849decd595d59d459c3161dda13e))
* **onboarding:** replace neon tour visuals with Apple-native treatment ([473d162](https://github.com/spokvulcan/tesseract/commit/473d162e66621cc3e52a45c5207dcf4a99103d38))
* **scripts:** resolve the built app via BUILT_PRODUCTS_DIR ([c3f705c](https://github.com/spokvulcan/tesseract/commit/c3f705c4634664cbcaab2f0e5185215edd0f3a17))
* **server:** never defer the .system stable-prefix SSD write ([02609cb](https://github.com/spokvulcan/tesseract/commit/02609cb1c6902785e0cd4db0322a6555a2de367e))


### Code Refactoring

* **onboarding:** apply /simplify cleanup pass ([1302c49](https://github.com/spokvulcan/tesseract/commit/1302c49aed8be13807ebda36b51720406d9850b6))


### Documentation

* **context:** add Appshot glossary term ([742fdba](https://github.com/spokvulcan/tesseract/commit/742fdbabe863297dced98b9f00f54cf4c970b9ba))
* **domain:** add Batch inference glossary terms and ADR-0022/0023 ([0fd372b](https://github.com/spokvulcan/tesseract/commit/0fd372b37caf2e544a8fb84f33d30318906c46b2))
* **domain:** add Onboarding tour glossary terms and ADR-0021 ([4b8095a](https://github.com/spokvulcan/tesseract/commit/4b8095a811b27aa3b97f41d1db5e8895cd966c13))


### Miscellaneous Chores

* **deps:** bump textual to 0.5.0, swift-readability to db890d6 ([8ee2192](https://github.com/spokvulcan/tesseract/commit/8ee2192e621ae6ac336f06f719b34b281fddf507)), closes [#161](https://github.com/spokvulcan/tesseract/issues/161) [#162](https://github.com/spokvulcan/tesseract/issues/162)

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
