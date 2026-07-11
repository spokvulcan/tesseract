# Changelog

## [1.7.0](https://github.com/spokvulcan/tesseract/compare/v1.6.0...v1.7.0) (2026-07-11)


### Features

* **agent:** vendor MoE PARO path — RotateSwitchGLU + loader passes ([#220](https://github.com/spokvulcan/tesseract/issues/220)) ([#225](https://github.com/spokvulcan/tesseract/issues/225)) ([ad3c798](https://github.com/spokvulcan/tesseract/commit/ad3c798f72f42fee84bba4da7030e2ea60e5f603))
* **agent:** wire the app for Qwen3.6-35B-A3B PARO (MoE) ([#210](https://github.com/spokvulcan/tesseract/issues/210)) ([#226](https://github.com/spokvulcan/tesseract/issues/226)) ([6c168eb](https://github.com/spokvulcan/tesseract/commit/6c168eb82640f796e120a468946ec8940f4b15e3))
* **bench:** add PARO reference-parity benchmark harness ([#217](https://github.com/spokvulcan/tesseract/issues/217)) ([e01bc8d](https://github.com/spokvulcan/tesseract/commit/e01bc8da7931b04672faa3d9adf7cef400b5ce99))
* **browser:** local-only tool telemetry for the Browser MCP server (ADR-0031) ([#239](https://github.com/spokvulcan/tesseract/issues/239)) ([a9ed929](https://github.com/spokvulcan/tesseract/commit/a9ed929775c197f66eade5a9c7ecfa6373451e29))
* **browser:** save screenshot pixel artifacts in MCP telemetry ([#240](https://github.com/spokvulcan/tesseract/issues/240)) ([ae07d98](https://github.com/spokvulcan/tesseract/commit/ae07d98cb7cc1e9a9f5d52edf3bbc9f190de15b3))
* **design:** adopt the ratified warm-orange AccentColor ([ed67f0c](https://github.com/spokvulcan/tesseract/commit/ed67f0c13740a451a3fa7572b1219892da3d68f7))
* **dictation:** face-lift the Dictation page to the design language ([#250](https://github.com/spokvulcan/tesseract/issues/250)) ([82a1f89](https://github.com/spokvulcan/tesseract/commit/82a1f8954c6c62f497773740a8073db3745b90c5))
* **menubar:** rewrite the status-bar surface — animated glyph, feature-aligned menu ([#266](https://github.com/spokvulcan/tesseract/issues/266)) ([de1f2f5](https://github.com/spokvulcan/tesseract/commit/de1f2f56df6edacc5da9fb778cced5d9f87d1f40))
* **models:** redesign the Models page — native grouped form + Liquid Glass action bar ([#264](https://github.com/spokvulcan/tesseract/issues/264)) ([eeb69eb](https://github.com/spokvulcan/tesseract/commit/eeb69ebd1c2fef3cf468b6487e3cc21aac20b07f))
* **onboarding:** face-lift the Welcome Tour to the design language ([#265](https://github.com/spokvulcan/tesseract/issues/265)) ([920b373](https://github.com/spokvulcan/tesseract/commit/920b3736caef36c8025d61669a73fe33654491e0))
* **onboarding:** recommend Qwen3.6-35B-A3B PARO on 48GB+ machines ([#228](https://github.com/spokvulcan/tesseract/issues/228)) ([6ae4a34](https://github.com/spokvulcan/tesseract/commit/6ae4a342af12bbc73330297be7fec56a42bfdb36))
* **server:** cut the Activity page over from the Dashboard (map [#269](https://github.com/spokvulcan/tesseract/issues/269)) ([#280](https://github.com/spokvulcan/tesseract/issues/280)) ([584cd8b](https://github.com/spokvulcan/tesseract/commit/584cd8b5071335ef4a6017bbeabe7ac105660b69))
* **server:** cut the Cache page over from Prompt Cache (map [#269](https://github.com/spokvulcan/tesseract/issues/269)) ([#278](https://github.com/spokvulcan/tesseract/issues/278)) ([df5354d](https://github.com/spokvulcan/tesseract/commit/df5354d698e6b5497e5a76dcddfedd51907363b9))
* **server:** default preserve-thinking on for declaring models ([#237](https://github.com/spokvulcan/tesseract/issues/237)) ([cf3f34a](https://github.com/spokvulcan/tesseract/commit/cf3f34af88ff65a9b3c39478b2862cc4da881858))
* **settings:** cut over to the native Settings window ([#243](https://github.com/spokvulcan/tesseract/issues/243)) ([ca04210](https://github.com/spokvulcan/tesseract/commit/ca04210b700f3e3176111050cd422f43f3e73c35))
* **settings:** native Settings window prototype ([#215](https://github.com/spokvulcan/tesseract/issues/215)) ([bfa7a6a](https://github.com/spokvulcan/tesseract/commit/bfa7a6adf3877ba6d1908cb5631e7da4079f1540))
* **speech:** face-lift the Speech page to the design language ([#263](https://github.com/spokvulcan/tesseract/issues/263)) ([f13c83f](https://github.com/spokvulcan/tesseract/commit/f13c83fcd6d7adcbc8a271e7453805029a69fa01))


### Bug Fixes

* **agent:** tool-result images now reach the model instead of being dropped ([#241](https://github.com/spokvulcan/tesseract/issues/241)) ([0e83c9c](https://github.com/spokvulcan/tesseract/commit/0e83c9c4d1fab085a8958fd6468c43ba7b062cb1))
* **design:** consistency audit of the locked surfaces (map [#211](https://github.com/spokvulcan/tesseract/issues/211), last ticket) ([#267](https://github.com/spokvulcan/tesseract/issues/267)) ([f33c612](https://github.com/spokvulcan/tesseract/commit/f33c6123f7ea420a4e52f5b0d2d1cb110d306e40))
* **server:** count reclaimable memory in RAM cache headroom ([#236](https://github.com/spokvulcan/tesseract/issues/236)) ([5a29335](https://github.com/spokvulcan/tesseract/commit/5a29335b3e28ffc8bf93a3d5b624b5d042efb9be))
* **vendor:** adopt upstream AutoAWQ converter fixes (theta-filter, scales dtype) ([#223](https://github.com/spokvulcan/tesseract/issues/223)) ([3f31a43](https://github.com/spokvulcan/tesseract/commit/3f31a4363f18ecd0057c78b430219bf6f0a0ddca))
* **vlm:** apply sRGB tone curve in Qwen3VL image preprocessing ([#242](https://github.com/spokvulcan/tesseract/issues/242)) ([78334f1](https://github.com/spokvulcan/tesseract/commit/78334f1dd2e6bf8ac2ab067e703464b54ae9d63f))


### Performance Improvements

* **agent:** default kvBits to nil (unquantized KV cache) ([#260](https://github.com/spokvulcan/tesseract/issues/260)) ([ca3c744](https://github.com/spokvulcan/tesseract/commit/ca3c7449b55f813741f0a39f862c70cc5a1519ff))
* **prefill:** balance the prompt chunks instead of leaving a remainder ([#261](https://github.com/spokvulcan/tesseract/issues/261)) ([c993da1](https://github.com/spokvulcan/tesseract/commit/c993da138b6d80a2daca39edc35347f4e43e6250))


### Code Refactoring

* **agent:** remove dead app-side ParoQuant duplicate ([#221](https://github.com/spokvulcan/tesseract/issues/221)) ([6086433](https://github.com/spokvulcan/tesseract/commit/6086433e9242f70ce936d848c86f87fa999d758b))


### Documentation

* **design:** ratify the app-wide design language one-pager ([cbc7c71](https://github.com/spokvulcan/tesseract/commit/cbc7c7166f71649d7261641ebc5b2c9cf19cff9d))

## [1.6.0](https://github.com/spokvulcan/tesseract/compare/v1.5.0...v1.6.0) (2026-07-08)


### Features

* **agent:** full MCP client with HTTP transports ([#190](https://github.com/spokvulcan/tesseract/issues/190)) ([#196](https://github.com/spokvulcan/tesseract/issues/196)) ([e294b5d](https://github.com/spokvulcan/tesseract/commit/e294b5de4d697fd5aef4fc12160345b158a85fef))
* **agent:** instant Pending Row on send + Waiting Row for model waits ([28c1ab9](https://github.com/spokvulcan/tesseract/commit/28c1ab9921034f9c9b7ba97a1d340e8bf3f715f8))
* **agent:** Tool Panels — specialized OpenCode-style tool-call rendering ([#201](https://github.com/spokvulcan/tesseract/issues/201)) ([f079f4c](https://github.com/spokvulcan/tesseract/commit/f079f4c99c4a3d0344e245da11a613fd9cfb6356))
* **browser:** Agent Browser + Browser MCP Server ([#189](https://github.com/spokvulcan/tesseract/issues/189)) ([#191](https://github.com/spokvulcan/tesseract/issues/191)) ([c92ee60](https://github.com/spokvulcan/tesseract/commit/c92ee607c1aff32287b351ed316bd046270509f0))
* **browser:** make the Browser MCP the sole web surface; render search in WebKit ([#199](https://github.com/spokvulcan/tesseract/issues/199)) ([7a8d86e](https://github.com/spokvulcan/tesseract/commit/7a8d86e24781ca179f8f14ae2f08ac90adecef2e))


### Bug Fixes

* **agent:** stable transcript rhythm — Waiting Row geometry, edge newlines, line spacing ([dc07223](https://github.com/spokvulcan/tesseract/commit/dc0722369ac9362275c2eee98391431fac4053ee))
* **browser:** enforce navigation timeout and bound every tool call ([#197](https://github.com/spokvulcan/tesseract/issues/197)) ([ca18828](https://github.com/spokvulcan/tesseract/commit/ca188280130c901c616707813612d9296eb97a8c))

## [1.5.0](https://github.com/spokvulcan/tesseract/compare/v1.4.0...v1.5.0) (2026-07-07)


### Features

* **dictation:** always-armed Voice Processing with SPI un-duck (ADR-0025) ([7a22ba4](https://github.com/spokvulcan/tesseract/commit/7a22ba472eb1141683b44f0c1cac3534449f4673)), closes [#188](https://github.com/spokvulcan/tesseract/issues/188)
* **models:** point Ornith 9B at vision-capable MLX 6-bit build ([762fd0b](https://github.com/spokvulcan/tesseract/commit/762fd0b9f833f80a9efe77a9f8a1ecfc70570dbc))


### Tests

* **agent:** essentials skills are pill-only (disable-model-invocation) ([b353003](https://github.com/spokvulcan/tesseract/commit/b3530034eebb5f741652871f8b0c656847523c1b))

## [1.4.0](https://github.com/spokvulcan/tesseract/compare/v1.3.0...v1.4.0) (2026-07-07)


### Features

* **agent:** icon-only model button in the composer's right cluster ([46bb480](https://github.com/spokvulcan/tesseract/commit/46bb480c31c72271d4af8ef4cbc503dcaefd8bfe))
* **agent:** stream tool calls in the chat — Open Tool Call row + Tool Clock ([5e06b7a](https://github.com/spokvulcan/tesseract/commit/5e06b7aafdbce8e51084be66c0798265b0aef73b))
* **agent:** tighten system prompt and skills for small-model clarity ([d6607c5](https://github.com/spokvulcan/tesseract/commit/d6607c57dd58cc5cac47f68a4dc6d9577b054a01))
* **agent:** transcript rhythm, tool row titles, and markdown accents ([0e48587](https://github.com/spokvulcan/tesseract/commit/0e48587733755712a36f460ac07c03a9419bd29d))


### Code Refactoring

* **agent:** hoist tool-call name-lock to the producers, tidy the fold ([baf6101](https://github.com/spokvulcan/tesseract/commit/baf610147787f2848e752ff9ca08d0a4a7da67a3))


### Continuous Integration

* **release:** fail fast and alert when Apple notary access is blocked ([#185](https://github.com/spokvulcan/tesseract/issues/185)) ([7599a1b](https://github.com/spokvulcan/tesseract/commit/7599a1b2efed1342e5d52d89a828a14d78629fa9))

## [1.3.0](https://github.com/spokvulcan/tesseract/compare/v1.2.0...v1.3.0) (2026-07-06)


### Features

* **agent:** chat rewrite — pi-ai parts model, Live Part rendering, flat document UI ([#184](https://github.com/spokvulcan/tesseract/issues/184)) ([1cf84cc](https://github.com/spokvulcan/tesseract/commit/1cf84cc3a2e480d08612619d1ee78f1714c76bb4))
* **agent:** Skill Pills — built-in essentials skills as instant-action pills above the composer ([#177](https://github.com/spokvulcan/tesseract/issues/177)) ([e25745e](https://github.com/spokvulcan/tesseract/commit/e25745ebad727ec4adf99bc98c467876e883337c))
* **dictation:** instant re-record, kept capture engine, Liquid Glass pill ([#180](https://github.com/spokvulcan/tesseract/issues/180)) ([3c1720c](https://github.com/spokvulcan/tesseract/commit/3c1720c8c20a443669603be44c8ee42599348d73))
* **dictation:** Voice Processing toggle, Capture Dump, anti-aliased resampler (PRD [#175](https://github.com/spokvulcan/tesseract/issues/175)) ([#178](https://github.com/spokvulcan/tesseract/issues/178)) ([b390ec9](https://github.com/spokvulcan/tesseract/commit/b390ec999836be96634613206f71a02dd8514bf0))
* **server:** Batch Engine — concurrent completion lanes over one GPU lease ([#176](https://github.com/spokvulcan/tesseract/issues/176)) ([8fef7c0](https://github.com/spokvulcan/tesseract/commit/8fef7c0fb0ca9368aa86eafe382cf3768aca0352))


### Bug Fixes

* **agent:** preserve the composer draft across new chat and thread switches ([#182](https://github.com/spokvulcan/tesseract/issues/182)) ([3c455d5](https://github.com/spokvulcan/tesseract/commit/3c455d526d094e12c6c1ae7ef9d9c134b5e50a60))
* **dictation:** arm Voice Processing per capture — idle no longer ducks system audio ([c687938](https://github.com/spokvulcan/tesseract/commit/c687938c983e44a3eccb7137a0c5226c33aa8f55))
* **dictation:** pill back to regular glass following the system appearance ([5763887](https://github.com/spokvulcan/tesseract/commit/5763887822148882a28bdd8536ebc88504d87507))
* **dictation:** pure clear Liquid Glass pill under forced light appearance ([6773e99](https://github.com/spokvulcan/tesseract/commit/6773e99829c5743a1afcf90eb94ebc6c7b622adf))


### Reverts

* feat(server): Batch Engine — concurrent completion lanes over one GPU lease ([#176](https://github.com/spokvulcan/tesseract/issues/176)) ([72d61ed](https://github.com/spokvulcan/tesseract/commit/72d61ed33e7f0133c415ef59c986e4a9b1e6472c))

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
