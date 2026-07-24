# Changelog

## [1.10.0](https://github.com/spokvulcan/tesseract/compare/v1.9.0...v1.10.0) (2026-07-24)


### Features

* **companion:** Notification Hub v1 — Jarvis reads every banner (PRD [#376](https://github.com/spokvulcan/tesseract/issues/376)) ([#382](https://github.com/spokvulcan/tesseract/issues/382)) ([409e2e7](https://github.com/spokvulcan/tesseract/commit/409e2e7f8b21fca914099fd97fcf2b5ad6d11d51))
* **companion:** the Event Fold, whole (ADR-0046, [#367](https://github.com/spokvulcan/tesseract/issues/367)–[#373](https://github.com/spokvulcan/tesseract/issues/373)) ([#374](https://github.com/spokvulcan/tesseract/issues/374)) ([9937d59](https://github.com/spokvulcan/tesseract/commit/9937d59f8c37265a5c183dbf26faebf40f232995))
* **models:** Nanbeige4.2-3B agent model (looped transformer, MLX 8-bit) ([#421](https://github.com/spokvulcan/tesseract/issues/421)) ([80b7ac5](https://github.com/spokvulcan/tesseract/commit/80b7ac5d2345d18195ba7706b7d1003c9681b74b))


### Bug Fixes

* **agent:** turn replay breaker + output-only presence penalty (ADR-0053) ([#390](https://github.com/spokvulcan/tesseract/issues/390)) ([59dd88c](https://github.com/spokvulcan/tesseract/commit/59dd88c5bba1aa19bd04ad1a3b1ce5d5710cecb4))
* **agent:** turn replay breaker + output-only presence penalty (ADR-0053) ([#392](https://github.com/spokvulcan/tesseract/issues/392)) ([c667c66](https://github.com/spokvulcan/tesseract/commit/c667c660604115e7dbfa558695b499dda91fd72f))
* review-round fixes for the C1–C13 mlx-core loop — C13 uncompilable kernel, C11/C12 model leak, hardening ([#426](https://github.com/spokvulcan/tesseract/issues/426)) ([3d1b15c](https://github.com/spokvulcan/tesseract/commit/3d1b15cc8f2e22c1ce137eb62e6bf99130668764))


### Performance Improvements

* inference-optimization experiment loop — E1/E2/E6b/E7/E11 accepted (+5-6% MoE prefill, +3% MoE decode, +5-6.5% dense decode, −205ms TTFT) ([#424](https://github.com/spokvulcan/tesseract/issues/424)) ([7b55591](https://github.com/spokvulcan/tesseract/commit/7b555911e1a603646bc101342cb535a45cb29ea1))
* **metal:** rows-per-expert-aware gather_qmm_rhs tile geometry (C1) ([e6b3702](https://github.com/spokvulcan/tesseract/commit/e6b370204ae094d7aa78870b2f957073f8c95389))
* **mlx-core:** C13 ACCEPTED — fused causal-mask+softmax for SDPA fallback (MoE 32K prefill ~+3%) ([dc93c18](https://github.com/spokvulcan/tesseract/commit/dc93c18256cd9a089483f824974a7a1988e388a0))
* **mlx-core:** C4 ACCEPTED — relaxed input cap + output-byte commit accounting ([79bf712](https://github.com/spokvulcan/tesseract/commit/79bf7122b838426655ef865b19ab6b03cd8bac08))
* **mlx-core:** C5 ACCEPTED — per-cbuf buffer-retention coalescing ([a25a297](https://github.com/spokvulcan/tesseract/commit/a25a297141c9b209c65394c35bdfa6d0433c07dd))
* **mlx-core:** C6 ACCEPTED — custom-kernel source memo (MoE decode +3.1-4.7%) ([4ac8873](https://github.com/spokvulcan/tesseract/commit/4ac88734c6986613a19ac5883d61d855626f7cc3))
* **mlx-core:** C7 ACCEPTED — per-model commit policy (MoE decode +3.7-5.9%) ([268ea1e](https://github.com/spokvulcan/tesseract/commit/268ea1e4dc7565e4ee1d9a5dd9b78d56a2cba2a1))
* **mlx-core:** C8 ACCEPTED — eval_impl flat degree map (MoE decode +1.4-2%) ([3308f25](https://github.com/spokvulcan/tesseract/commit/3308f25b1fff31738260c7fd25c685700c09d694))
* **mlx-core:** C9 ACCEPTED — gather identity-index cache (MoE 8K prefill +3.6%) ([909d589](https://github.com/spokvulcan/tesseract/commit/909d58914df443c20093b6d1c6f3eab61a0efb75))
* **qwen35:** C11 ACCEPTED — compiled MoE block in decode (MoE decode +3-7%) ([7106e77](https://github.com/spokvulcan/tesseract/commit/7106e7729077fb43dbee3bddc148fe6ffa2fef6a))
* **qwen35:** C12 ACCEPTED — compiled GDN decode step (dense 128 decode +1.75%, MoE +0.94%) ([863eb66](https://github.com/spokvulcan/tesseract/commit/863eb66930ee6e5fdbe9b6460eaa8c007a74f537))


### Code Refactoring

* **agent:** Active Tool Set — one resolve for tools and prompt (ADR-0048) ([#383](https://github.com/spokvulcan/tesseract/issues/383)) ([f7c71a4](https://github.com/spokvulcan/tesseract/commit/f7c71a46b95ec39a42f243d857e4f32c10cfe4a1))
* **agent:** Chat Session diet — skill execution and opening context become leaves ([#414](https://github.com/spokvulcan/tesseract/issues/414)) ([b991ef1](https://github.com/spokvulcan/tesseract/commit/b991ef1340d435ddc8d10d37f461fd443ea03c1c))
* **agent:** one seam for the generation logit-processor decision ([#405](https://github.com/spokvulcan/tesseract/issues/405)) ([#419](https://github.com/spokvulcan/tesseract/issues/419)) ([b4af19f](https://github.com/spokvulcan/tesseract/commit/b4af19fe0654123392b52de812dbfe3fef4fdc46))
* **agent:** Prefill Strategy — one route decision for the raw arms (ADR-0044) ([#364](https://github.com/spokvulcan/tesseract/issues/364)) ([df6a6ef](https://github.com/spokvulcan/tesseract/commit/df6a6ef3949b8527c50455604ffa13dae3a03d46))
* **agent:** Skill Envelope — render and parse as enforced inverses ([#413](https://github.com/spokvulcan/tesseract/issues/413)) ([8ba2c82](https://github.com/spokvulcan/tesseract/commit/8ba2c820e1d4ed2cffd0ae8f7add5b69542c6a99))
* **app:** Bootstrap Sequence — setup()'s ordering invariants, declared and assertable ([#415](https://github.com/spokvulcan/tesseract/issues/415)) ([10bd642](https://github.com/spokvulcan/tesseract/commit/10bd642d09cbda1eb15a8ef7e6058fe31cac0187))
* **app:** Model Selection Healing — both availability rules in one decider ([#416](https://github.com/spokvulcan/tesseract/issues/416)) ([8ec0464](https://github.com/spokvulcan/tesseract/commit/8ec04645c03f971d82b876cd7aaa680b8aeb2429))
* **audio:** Hold Wiring Arbiter — the voice hold's races as a value machine (ADR-0050) ([#385](https://github.com/spokvulcan/tesseract/issues/385)) ([1febfa7](https://github.com/spokvulcan/tesseract/commit/1febfa71f0fa61c33cb4bfa0fdf5275a0992331b))
* **companion:** Companion Fold Reducer — the Event Fold's write side as one decider (ADR-0051) ([#386](https://github.com/spokvulcan/tesseract/issues/386)) ([b1b22f1](https://github.com/spokvulcan/tesseract/commit/b1b22f1583a578afda053a48329cd0ae2eba70e7))
* **companion:** Companion Fold Render — shared line primitives under both briefings ([#417](https://github.com/spokvulcan/tesseract/issues/417)) ([f94429b](https://github.com/spokvulcan/tesseract/commit/f94429b5ec976b5541079cbcb0d16104fc8c89d5))
* **companion:** due-wake presentation de-dup — render once, count once ([#418](https://github.com/spokvulcan/tesseract/issues/418)) ([47d12eb](https://github.com/spokvulcan/tesseract/commit/47d12eb4c9c5a11fcf83e0965a40c81ccfcbf96a))
* **companion:** Reaction Single-Homing — every surface reports through the reducer ([#394](https://github.com/spokvulcan/tesseract/issues/394)) ([ac44bab](https://github.com/spokvulcan/tesseract/commit/ac44bab3fded160114fa8c400928da6bb6b7d813))
* **companion:** Trace Vocabulary — typed flight-recorder events ([#398](https://github.com/spokvulcan/tesseract/issues/398)) ([7b274ec](https://github.com/spokvulcan/tesseract/commit/7b274ecdfc5f9b3ac879c6191405e81931f4430c))
* **companion:** Voice Session Machine — the session loop as a pure reducer (ADR-0042) ([#362](https://github.com/spokvulcan/tesseract/issues/362)) ([30c10e7](https://github.com/spokvulcan/tesseract/commit/30c10e75f7538e08e60047756dc565af54467aa4))
* **companion:** Wake Evaluator — the pure decider ADR-0040 promised (ADR-0043) ([#363](https://github.com/spokvulcan/tesseract/issues/363)) ([64e042b](https://github.com/spokvulcan/tesseract/commit/64e042b5a03b13f9f0e6f8e635eeeb31bbe60cc6))
* **memory:** Conversation Memory — the chat's memory fold out of the Chat Session (ADR-0045) ([#365](https://github.com/spokvulcan/tesseract/issues/365)) ([974958e](https://github.com/spokvulcan/tesseract/commit/974958e7cb1d5d8e9b917649b5f6c35d75d15883))
* **memory:** delete the unwired MemoryCallback ([#420](https://github.com/spokvulcan/tesseract/issues/420)) ([1457f13](https://github.com/spokvulcan/tesseract/commit/1457f131ed0544f1fb1d4b0f27efc7ba19728ed9))
* **platform:** Menu Bar Activity Resolver — the priority merge beside its precedent ([#409](https://github.com/spokvulcan/tesseract/issues/409)) ([7bb5e3c](https://github.com/spokvulcan/tesseract/commit/7bb5e3cac9ef2b4aae57e2ab61ce163588cbbf15))
* **server:** Eviction Candidate Policy — pure victim selection on both tiers (ADR-0049) ([#384](https://github.com/spokvulcan/tesseract/issues/384)) ([b4d6d32](https://github.com/spokvulcan/tesseract/commit/b4d6d322db726ddbb83c4a6ff13b80238ae79314))
* **server:** Snapshot Resolution ladder — pure hydration decisions, ownership unmoved ([#412](https://github.com/spokvulcan/tesseract/issues/412)) ([aaf0196](https://github.com/spokvulcan/tesseract/commit/aaf01963346869e6b378727cf4d684a1119ce7b0))
* **server:** Stream Lifecycle Driver — the SSE transport race, testable without a socket ([#411](https://github.com/spokvulcan/tesseract/issues/411)) ([a1e5ac9](https://github.com/spokvulcan/tesseract/commit/a1e5ac978e6d705157b6ecbd62d57a3845f6d963))
* **server:** Warm-Start Planner — the ledger's pure rebuild decisions (ADR-0055) ([#410](https://github.com/spokvulcan/tesseract/issues/410)) ([a2113ca](https://github.com/spokvulcan/tesseract/commit/a2113ca1e215edeb9149a848a5a4bd699513889e))
* **speech:** Streaming Scheduler — one value machine under both playback adapters (ADR-0054) ([#402](https://github.com/spokvulcan/tesseract/issues/402)) ([bbfb3c4](https://github.com/spokvulcan/tesseract/commit/bbfb3c4583dfea4bdbcfd36a1019f84a47420d94))


### Documentation

* **bench:** C10 REJECTED — metadata-only fast path; CPU slack means spread-out CPU cuts no longer convert ([756943e](https://github.com/spokvulcan/tesseract/commit/756943e88989a2a2388d7fa85304b1fd47da48dc))
* **bench:** C2 REJECTED — MLX_MAX_OPS_PER_BUFFER raise (M2 probe) ([cb9e581](https://github.com/spokvulcan/tesseract/commit/cb9e581fe3daab56980eb37b2f8426e1a170c7b3))
* **bench:** C3 REJECTED at probe — no gather_qmv geometry lever (M3) ([baf3bab](https://github.com/spokvulcan/tesseract/commit/baf3bab5d539f49778c5a5eb386bdade57a6a81d))
* **bench:** M4 REJECTED at probe — fused rotate+qmv bitwise-exact but 2x slower (geometry); two-kernel pipeline wins ([8d47f12](https://github.com/spokvulcan/tesseract/commit/8d47f12253153133dee0eb2671ba3cf1533b0bc4))
* **bench:** M8 REJECTED at probe — expert routing locality 2.4/8, prefetch dead ([90c6942](https://github.com/spokvulcan/tesseract/commit/90c6942916fbe17981864b5aa280578195647438))
* **bench:** persist operational state after C4 (pins, binaries, C5 queue) ([e24c621](https://github.com/spokvulcan/tesseract/commit/e24c621d306d44fac244f98ee439f231198005e0))
* **bench:** persist operational state for session resume ([2b17481](https://github.com/spokvulcan/tesseract/commit/2b174818eefabb8a36cbeab07b5d8806477a6378))
* **roadmap:** amend M2/M3 with C2/C3 verdicts ([b7c8519](https://github.com/spokvulcan/tesseract/commit/b7c8519bad9a0517a0a772fd3d2988e05d0e0743))
* **roadmap:** future optimization work — ranked remaining opportunities after C1-C13 ([fabf9f7](https://github.com/spokvulcan/tesseract/commit/fabf9f736dacd1c39952b1a75089d9e116b4fd2a))


### Build System

* **deps:** pin Cmlx via spokvulcan/mlx + mlx-swift forks ([a6bc8c8](https://github.com/spokvulcan/tesseract/commit/a6bc8c8f891c61a5a7ed31ae1fa1f0432898e4bd))


### Miscellaneous Chores

* **bench:** BENCH_RUNS env in parity-ab.sh (default A/B is 3 pairs now) ([033f900](https://github.com/spokvulcan/tesseract/commit/033f9009372e8cb9f0e61a436a19cbac579f887e))
* Package.resolved for C6 pins ([1f49e4c](https://github.com/spokvulcan/tesseract/commit/1f49e4c6cff36517c3ea88676948333e5803f491))
* **server:** persist per-request decode tok/s at notice level ([ca6b47d](https://github.com/spokvulcan/tesseract/commit/ca6b47d975e5ee1d2f1f05ba8b68a548e6989a1e))
* **vendor:** re-pin mlx-swift-lm on upstream 343cae3; drop the gemma carries ([f6fc839](https://github.com/spokvulcan/tesseract/commit/f6fc839ee5e7336cb09f7f76a79370a079ecf9ca))
* **vendor:** re-pin mlx-swift-lm on upstream eaefe75 ([#423](https://github.com/spokvulcan/tesseract/issues/423)) ([5d955f4](https://github.com/spokvulcan/tesseract/commit/5d955f46fe76a621eefa88632a315b68c642d350))

## [1.9.0](https://github.com/spokvulcan/tesseract/compare/v1.8.1...v1.9.0) (2026-07-16)


### Features

* **speech:** voice engine v2 — package engine, PinnedVoice, q6 default ([#334](https://github.com/spokvulcan/tesseract/issues/334)) ([#348](https://github.com/spokvulcan/tesseract/issues/348)) ([22f5898](https://github.com/spokvulcan/tesseract/commit/22f5898cb6d2071a1d3bee586b6004101c0d4cb9))


### Bug Fixes

* **dictation:** return the Clipboard Loan — screenshot-sized restores, clear when nothing saved ([#351](https://github.com/spokvulcan/tesseract/issues/351)) ([21cfa25](https://github.com/spokvulcan/tesseract/commit/21cfa259d4f673a289f3fd36e59a1f687f9edb8b))
* **memory:** avoid wake notification actor crash ([4b84a62](https://github.com/spokvulcan/tesseract/commit/4b84a623e5095e20e644ab1917ce5bd12a94aae7))


### Documentation

* mlx-swift-lm fork ledger + ADR-0014 amendment (1,280 vision-token budget) ([d31c38d](https://github.com/spokvulcan/tesseract/commit/d31c38d8bd932b984b8c9778625dd91df9576b9f))


### Miscellaneous Chores

* **vendor:** re-pin mlx-swift-lm on upstream f1573a9 — [#399](https://github.com/spokvulcan/tesseract/issues/399)/[#411](https://github.com/spokvulcan/tesseract/issues/411)/[#418](https://github.com/spokvulcan/tesseract/issues/418) merged ([7c07942](https://github.com/spokvulcan/tesseract/commit/7c07942c659f9c8f07763958d5123acec26f9a84))

## [1.8.1](https://github.com/spokvulcan/tesseract/compare/v1.8.0...v1.8.1) (2026-07-12)


### Tests

* **memory:** the corpus gate fails open on an empty directory — CI ran the eval against nothing ([566c46e](https://github.com/spokvulcan/tesseract/commit/566c46eca9e2963b71d22884ec48594b599a1caa))

## [1.8.0](https://github.com/spokvulcan/tesseract/compare/v1.7.0...v1.8.0) (2026-07-12)


### Features

* **agent:** full-size type for inline code and code blocks ([7ebd0c8](https://github.com/spokvulcan/tesseract/commit/7ebd0c8fc9ed2668ec40a3c01b473bab65aeaa67))
* **agent:** inline-code chip, blockquote bar, and the Markdown Gallery ([94e521d](https://github.com/spokvulcan/tesseract/commit/94e521df99f008aaf89738dfa530dc2886fb1d19))
* **agent:** Prepared Checkpoint for PARO loads (ADR-0032) ([9ee816e](https://github.com/spokvulcan/tesseract/commit/9ee816ed7b12cf01e60e9e3a8ffd7a5156ec9acd))
* **companion:** walking skeleton — the lived-with heartbeat ([#303](https://github.com/spokvulcan/tesseract/issues/303)) ([#330](https://github.com/spokvulcan/tesseract/issues/330)) ([1538502](https://github.com/spokvulcan/tesseract/commit/1538502072de20ba48cf22e38e99aa777a04e750))
* **dictation:** baseline latency instrumentation (DictationPerf) ([#296](https://github.com/spokvulcan/tesseract/issues/296)) ([ea2e93f](https://github.com/spokvulcan/tesseract/commit/ea2e93f8a3c249fc16d552ced1b5d522aa453b73))
* **dictation:** Correction Pair flywheel — capture, affordances, store ([#300](https://github.com/spokvulcan/tesseract/issues/300)) ([e42d8c1](https://github.com/spokvulcan/tesseract/commit/e42d8c13549e2a30b4ffb522a469d1789d45a333))
* **dictation:** five overlay variant explorations for the redesign batch ([#312](https://github.com/spokvulcan/tesseract/issues/312)) ([b9fb7bb](https://github.com/spokvulcan/tesseract/commit/b9fb7bbcbae2aac7cf4ed4768ec03fb41cf4db2e))
* **dictation:** live partial transcription + the Caption overlay variant ([#329](https://github.com/spokvulcan/tesseract/issues/329)) ([522e983](https://github.com/spokvulcan/tesseract/commit/522e98340c6e72c661e1d4b2fda0c91b9a39d50f))
* **dictation:** Proofread Pass v1 — own MLX model, skip-when-busy (ADR-0034) ([#299](https://github.com/spokvulcan/tesseract/issues/299)) ([96fe775](https://github.com/spokvulcan/tesseract/commit/96fe775fc383691c7a3683ba7b95afd3a22a0b6d))
* **memory:** "that's wrong" now sends the belief back to the evidence ([3a0d4a1](https://github.com/spokvulcan/tesseract/commit/3a0d4a1b1a5c2456eb07bc2ae883d37a42e6f31a))
* **memory:** ground and check the morning callback before it speaks ([aed962f](https://github.com/spokvulcan/tesseract/commit/aed962f0f53c9dee47cfbbb8abdbd5cc6b0843be))
* **memory:** sleep consolidation, idle detection, and the morning callback (ADR-0035) ([e5692ea](https://github.com/spokvulcan/tesseract/commit/e5692eadc0f2c584611e779cb1094ce2fa12dc7a))
* **memory:** the correction gets a door — contest tool, multi-REPLACES, and the reply that never arrived ([#333](https://github.com/spokvulcan/tesseract/issues/333)) ([e0ba7e4](https://github.com/spokvulcan/tesseract/commit/e0ba7e4aceeaeb905a2c96e1ea7fbd62c522ba54))
* **memory:** the Memory window — what I believe about you, and why (ADR-0035 §9) ([fb3fb29](https://github.com/spokvulcan/tesseract/commit/fb3fb29aff743593ec516006f5d67e57c42c17d5))
* **memory:** the owner's switches, and a stricter test for "he said it" ([1e59cdf](https://github.com/spokvulcan/tesseract/commit/1e59cdfe391a68a86f5e86301f0d4d96c00ce6af))
* **memory:** the two-layer living store — schema, lifecycle, SQLite, embedder (ADR-0035) ([ff36885](https://github.com/spokvulcan/tesseract/commit/ff368852cca96d274a3d7505e2024d16afb5de34))
* **memory:** wire the living memory into chat, dictation, and the cold start (ADR-0035) ([b702c68](https://github.com/spokvulcan/tesseract/commit/b702c68577a3423502001b904ceeaa22920e15f0))


### Bug Fixes

* **memory:** every short memory embedded to the same vector — the EOS attractor ([#332](https://github.com/spokvulcan/tesseract/issues/332)) ([13fa8c9](https://github.com/spokvulcan/tesseract/commit/13fa8c962b431ac613713c152cec86476e174b65))
* **memory:** journal confirmations, and capture through the tested unwrap ([e1994ab](https://github.com/spokvulcan/tesseract/commit/e1994abc48cd36c256ad0e5d74016d74db771910))
* **memory:** nine seam defects from the first outside look, and the store takes ownership of its writes ([f7371e6](https://github.com/spokvulcan/tesseract/commit/f7371e67ee051b3c2fc496fde1c66e68f09429a9))
* **memory:** remember speaks the agent's voice, and dictation keeps one door per testimony ([#333](https://github.com/spokvulcan/tesseract/issues/333)) ([7401300](https://github.com/spokvulcan/tesseract/commit/7401300a648888d27b1029e59cd2816e45284afb))
* **memory:** sleep must not eat the store it is trying to build ([67ed1bc](https://github.com/spokvulcan/tesseract/commit/67ed1bcfa0ce4c6fda3a2835252056d3d6e07d96))
* **memory:** the model was never actually shown a single memory ([b52e31d](https://github.com/spokvulcan/tesseract/commit/b52e31dcbda2e07ef73e7302804ad42d397d3b66))
* **memory:** tier comparison ran backwards — retirements were counted as promotions ([aff06d2](https://github.com/spokvulcan/tesseract/commit/aff06d2a671545eed0e5ce9866cc99ef26910f7a))


### Code Refactoring

* **agent:** collapse the Agent event double-fold to the message log ([00ef9e6](https://github.com/spokvulcan/tesseract/commit/00ef9e63a30f3086f9f103a5dcbbef5dd8469f64))
* **agent:** Managed Generation Driver — one envelope for both spines ([afcfe7c](https://github.com/spokvulcan/tesseract/commit/afcfe7cb1a6fa7d8b7e76bc59264327a5beefc78))
* **agent:** one conversation-switch sequence in Chat Session ([6c6643d](https://github.com/spokvulcan/tesseract/commit/6c6643da10e85183119c1b523f7ce67201c28142))
* **agent:** Vision Availability leaf controller ([eae7d98](https://github.com/spokvulcan/tesseract/commit/eae7d9819c1f1ec73886fd9dbf277b5fcbe6e2b9))
* **dictation:** delete the full-screen border overlay style ([#295](https://github.com/spokvulcan/tesseract/issues/295)) ([8ee6dd6](https://github.com/spokvulcan/tesseract/commit/8ee6dd68b39a3d90f4e05e4854d9cb5e24c3b3d4))
* **dictation:** extract the Capture Engine Lifecycle decision table ([a1fc052](https://github.com/spokvulcan/tesseract/commit/a1fc052abb8596c0b0ee417e54399376dbdc0f3c))
* **dictation:** pipeline hardening — audit items 7–10 ([#298](https://github.com/spokvulcan/tesseract/issues/298)) ([dd56f9e](https://github.com/spokvulcan/tesseract/commit/dd56f9ea0af27ca96e2cd06e953ce54b4b9f570d))
* **personal-assistant:** retire the memories.md era — the living memory replaces it ([109ae25](https://github.com/spokvulcan/tesseract/commit/109ae253d1ef4e239968560918e2a236702d6d78))
* **platform:** one hotkey matcher behind two thin event adapters ([9c127ad](https://github.com/spokvulcan/tesseract/commit/9c127adb707689c190e0b9e9f5b3a10a65e26b74))
* **server:** carve the Leaf Store phase out of the completion drive ([1bee73a](https://github.com/spokvulcan/tesseract/commit/1bee73abe2e3ae2e80a203c4cb403695a3d5c44c))
* **server:** carve the Request Keying phase out of the generation build ([7482787](https://github.com/spokvulcan/tesseract/commit/7482787cd6660d744b3e0ea798989164f213933b))
* **server:** extension-transfer shield becomes a claim the ledger enforces ([2cb881a](https://github.com/spokvulcan/tesseract/commit/2cb881a1197764660038bb91671945f6c12d1596))
* **server:** one home for completion-trace assembly ([d72d7d0](https://github.com/spokvulcan/tesseract/commit/d72d7d0a46e6a166e522368321722f187c3009d5))
* **server:** SSD Residency — one typed observation replaces the ForTesting reads ([a067cbb](https://github.com/spokvulcan/tesseract/commit/a067cbbae3d300ceb0a2be77911bd23b17242246))
* **speech:** extract PlaybackDiagnosticsDump from the playback adapter ([803e819](https://github.com/spokvulcan/tesseract/commit/803e81950b5fd6f5ebb1fc14ed1a2d023cea2c15))
* **speech:** one home for the TTS transient-error tail and voice context ([245161f](https://github.com/spokvulcan/tesseract/commit/245161fc3b1e3207c19da7e83d0d81119f216a9b))
* **transcription:** one owner for the Whisper model file contract ([68a0179](https://github.com/spokvulcan/tesseract/commit/68a0179d61b0928766dc75925dd488dfdf033567))


### Documentation

* **adr-0035:** record round two — the correction had no door, and the journal corrects the attribution ([c9aa673](https://github.com/spokvulcan/tesseract/commit/c9aa673b219e1288bc4ee535285e6d600d649e48))
* **adr-0035:** record the EOS attractor — and name the three read paths ([#332](https://github.com/spokvulcan/tesseract/issues/332)) ([6edb475](https://github.com/spokvulcan/tesseract/commit/6edb475c47d65e2b023ff4bb36e13c6948bada58))
* **adr-0035:** record the first outside look — nine defects, two kept departures ([0af817a](https://github.com/spokvulcan/tesseract/commit/0af817a012a0e340c6447fd3ac43bbc26a0c5ffa))
* **adr-0035:** record the inverted voice and the doubled doors ([#333](https://github.com/spokvulcan/tesseract/issues/333)) ([86c216e](https://github.com/spokvulcan/tesseract/commit/86c216ecaf4eb7823e599424606dfb79df015380))
* **adr-0035:** record the sixth finding — the line the owner would actually read ([c82a863](https://github.com/spokvulcan/tesseract/commit/c82a863498dfe9570746a00b39be48f28456c2fa))
* **adr-0035:** record what was actually built, and what the eval really said ([831b72c](https://github.com/spokvulcan/tesseract/commit/831b72c6bedec4b3e874ce3a13d7287bc6d00d34))
* **adr-0035:** the five bugs the live run caught, and the eval's post-clean numbers ([9c9fa0c](https://github.com/spokvulcan/tesseract/commit/9c9fa0c9b954a9c3b6b819414246d1b4f66e0e50))
* **adr:** record the Server Completion phase decomposition (ADR-0033) ([cac66be](https://github.com/spokvulcan/tesseract/commit/cac66be62b936633d3fe368aa7b78a2526710893))
* **context:** glossary entries for the deepening program's new modules ([7882750](https://github.com/spokvulcan/tesseract/commit/7882750bed57fb156ab8495a9aa6cdd4a0f68f41))


### Tests

* **memory:** the gates that would have caught the attractor, and the belief-recall yardstick ([#332](https://github.com/spokvulcan/tesseract/issues/332)) ([c26e9fd](https://github.com/spokvulcan/tesseract/commit/c26e9fdfbb578961ebe50f6d68a1dfc0c09e51ea))
* **memory:** the yardstick — and it says the retrieval win is not real ([5731ff3](https://github.com/spokvulcan/tesseract/commit/5731ff3b69e67100e43c02afde9facf016d9ec86))
* **server:** retarget the source-shape pins to the ADR-0033 phase files ([e31c957](https://github.com/spokvulcan/tesseract/commit/e31c9579ab8a770d4f60f7f7b1024a930d852fee))
* **transcription:** pin TranscriptionPostProcessor behaviour directly ([349dbb0](https://github.com/spokvulcan/tesseract/commit/349dbb0a26a4816730192af03a898c7bc2f4d034))

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
