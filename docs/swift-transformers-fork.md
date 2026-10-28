# swift-transformers fork & pin scheme

How tesseract consumes a writable `huggingface/swift-transformers` — the
third fork level, established 2026-07-27 for the render+token cache
(experiments-ledger C25). Same rules as the mlx/mlx-swift forks
(`docs/mlx-core-fork.md`): pin branches are append-only, changes must be
general and upstreamable, exact-revision pins keep history reproducible.

## The pin chain

```
tesseract.xcodeproj
  └─ Vendor/tesseract-speech/Package.swift: the package's ONLY declarer
     in the app graph since ADR-0071 (before that, the retired
     Vendor/mlx-audio-swift; MLXLMCommon's `import Tokenizers` and the
     app's resolve through the Xcode workspace module leak, same as MLXFast):
       url: https://github.com/spokvulcan/swift-transformers
       revision: <exact commit on pin-tesseract>
  └─ spokvulcan/swift-transformers @ pin-tesseract-2026-09-15
       = huggingface/swift-transformers @ 1.3.4 + one commit per accepted
         carry
```

Pin branches are append-only and a base change gets a new dated branch (the
mlx-core rule): `pin-tesseract` (base 1.3.3, head `08933b6`) stays for the
historical tesseract pins; `pin-tesseract-2026-09-15` (base 1.3.4, head
`c99afdf`) is what the app builds. The 1.3.3 → 1.3.4 delta (revision
support in `AutoTokenizer.from(pretrained:)`, `Sendable` on the Hub types,
the swift-jinja 2.4.2 floor, a sampling fix for vocabularies over 65,536
entries) touches none of the carried files; all four rebased clean.

## Carries on `pin-tesseract-2026-09-15`

| Commit | What it does | Upstream status |
| --- | --- | --- |
| `8025d3d` (was `63edf42`) `feat(tokenizers): expose renderChatTemplate` | Splits the render half of `applyChatTemplate` into a public `renderChatTemplate` (pure refactor, byte-identical output; public default impl keeps third-party conformers compiling). Enables the C25 render+token cache | Not filed (queued — owner go-ahead) |
| `fd1a169` (was `a524093`) `perf(tokenizers): byte-native BPE inner loop + byte-keyed lookup tables` | Same serial algorithm, same merge order, byte-identical output on byte-level BPE vocabs: byte-range symbols in one UTF-8 buffer instead of per-scalar Strings and per-merge concats; open-addressed byte-keyed rank/id tables (FNV-1a + full byte-compare) instead of String-keyed dictionary probes. 1.22×/1.21×/1.20× encode at 32K/8K/128; 88/88 corpus items byte-identical (experiments-ledger C24). **Narrows the merge-table match semantics — see the semantics note below; the commit message's "byte-identical output" claim is scoped to byte-level vocabs.** Adds ~20 MB resident per loaded tokenizer (the byte tables sit alongside `bpeRanks`/`tokensToIds`, both still live) | Not filed (queued — owner go-ahead) |
| `d1f7ba1` (was `0033bc7`) `fix(tokenizers): C24 review round — eager byte tables, correct semantics note` | Replaces the lazy double-checked-locked `byteTablesCache` with a `let` built in `init` (the unlocked fast-path read had no acquire semantics — a reader on a weakly-ordered core could see the published pointer before the table's array buffers); corrects the `BytePairTables` doc comment, which claimed byte-exactness was equivalent to what it replaced; bounds the leading-byte scalar-width walk against a malformed buffer. No behavior change on well-formed input | Not filed (queued — owner go-ahead) |
| `fe95f0a` (was `08933b6`) `fix(tokenizers): render tojson the way transformers does`, then `c99afdf` `fix(tokenizers): render tojson with swift-jinja's json.dumps; drop the override` | Net: no carry. `fe95f0a` installed a `JSONEncoder` `tojson` in the `renderChatTemplate` context (a context filter wins over swift-jinja's built-in) to stop the built-in's `<\/style>` and `\u2014` escapes, which broke the canonical re-render of tool-call arguments; it kept sorted keys and compact separators. Once swift-jinja 2.5.0 rendered `tojson` as transformers does (`json.dumps(ensure_ascii=False)`: `", "` / `": "`, no escapes, Python floats), the override was the only thing still rendering `{"type":"string"}`, `2` for `2.0` and `"k" : v` when indented, in every tools block the models read. `c99afdf` removes it and raises the swift-jinja floor to 2.5.0. Sorted keys were never the override's to keep: `Value(any:)` sorts a Swift dictionary's keys, and the built-in keeps a value's order, so a caller passing Jinja values keeps theirs; the app passes its tools that way, as the `additionalContext` entry `tools` (ADR-0098). Two offline tests in `ChatTemplateTests` | swift-jinja issue [#71](https://github.com/huggingface/swift-jinja/issues/71) listed the divergences; our PR [#72](https://github.com/huggingface/swift-jinja/pull/72) (slash escaping) and upstream's [#74](https://github.com/huggingface/swift-jinja/pull/74) fixed them in 2.5.0. Nothing left to file |

### Merge-table semantics note (read before filing upstream)

`rank(in:...)` matches merge pairs by **raw UTF-8 bytes**. What it replaced —
`bpeRanks[BytePair(left, right)]` — matched under **Unicode canonical
equivalence**, because `BytePair` holds Swift `String`s and `String`'s
`Hashable` conformance is canonical (`BPETokenizer.swift`, `struct BytePair`).
So an NFD merge key used to match NFC text and vice versa, and two
normalization variants of one merge collapsed onto a single dictionary slot
(last write wins).

The `tokensToIds` side is genuinely equivalent: those keys are `NSString`,
which compares UTF-16 code units, i.e. byte-exact already.

Byte-exact is the **intended** semantics — it is what
`huggingface/tokenizers` and `tiktoken` do, the merge table is a byte-level
artifact, and normalization belongs to the normalizer, not the BPE inner
loop. It is unobservable on byte-level BPE vocabs (the GPT-2 byte→unicode map
emits no combining marks, so no vocab key has a distinct normalization form),
which is why the C24 gate measured 88/88 byte-identical items over 6.7M tokens
on both PARO tokenizers. It **can** change output on a non-byte-level BPE
vocab carrying mixed normalization forms — there it is a fix, and the upstream
PR must say so rather than presenting the carry as a pure refactor. Upstream
has already fixed two Unicode bugs in this exact code (`#352` Bugs 3 and 4),
so expect the question.

## Pin state

`Vendor/tesseract-speech`'s `Package.swift` and `Package.resolved` pin
`c99afdf6866817284af2061054557fca885a4fda` — the head of
`pin-tesseract-2026-09-15`, so the app tree resolves and builds the three
carries left on the 1.3.4 base. Appending to the pin branch does
not move the app: the pin is an exact revision, and every append needs this same
two-file edit.

Moved 2026-07-27 (`a524093` → `0033bc7`) with the workflow's verification steps
done: re-resolved, the DerivedData checkout's `1.3.3..HEAD` diff confirmed
byte-identical to the accepted fork working-copy diff (486 lines), and the
mandatory gate leg re-run against the built tree —
`--tokenize-cache-bench` PASS on both PARO models with 0 token mismatches / 0
parity failures / 0 path failures, and `--prefix-cache-e2e` PASS 32/32. See
`benchmarks/experiments-ledger.md` → "Review round 2026-07-27". The clean-build
confirmation is CI's `build-release` on a pristine runner rather than a local
`dev.sh clean`, so the timing legs above were measured on a thermally quiet
machine.

Moved 2026-09-06 (`0033bc7` → `08933b6`): re-resolved, the DerivedData checkout
confirmed at the new head with a diff against `0033bc7` equal to the fork commit,
and the replay gate (three ~3k-token tool prompts with emoji, dashes and slashes,
reconstructed from the 2026-09-06 Pi session) re-run against the built tree with
every previously divergent `write` turn stored live and zero duplicated graphemes.

Moved 2026-10-28 (`fe95f0a` → `c99afdf`, ADR-0098): resolved locally from the
fork working copy before the push, the DerivedData checkout confirmed clean at
`c99afdf`, the whole app test target green on it with `ToolPromptParityTests`
token-identical to transformers 5.21 on Qwen3.5, Qwen3.8, LFM3 and LFM3-Audio,
and the fork's `ChatTemplateTests` passing (16 of 16). `--tokenize-cache-bench`
PASS on Qwen3.5-4B-PARO and Qwen3.8-27B-PARO with 0 token mismatches / 0 parity
failures / 0 path failures; a render costs about 0.3 ms more with the bench's 40
tools. On Qwen3.8 the C25 hit path is slower than the fused baseline (41 ms
against 37 ms per hit turn); it was before the move too (43 against 35 on a
Release build of `2e070312`). Pushed the same day; a fresh resolve from GitHub,
with no dependency cache, checks out `c99afdf`.

## Per-iteration workflow

Same shape as `docs/mlx-core-fork.md`: edit the DerivedData checkout
(`chmod u+w` first) for the fast loop; on ACCEPT port the diff verbatim
to `~/projects/swift-transformers` (`pin-tesseract`), commit
(Conventional Commits), push, move the `Vendor/tesseract-speech`
Package.swift pin, commit in tesseract with the ledger entry, re-resolve
and verify the checkout diff equals the accepted diff, then the
mandatory clean-build confirmation. On REJECT: `git checkout -- .` in
the DerivedData checkout (+ `git clean -fd`).

Working copy: `~/projects/swift-transformers` (remote `upstream` =
huggingface/swift-transformers), branch `pin-tesseract-2026-09-15`.
