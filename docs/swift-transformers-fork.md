# swift-transformers fork & pin scheme

How tesseract consumes a writable `huggingface/swift-transformers` — the
third fork level, established 2026-07-27 for the render+token cache
(experiments-ledger C25). Same rules as the mlx/mlx-swift forks
(`docs/mlx-core-fork.md`): pin branches are append-only, changes must be
general and upstreamable, exact-revision pins keep history reproducible.

## The pin chain

```
tesseract.xcodeproj
  └─ Vendor/mlx-audio-swift/Package.swift — the package's ONLY declarer
     in the app graph (MLXLMCommon's `import Tokenizers` resolves through
     the Xcode workspace module leak, same as MLXFast):
       url: https://github.com/spokvulcan/swift-transformers
       revision: <exact commit on pin-tesseract>
  └─ spokvulcan/swift-transformers @ pin-tesseract
       = huggingface/swift-transformers @ 1.3.3 (the tag the graph
         resolved) + one commit per accepted carry
```

## Carries on `pin-tesseract`

| Commit | What it does | Upstream status |
| --- | --- | --- |
| `63edf42` `feat(tokenizers): expose renderChatTemplate` | Splits the render half of `applyChatTemplate` into a public `renderChatTemplate` (pure refactor, byte-identical output; public default impl keeps third-party conformers compiling). Enables the C25 render+token cache | Not filed (queued — owner go-ahead) |
| `a524093` `perf(tokenizers): byte-native BPE inner loop + byte-keyed lookup tables` | Same serial algorithm, same merge order, byte-identical output on byte-level BPE vocabs: byte-range symbols in one UTF-8 buffer instead of per-scalar Strings and per-merge concats; open-addressed byte-keyed rank/id tables (FNV-1a + full byte-compare) instead of String-keyed dictionary probes. 1.22×/1.21×/1.20× encode at 32K/8K/128; 88/88 corpus items byte-identical (experiments-ledger C24). **Narrows the merge-table match semantics — see the semantics note below; the commit message's "byte-identical output" claim is scoped to byte-level vocabs.** Adds ~20 MB resident per loaded tokenizer (the byte tables sit alongside `bpeRanks`/`tokensToIds`, both still live) | Not filed (queued — owner go-ahead) |
| `0033bc7` `fix(tokenizers): C24 review round — eager byte tables, correct semantics note` | Replaces the lazy double-checked-locked `byteTablesCache` with a `let` built in `init` (the unlocked fast-path read had no acquire semantics — a reader on a weakly-ordered core could see the published pointer before the table's array buffers); corrects the `BytePairTables` doc comment, which claimed byte-exactness was equivalent to what it replaced; bounds the leading-byte scalar-width walk against a malformed buffer. No behavior change on well-formed input | Not filed (queued — owner go-ahead) |

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

## Pending — the pin has not moved

`pin-tesseract` is at `0033bc7` (pushed), but `Vendor/mlx-audio-swift`'s
`Package.swift` and `Package.resolved` still pin `a524093`, so the app tree
resolves and builds the pre-review-round fork. Appending to the pin branch
cannot change that: the pin is an exact revision.

What remains, per the per-iteration workflow below: move the
`Vendor/mlx-audio-swift` pin to `0033bc7`, re-resolve, verify the checkout diff
equals the accepted diff, then the mandatory clean-build confirmation + parity
smoke leg. Deliberately left to the owner — moving the pin invalidates the
current clean-build confirmation, and the gate is a Release build on a quiet
machine.

## Per-iteration workflow

Same shape as `docs/mlx-core-fork.md`: edit the DerivedData checkout
(`chmod u+w` first) for the fast loop; on ACCEPT port the diff verbatim
to `~/projects/swift-transformers` (`pin-tesseract`), commit
(Conventional Commits), push, move the `Vendor/mlx-audio-swift`
Package.swift pin, commit in tesseract with the ledger entry, re-resolve
and verify the checkout diff equals the accepted diff, then the
mandatory clean-build confirmation. On REJECT: `git checkout -- .` in
the DerivedData checkout (+ `git clean -fd`).

Working copy: `~/projects/swift-transformers` (remote `upstream` =
huggingface/swift-transformers), branch `pin-tesseract`.
