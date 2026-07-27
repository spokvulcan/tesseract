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
