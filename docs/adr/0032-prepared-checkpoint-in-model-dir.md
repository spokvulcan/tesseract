# ADR-0032: Prepared Checkpoint — persisted PARO conversion beside the checkpoint

- Status: Accepted
- Date: 2026-07-11
- Relates to: ADR-0006 (vendored mlx-swift-lm), PRD #72 / `ModelFingerprint` (weight identity)

## Context

Every load of a PARO checkpoint re-runs the AutoAWQ→MLX conversion
(unpack/reorder/repack, Mamba split, MoE expert stacking, rotation-key
remap). Measured on the 35B-A3B MoE (Release, Mac15,9 48GB): the load
cost 41–43 s, of which ~29 s was a quadratic key/prefix string-matching
loop in `convertAutoAWQ` — fixed outright by an O(1) suffix-strip
matcher, bringing the load to 17.7 s. The equivalent-size native MLX
checkpoint (unsloth 35B, no conversion) loads in 6.4 s: the remaining
~11 s gap is lazy-graph construction for ~12K per-expert tensors (5.2 s)
plus conversion kernels inside the final `eval`. Loads repeat on first
use after launch, model switch, vision-mode upgrade, and after manual
offload.

PARO's correctness hinges on runtime activation rotations: `theta` /
`pairs` / `channel_scales` ship in the checkpoint, load verbatim, and
every rotation-derived value is recomputed at load
(`prepareDerivedRotationState()`); the forward pass is
`quantizedMM(rotate(x), W)`. Any artifact that baked rotation state
would put the quantization scheme itself at risk.

## Decision

Persist the converted weights dict as a **Prepared Checkpoint** —
`prepared_checkpoint.safetensors` written into the model directory by
the vendored loader after the first (converting) load completes, in the
background (temp file + atomic rename), and loaded in place of raw
sources + conversion on later loads.

- **Snapshot point: post-convert/stack/remap, pre-sanitize.** The last
  container-agnostic point — one artifact serves both the LLM and VLM
  containers; sanitize, layer patching, quantize passes, `update`, and
  rotation-state derivation stay live code on every load. Rotation
  tensors are stored byte-for-byte as the checkpoint shipped them;
  nothing semantic is baked in. The maximal alternative (fully-final
  parameters) was rejected: it saves a measured ~0.2 s and ties the
  artifact to model-class behavior.
- **In the model dir, vendored-owned.** Self-contained in
  `MLXLMCommon/ParoQuant/` (upstreamable), zero lifecycle plumbing —
  the artifact dies with the model directory. Rejected: an app-owned
  external cache dir (lifecycle plumbing, loader API changes) and
  replacing the originals (destroys re-conversion and HF-repo parity).
- **Weight-identity contract.** `ModelFingerprint` hashes every
  `*.safetensors` in the directory; the artifact names
  (`ParoQuantPreparedCheckpoint.excludedFileNames`) are excluded on
  both sides — loader scan and fingerprint — otherwise the first
  background write would shift `CachePartitionKey` and orphan every
  persisted prefix-cache snapshot for the model. This is a deliberate
  two-place magic-filename contract; the vendored constant is the
  single source of truth.
- **Invalidation: source manifest + format version.** Safetensors
  metadata records (name, size, mtime-ns) of every conversion source
  plus a `formatVersion` bumped on conversion-semantics changes.
  Byte-hashing is deliberately skipped (same rationale as
  `ModelFingerprint`): it would cost seconds on exactly the path the
  artifact exists to speed up.
- **Self-heal, never fail.** Absent/stale/corrupt artifacts are deleted
  and the loader falls back to full conversion plus a background
  rewrite; a failed write only means the next load converts again. The
  artifact can never fail a load that would otherwise succeed.

## Consequences

- PARO loads drop to near the native floor (target ~8 s for the 35B);
  the first load after download/invalidation still pays full conversion
  plus a one-time ~19 GB background write per model.
- Disk cost ≈ one extra checkpoint per PARO model. Writes are skipped
  below a free-space floor (2× artifact size).
- Acceptance gate: bit-exact artifact round-trip (unit) and
  fresh-vs-prepared greedy token parity on the 35B
  (`--prepared-checkpoint-parity` harness), which also asserts the
  prepared path was actually taken.
