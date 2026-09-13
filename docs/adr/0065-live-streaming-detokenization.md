---
status: accepted
---

# Live streaming detokenization: preserve delivery while optimizing compatible decoders

Design for [#487](https://github.com/spokvulcan/tesseract/issues/487).
The complete design and agent brief were accepted on 2026-09-13.

The live path must preserve the naive detokenizer's ordered chunk bytes and
release each chunk on the same token step. There is zero additional token
delay, including for newline-free tool calls. Existing incomplete-Unicode
withholding remains part of that reference behavior; this is not a
wall-clock delivery deadline. ADR-0020's streamed Argument Fragments cannot
be retracted or corrected after emission.

Only decoders whose compatibility can be established need an optimized live
path in this issue. Unknown and non-byte decoders may keep naive live
streaming, with identical behavior and its existing cost. The verified
window path remains available for fidelity replay. This deliberately
narrows the issue's proposed live window-path optimization; tokenizer
support is retained.

The existing linear implementation buffers a segment until newline or end
of stream, verifies its decode, and can replace its pending chunks with
naive recomputation. Individual token byte checks do not justify early
release: a decoder that collapses doubled spaces passes those checks but
produces a different chunk for the second space. A later verification
cannot repair already-emitted bytes. Live optimization therefore requires
an eligibility contract stronger than the current individual-token probe.

Eligibility comes from recognizing the effective decoder configuration
used to construct the tokenizer, at model load. The initial supported
profile is top-level ByteLevel decoding with tokenization-space cleanup
disabled. Its added-token boundaries and literal rendering must be
preserved, including their effect on incomplete UTF-8 sequences. The
contract describes the actual live decode mode, which retains special
tokens. Unsupported or unknown configurations use naive live streaming.
Model names, an allowlist of model families, and finite sample probes are
not evidence sufficient to grant eligibility.

The optimized live path does not perform an additional full-segment decode
audit. Eligibility is established before streaming and exact chunk bytes
and release steps are checked in tests. The existing verified fidelity
replay remains in place for eligible server turns, including its window
path and naive recomputation. This avoids duplicating segment verification
in live generation where detecting a mismatch could not repair the output
already sent.
