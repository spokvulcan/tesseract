# Third-party code in tesseract-speech

## Qwen3-TTS inference (`Sources/Qwen3TTS`)

The Qwen3-TTS model code started as a copy of the Qwen3-TTS model in
[Blaizzy/mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift), tag
**v0.1.3** (`d302a5c6080d2bb97bae38c7418f82abb76013b6`, 2026-07-09), with the
Tesseract patches listed in the retired `Vendor/mlx-audio-swift/TESSERACT-PATCHES.md`
(in git history). Since ADR-0071 it is maintained here as first-party code;
upstream is a reference, not a sync source.

Kept from upstream, with changes: the talker, the code predictor, the model
configuration and the VoiceDesign/CustomVoice prompt. Since ADR-0074 the
codec decoder, the sampler and the frame loop are rewritten, against Qwen's
own implementation (below). Left out: every other model family, the speech
tokenizer's encoder, the speaker encoder, and voice cloning from reference
audio.

The upstream license applies to the code derived from it:

```
MIT License

Copyright (c) 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen's reference implementation

The codec decoder (its sliding-window transformer and exact streaming
form), the sampler's logits processors and the prompt layouts follow the
official Qwen3-TTS implementation, the `qwen_tts` package in
[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e`), and are checked against it
numerically (ADR-0074). No Python is included. The implementation is
Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team,
licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Core ML model format (`Sources/Qwen3TTS/NeuralEngine`)

`MLProgramBuilder` writes a Core ML ML Program package directly: the `Model`
and `MIL` protobuf messages by their field numbers, and weights in the MILBlob
v2 storage layout. It follows those formats as Apple's
[coremltools](https://github.com/apple/coremltools) defines them
(`mlmodel/format/Model.proto`, `MIL.proto`, `FeatureTypes.proto`,
`mlmodel/src/MILBlob/Blob/StorageFormat.hpp`). No coremltools code is included.
coremltools' license:

```
Copyright © 2020-2023, Apple Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder(s) nor the names of any
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
