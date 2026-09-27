# Third-party code in tesseract-speech

## Qwen3-TTS inference (`Sources/Qwen3TTS`)

The Qwen3-TTS model code started as a copy of the Qwen3-TTS model in
[Blaizzy/mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift), tag
**v0.1.3** (`d302a5c6080d2bb97bae38c7418f82abb76013b6`, 2026-07-09), with the
Tesseract patches listed in the retired `Vendor/mlx-audio-swift/TESSERACT-PATCHES.md`
(in git history). Since ADR-0071 it is maintained here as first-party code;
upstream is a reference, not a sync source.

Kept from upstream, with changes: the talker, the code predictor, the model
configuration, the speech tokenizer's decoder, and the VoiceDesign/CustomVoice
generation path. Left out: every other model family, the speech tokenizer's
encoder, the speaker encoder, and voice cloning from reference audio.

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
