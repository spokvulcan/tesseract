# SSD read experiment #532

Owner plan: approved-plan.json. Model: qwen3.8-27b.
Hardware: Mac15,9, 48GB. Revision: 440acfca9c61fde7c9a39660c2137aee736db8d7.

Six balanced blocks, target loaded once, OS page cache warmed; throughput includes MLX evaluation.

| Arm | Median GB/s (decimal, materialized bytes) | Ratio to mapped |
| --- | ---: | ---: |
| mapped | 5.128 | 1.000 |
| sequentialMap | 5.595 | 1.091 |
| positional | 3.738 | 0.729 |

Qualifying arms at the pre-registered >=2x gate: [].
Owner must assess memory/RSS, validity, and representative cache conditions before adoption. Production still uses mapped.
