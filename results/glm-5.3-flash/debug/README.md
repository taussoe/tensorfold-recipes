# Debug runs (27 September 2026)

Runs from builds with two bugs, kept as the record of how they were found. Not in the leaderboard.

| Run | Build | What was wrong |
| --- | --- | --- |
| 112815 | prefill 1024, MoE member tile 64 | tile 64 changed a row's bits: the greedy 32k reply differed from 64-row prefill |
| 113706, 113918, 115016 | prefill 1024, tile 32 | bits fixed, but prefill fed the MTP head stale final-norm rows (logits skipped): draft acceptance fell, decode 10-22% slower |
| 115624 | same code, prefill 64 | A/B: still slow, so not the chunk size |
| 120134 | the first latent build (297c3a9) | A/B: fast, so a code change; traced to `last_logits` |

Fixes: `qmm.EXACT_MEMBER_TILES = (16, 32)` and `decode.last_logits` normalizing every row. The final run is
`../20260927-121318-*.json`.
