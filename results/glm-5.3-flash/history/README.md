# Intermediate steps (27 September 2026)

Runs of intermediate builds of the engine/ branch, kept for the record. Not in the leaderboard.

| Run | Build |
| --- | --- |
| 123622 | 16-bit absorb, default (2,051-token) context: showed long-context mode costs nothing at short prompts |
| 130002, 130051 | first 4-bit absorb kernels (rows looped in one program): faster steps, prefill 443 tok/s |
| 132423 | 4-bit kernels with one program per row, then row blocks |

The final build of that day is `../20260927-144257-*` (4-bit absorb, 1,024-token prefill, visible-pool selection).
