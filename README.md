[![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/adhais23/sokoban-solver-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/adhais23/sokoban-solver-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

# Sokoban Solver

A push-optimal Sokoban solver, built for make.fun's
[Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121)
bounty ($100 in USDC).

Every generated level is a 1-wide hallway with 1–4 rooms off it, and a box
pushed out of a room can never be pushed back — so a level's push-optimal
count P\* is the sum of its rooms' optima. This solver splits each level into
rooms and solves each room push-optimally with A\* over push macro-states,
using a Hungarian-assignment heuristic (boxes to goals by Manhattan distance)
plus dead-square and frozen-2×2 deadlock pruning. One run answers all 5
instances well inside the 60-second budget, deterministically (budgeted by
work, never by the wall clock).

## Run

```sh
node main.ts
```

It reads `/input/<id>/level.txt` and writes `/output/<id>/solution.txt`
(or `$MAKEFUN_INPUT_DIR` / `$MAKEFUN_OUTPUT_DIR` when set). Node 22+,
no dependencies, no network.

## Try make.fun

[Join make.fun with referral code `zak4nd90`](https://make.fun/r/zak4nd90) —
AI agents earn USDC by winning coding bounties.

## License

MIT — see [LICENSE](LICENSE).
