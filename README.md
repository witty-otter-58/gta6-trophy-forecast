# GTA VI trophy rarity forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/witty-otter-58/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/witty-otter-58/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy. Built for
make.fun's `gta6-trophy-rarity-forecast-01` bounty ($250 in USDC).

## The model

An Exophase earn rate is a ratio — earners over players tracked — and both
sides move. Players join in waves (which dilutes every rate) while the players
already tracked keep earning trophies (which lifts them). Holding the last
rate, as the house baseline does, ignores both. This program models the two
sides separately and recombines them:

- **Warped time.** `x = ln(1 + hours since release)`. Launch growth is fast
  then flattens, and this warp makes both the player count and the earners
  per trophy roughly straight lines in `x`.
- **Two lines per trophy.** A least-squares line to `logit(rate)` against `x`,
  and a second line to `log(earners)` against `x` (earners rebuilt from the
  printed rate × players tracked), recombined with a fitted line to
  `log(players tracked)`. The median forecast is the average, on the logit
  scale, of the direct rate line and the earners-over-players line at the
  target. Slopes are capped and the median is capped per tier (platinum
  lowest), so a steep launch-night line cannot extrapolate to an absurd rate.
- **Honest spread.** The spread combines a random-walk term growing with the
  square root of the horizon, the prediction-interval width of each fit
  (residual scatter × extrapolation leverage), and a **staleness** term:
  Exophase refreshes on its own schedule, not with every read, so a page
  that has sat unchanged for hours is overdue a refresh that moves every
  rate at once. When a case's history contains no refresh at all — every
  read identical, as on launch night — the staleness term is large and the
  median is also pulled down by a first-refresh dilution (the tracked base
  roughly quadruples when the launch wave first syncs). A trophy already
  near 100% is pulled down far less, because new players earn an early
  trophy almost immediately. Once refreshes have been seen, the fits carry
  the dilution themselves and staleness shrinks to a small guard.

## Run it

Python 3.13, standard library only, no network, deterministic (pure
arithmetic — two runs write byte-identical outputs).

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out`
per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Practice check

On the kit's practice cases (`p00`, `p01` — checkpoints 03:00/03:30 UTC on
19 Nov, both straddling Exophase's first refresh, which diluted every rate
as players tracked jumped 638 → 3,868) the program's median sits well below
the last rate and its 90% band is wide enough to cover the refresh, where
the house baseline's narrow band misses it entirely. See `main.py` for the
model; the numbers are reproducible from the kit's `input/`.
