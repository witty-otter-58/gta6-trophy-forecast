#!/usr/bin/env python3
# GTA VI trophy rarity forecaster (Python 3.13, standard library only).
#
# The model, in short. An Exophase earn rate is a ratio, earners over players
# tracked, and both sides move: players join in waves (which dilutes every
# rate) while the players already tracked keep earning trophies (which lifts
# them). So instead of holding the last rate, this program models the two
# sides separately and recombines them:
#
#   * Time is warped as x = ln(1 + hours since the game's release). Launch
#     growth is fast at first and then flattens, and this warp makes both
#     the player count and the earners per trophy roughly straight lines in x.
#   * For every trophy it fits, by least squares, a line to logit(rate)
#     against x, and a second line to log(earners) against x, where earners
#     are rebuilt from the printed rate times the players tracked. It also
#     fits a line to log(players tracked) against x. The median forecast is
#     the average, on the logit scale, of the direct rate line and the
#     earners-over-players line read at the target. Averaging the two hedges
#     the case where one of them is fooled (a flat rate line when the player
#     base is about to jump, or a noisy earner count for a rare trophy).
#     Slopes are capped so a steep launch-night line cannot extrapolate to
#     an absurd rate, and the median is capped per tier (platinum lowest),
#     because a trophy nobody can have earned yet cannot be at 90% in a week.
#   * The spread is honest about what a 4.5-day horizon hides. It combines
#     a random-walk term that grows with the square root of the horizon, the
#     prediction-interval width of each fit (residual scatter times the
#     leverage of extrapolating past the last read), and a staleness term:
#     Exophase refreshes its counts on its own schedule, not with every
#     read, so a page that has sat unchanged for hours is overdue a refresh
#     that will move every rate at once. When the case history contains no
#     refresh at all -- every read identical, as on launch night -- the
#     staleness term is large and the median is also pulled down by a
#     first-refresh dilution: the tracked base roughly quadruples when the
#     launch wave first syncs, so a rate near 1% is expected near 0.25%
#     after it. A trophy already near 100% is pulled down far less, because
#     new players earn an early trophy almost immediately. Once refreshes
#     have been seen, the fits carry the dilution themselves and the
#     staleness term shrinks to a small guard for a refresh that is late.
#   * One snapshot, one point. Between refreshes the house re-reads the
#     same frozen page, so a run of identical reads is a single snapshot
#     repeated, not many independent observations. The fits below use
#     each distinct snapshot once (its first read), or a long frozen
#     stretch would drag every line flat and fake a tiny residual.
#
# The program reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and
# writes one <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output). It is
# deterministic: pure arithmetic, no randomness, no network, no clock reads
# beyond the case file itself, so two runs write byte-identical outputs.

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Spread of a rate's move, in logit units per square-root day (the house
# baseline's random-walk rate; kept, and added to rather than replaced).
SIGMA_RW = 0.5
# Hardest cap on any fitted slope, in logit (or log-count) units per unit of
# warped time. A launch-night line steeper than this is noise, not signal.
SLOPE_CAP = 2.0
SLOPE_CAP_COUNTS = 2.5
# A plausible week-one ceiling per tier, in percent. Platinum needs the whole
# game finished; bronze holds the early story trophies that saturate fast.
TIER_CEILING = {"platinum": 20.0, "gold": 50.0, "silver": 70.0,
                "bronze": 99.99}
# First-refresh dilution: the tracked base is assumed to roughly quadruple
# when a page that has never refreshed finally does (launch wave syncing).
FIRST_REFRESH_FACTOR = 0.25
# Staleness spread: logit units per square-root hour of a page sitting
# unchanged. Large while no refresh has ever been seen, small once they have.
STALE_UNSEEN = 0.8
STALE_SEEN = 0.25
STALE_SEEN_CAP = 0.5
STALE_UNSEEN_CAP = 2.0
# The spread never goes below this, in logit units.
SPREAD_FLOOR = 0.10
SPREAD_CAP = 3.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_from_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    return round(x, 4)


def parse_time(s):
    """ISO 8601 timestamp (with Z or an offset) to an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_release(game):
    """Release date from the case ('2026-11-19') as midnight UTC."""
    raw = (game or {}).get("release_date") or ""
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, xbar, sxx, n).

    Degenerate input (one distinct x, or a single point) yields a flat line
    through the mean with zero residual scatter.
    """
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0.0:
        return (ybar, 0.0, 0.0, xbar, 0.0, n)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    if n > 2:
        resid = math.sqrt(
            sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return (a, b, resid, xbar, sxx, n)


def pred_se(resid_sd, xbar, sxx, n, x_target):
    """Prediction-interval half-width factor for a new point at x_target."""
    if n < 3 or sxx <= 0.0:
        return 0.0
    leverage = 1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx
    return resid_sd * math.sqrt(leverage)


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case %s has no history" % case.get("id"))

    release = parse_release(case.get("game"))
    if release is None:
        release = parse_time(history[0]["at"])
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])

    def warped(t):
        hours = (t - release).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    xs = [warped(parse_time(h["at"])) for h in history]
    x_target = warped(target)
    horizon_days = case["horizon_minutes"] / 1440.0

    # Has Exophase refreshed at all inside this case? A refresh shows up as
    # the players-tracked count moving between reads.
    players = [h["players_tracked"] for h in history]
    refresh_seen = any(p != players[0] for p in players[1:])
    if refresh_seen:
        last_change = 0
        for j in range(1, len(players)):
            if players[j] != players[j - 1]:
                last_change = j
        stale_hours = (checkpoint - parse_time(
            history[last_change]["at"])).total_seconds() / 3600.0
        stale_spread = min(STALE_SEEN_CAP,
                           STALE_SEEN * math.sqrt(max(0.0, stale_hours)))
    else:
        stale_hours = (checkpoint - parse_time(
            history[0]["at"])).total_seconds() / 3600.0
        stale_spread = min(STALE_UNSEEN_CAP,
                           STALE_UNSEEN * math.sqrt(max(0.0, stale_hours)))

    # Collapse runs of identical reads into one snapshot each (first read
    # of the run): between refreshes the page is frozen, so the repeats
    # carry no new information. The fits use snapshots; the staleness and
    # last-rate logic above still use the full history.
    snaps = []
    prev_key = None
    for x, h in zip(xs, history):
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev_key:
            snaps.append((x, h["players_tracked"], h["rates"]))
            prev_key = key
    snap_xs = [s[0] for s in snaps]
    snap_players = [s[1] for s in snaps]
    snap_rates = [s[2] for s in snaps]

    # The player base's own line: log(players) against warped time.
    log_p = [math.log(max(1, p)) for p in snap_players]
    a_p, b_p, sd_p, xbar_p, sxx_p, n_p = ols(snap_xs, log_p)
    b_p = max(-SLOPE_CAP_COUNTS, min(SLOPE_CAP_COUNTS, b_p))
    log_p_target = a_p + b_p * x_target
    dil_se = pred_se(sd_p, xbar_p, sxx_p, n_p, x_target)

    rw_spread = SIGMA_RW * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        series = [(x, rates[i]) for x, rates in zip(snap_xs, snap_rates)
                  if rates[i] is not None]
        last_rate = history[-1]["rates"][i]
        if last_rate is None:
            last_rate = series[-1][1] if series else 0.0

        med_logit = None
        fit_se = 0.0
        if len(series) >= 2:
            # Direct line: logit(rate) against warped time.
            a_r, b_r, sd_r, xbar_r, sxx_r, n_r = ols(
                [x for x, _ in series],
                [logit(r) for _, r in series])
            b_r = max(-SLOPE_CAP, min(SLOPE_CAP, b_r))
            direct = a_r + b_r * x_target
            fit_se = pred_se(sd_r, xbar_r, sxx_r, n_r, x_target)
            # Earners line: log(rate * players / 100) against warped time,
            # recombined with the player line at the target.
            pairs = [(x, rates[i], p) for x, p, rates in
                     zip(snap_xs, snap_players, snap_rates)
                     if rates[i] is not None]
            a_e, b_e, _sd_e, _xb_e, _sxx_e, _n_e = ols(
                [x for x, _r, _p in pairs],
                [math.log(max(0.5, r / 100.0 * p))
                 for _x, r, p in pairs])
            b_e = max(-SLOPE_CAP_COUNTS, min(SLOPE_CAP_COUNTS, b_e))
            log_e_target = a_e + b_e * x_target
            rate_ep = 100.0 * math.exp(log_e_target - log_p_target)
            med_logit = 0.5 * (direct + logit(rate_ep))
        if med_logit is None:
            med_logit = logit(last_rate)

        median = pct_from_logit(med_logit)

        # No refresh has ever been seen: the page is a stale snapshot and
        # the next refresh dilutes every rate as the launch wave syncs in.
        # Near-saturated trophies dilute least (new players earn them fast).
        if not refresh_seen:
            dilute = FIRST_REFRESH_FACTOR + (
                1.0 - FIRST_REFRESH_FACTOR) * (last_rate / 100.0)
            median = median * dilute

        # Guards: per-tier ceiling, and never let a steep line carry a
        # trophy orders of magnitude past where it was last seen.
        ceiling = TIER_CEILING.get(trophy.get("tier"), 99.99)
        median = min(median, ceiling)
        if last_rate > 0:
            median = min(median, last_rate * 60.0 + 5.0)
        median = clamp_pct(median)

        spread = math.sqrt(rw_spread ** 2 + fit_se ** 2 + dil_se ** 2
                           + stale_spread ** 2)
        spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))

        centre = logit(median)
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(pct_from_logit(centre + z * spread)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[:-len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r",
                  encoding="utf-8") as fh:
            case = json.load(fh)
        if list(case.get("levels", [])) != LEVELS:
            raise ValueError("%s asks for levels %r, not %r"
                             % (name, case.get("levels"), LEVELS))
        out = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
