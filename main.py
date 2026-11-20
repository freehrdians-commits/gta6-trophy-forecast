"""Forecast GTA VI trophy earn rates on Exophase at the case target.

Model (written from scratch, standard library only):
  For each trophy, take the case history's distinct states (a read only
  counts when the page actually refreshed: same players and rates as the
  previous read is the same stale page, not new information).  Fit a
  recency-weighted line of logit(rate) against ln(1 + hours since the
  first read), cap the slope so a launch-night rush cannot extrapolate
  forever, and read the line at the target for the median.  The spread
  combines the random-walk term (0.5 * sqrt(horizon in days)), the fit
  residual, and the slope's uncertainty times the extrapolation distance,
  so a long or noisy forecast is honestly wide.  Quantiles are the normal
  quantiles of that logit distribution, mapped back to percent.

  When the history is flat (fewer than 3 distinct states, as on launch
  night before the first refresh), no trend can be fit: the median is the
  last rate shifted down by 0.7 logit units -- a refresh during launch
  grows the tracked-player denominator faster than earners, so every rate
  but the one at 100% falls -- with a wide spread (1.2) for the unknown
  size of that drop.

  A trophy at 100% in the last read stays at 100%: every quantile is
  99.99 (the scoring clamp).

  Deterministic: no randomness, no wall clock, no network.  Reads every
  *.in file in the input directory (name order) and writes one .out per
  case, exactly {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
# Entry contract (gta6-trophy-rarity-forecast-01): main.py is the only
# entrypoint; it reads every *.in in the input dir and writes one .out
# per case, always valid, deterministic, under 8 seconds per case.

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Random-walk spread of a rate's move, logit units per sqrt day.
SIGMA_RW = 0.5
# Hard cap on the fitted trend slope (logit per ln-hour).
SLOPE_CAP = 1.5
# Recency half-life for the trend fit, hours.
HALF_LIFE_H = 24.0
# Fallback (flat history): median shift down and spread, logit units.
FLAT_SHIFT = 0.7
FLAT_SIGMA = 1.2
# Spread clamp, logit units.
SIGMA_MIN = 0.35
SIGMA_MAX = 3.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up; a whole number comes back as int."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> float:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.timestamp()


def distinct_states(history):
    """History reads, deduplicated to states where the page changed.

    A read whose players and rates match the previous state is the same
    stale page, not new information.  Keeps the first time each state
    was seen.  Returns [(epoch, players, rates_tuple)].
    """
    states = []
    seen = set()
    for h in history:
        rates = tuple(h["rates"])
        key = (h["players_tracked"], rates)
        if key in seen:
            continue
        seen.add(key)
        states.append((parse_time(h["at"]), h["players_tracked"], rates))
    return states


def fit_trend(pts):
    """Weighted least squares y = a + b*x.  pts: [(x, y, weight)].

    Returns (a, b, resid_var, x_mean, sxx, sw) or None when degenerate.
    """
    sw = sum(w for _, _, w in pts)
    if sw <= 0:
        return None
    x_mean = sum(w * x for x, _, w in pts) / sw
    y_mean = sum(w * y for _, y, w in pts) / sw
    sxx = sum(w * (x - x_mean) ** 2 for x, _, w in pts)
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - x_mean) * (y - y_mean) for x, y, w in pts)
    b = sxy / sxx
    a = y_mean - b * x_mean
    resid = sum(w * (y - (a + b * x)) ** 2 for x, y, w in pts) / sw
    return a, b, resid, x_mean, sxx, sw


def forecast_trophy(pts, last_rate, target_x, horizon_days):
    """pts: [(x, logit, epoch)] distinct-state points for one trophy.

    Returns (median_logit, sigma).
    """
    last_logit = logit(last_rate)
    if len(pts) < 3:
        # Flat history: no trend to fit.  Anticipate the refresh drop.
        return last_logit - FLAT_SHIFT, FLAT_SIGMA
    t_last = pts[-1][2]
    weighted = []
    for x, y, t in pts:
        age_h = max(0.0, (t_last - t) / 3600.0)
        w = 0.5 ** (age_h / HALF_LIFE_H)
        weighted.append((x, y, w))
    fit = fit_trend(weighted)
    if fit is None:
        return last_logit - FLAT_SHIFT, FLAT_SIGMA
    a, b, resid, x_mean, sxx, sw = fit
    b = max(-SLOPE_CAP, min(SLOPE_CAP, b))
    med = a + b * target_x
    # Spread: random walk + fit residual + slope uncertainty over the
    # extrapolation distance.
    resid_var = max(resid, 0.0)
    slope_var = resid_var / sxx if sxx > 0 else 0.0
    # Guard: with very few points the residual underestimates error.
    if len(pts) < 6:
        slope_var = max(slope_var, 0.04)
    extrap = (target_x - x_mean) ** 2 * slope_var
    sigma = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2 + resid_var + extrap)
    sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
    return med, sigma


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    states = distinct_states(case["history"])
    if not states:
        raise ValueError(f"{name} has no usable history")
    t0 = states[0][0]
    target_x = math.log1p(max(0.0, (parse_time(case["target"]) - t0) / 3600.0))
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = states[-1][2][i]
        if last_rate is None:
            last_rate = 0.0
        if last_rate >= 99.99:
            # Everybody tracked has it; it stays earned by everybody.
            rows.append({"id": trophy["id"], "q": [99.99, 99.99, 99.99, 99.99, 99.99]})
            continue
        pts = []
        for (t, _n, rates) in states:
            r = rates[i]
            if r is None:
                continue
            hours = max(0.0, (t - t0) / 3600.0)
            pts.append((math.log1p(hours), logit(r), t))
        med, sigma = forecast_trophy(pts, last_rate, target_x, horizon_days)
        qs = [round4(clamp(logistic(med + z * sigma))) for z in Z]
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
