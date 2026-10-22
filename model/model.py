#!/usr/bin/env python3
"""
Restaurant demand model.

Learns from agent/stats.json (per-dish POS actuals) joined with observed
weather (predict-agent/weather_cache.json) and produces a per-dish portion
recommendation for a target day.

Two layers:

  1. TOTAL UNITS  -- multiplicative decomposition
         total = level(date) * weekday_factor * weather_mult * uplift
     `level` is a robust trend through the whole series (the restaurant has
     been getting busier); `weekday_factor` is the robust median ratio of that
     weekday to the trend line.

  2. DISH MIX -- per-dish share of total units, regressed on
         log(share) ~ a + b*months + c*temp_z + d*wet
     fitted with ridge-regularised weighted least squares (weights: recency x
     sample size). This is the "what do people order when it's cold and wet"
     layer: c < 0 means the dish sells relatively better when it's cold,
     d > 0 means it sells relatively better when it rains.

Everything is pure stdlib so it runs anywhere python3 does.

Usage:
    python3 model.py --date 2026-09-18 --tmax 19.5 --precip-mm 4.2 \
        --reserved-guests 12 --event-factor 1.05
    python3 model.py --explain          # show what the model has learned
    python3 model.py --backtest         # leave-one-out accuracy check
"""

import argparse
import datetime as dt
import json
import math
import os
import statistics as stat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATS = os.path.join(ROOT, "agent", "stats.json")
WEATHER = os.path.join(HERE, "weather_cache.json")

# --- knobs -------------------------------------------------------------
HALF_LIFE_DAYS = 45.0      # recency half-life for sample weights
RIDGE = 2.0                # shrinkage on the mix regression (higher = flatter)
EXTRAPOLATE_MONTHS = 1.0   # never run a trend more than a month past the data
WET_MM = 2.0               # precipitation_sum above this counts as a wet day
TEMP_REF = 24.0            # temperature centre, degC
TEMP_SCALE = 5.0           # degC per unit of temp_z
BUFFER = 1.10              # safety buffer on final portions
MIN_SHARE_SAMPLES = 4      # below this, fall back to the flat mean share

DISH_LABEL = {
    "PLOV": "PLOV",
    "MANTY_MEAT": "MANTY meat",
    "MANTY_VEG": "MANTY veg",
    "SAMSA_BEEF": "SAMSA beef",
    "SAMSA_CHICKEN": "SAMSA chicken",
    "SAMSA_VEG": "SAMSA veg",
    "LAGMAN": "LAGMAN",
    "LAGMAN_SOUP": "LAGMAN soup",
}


# --- tiny linear algebra ----------------------------------------------
def solve(A, b):
    """Gaussian elimination with partial pivoting. A is n x n, b is n."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / p
            if f:
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def ridge_wls(X, y, w, lam, protect_intercept=True):
    """min sum w_i (y_i - x_i.beta)^2 + lam * ||beta_(not intercept)||^2"""
    p = len(X[0])
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for xi, yi, wi in zip(X, y, w):
        for j in range(p):
            b[j] += wi * xi[j] * yi
            for k in range(p):
                A[j][k] += wi * xi[j] * xi[k]
    for j in range(p):
        if protect_intercept and j == 0:
            continue
        A[j][j] += lam
    beta = solve(A, b)
    return beta if beta else [0.0] * p


# --- data --------------------------------------------------------------
def load():
    stats = json.load(open(STATS))
    weather = json.load(open(WEATHER)) if os.path.exists(WEATHER) else {}
    dishes = stats["dishes"]
    rows = []
    for weekday, samples in stats["samples"].items():
        for s in samples:
            if not s.get("total_units"):
                continue
            wx = weather.get(s["date"], {})
            rows.append({
                "date": s["date"],
                "d": dt.date.fromisoformat(s["date"]),
                "weekday": weekday,
                "total": float(s["total_units"]),
                "qty": {k: float(s.get(k, 0)) for k in dishes},
                "tmax": wx.get("tmax"),
                "precip": wx.get("precip_mm"),
                "has_wx": "tmax" in wx,
            })
    rows.sort(key=lambda r: r["d"])
    return dishes, rows


def temp_z(tmax):
    if tmax is None:
        return 0.0
    return (tmax - TEMP_REF) / TEMP_SCALE


def wet_flag(precip_mm):
    if precip_mm is None:
        return 0.0
    return 1.0 if precip_mm >= WET_MM else 0.0


def recency_w(row, ref):
    age = max((ref - row["d"]).days, 0)
    return 0.5 ** (age / HALF_LIFE_DAYS)


# --- layer 1: total units ---------------------------------------------
def fit_total(rows, ref):
    """Robust multiplicative decomposition: level trend x weekday factor."""
    t0 = rows[0]["d"]
    months = [ (r["d"] - t0).days / 30.4375 for r in rows ]
    logs = [ math.log(r["total"]) for r in rows ]

    wd_factor = {}
    a = b = 0.0
    for _ in range(12):
        # trend through log(total) - log(weekday factor), recency weighted
        X, y, w = [], [], []
        for r, m, lg in zip(rows, months, logs):
            X.append([1.0, m])
            y.append(lg - wd_factor.get(r["weekday"], 0.0))
            w.append(recency_w(r, ref))
        a, b = ridge_wls(X, y, w, 0.5)
        # weekday factors = median residual off that trend
        buckets = {}
        for r, m, lg in zip(rows, months, logs):
            buckets.setdefault(r["weekday"], []).append(lg - (a + b * m))
        new = {k: stat.median(v) for k, v in buckets.items()}
        if all(abs(new.get(k, 0) - wd_factor.get(k, 0)) < 1e-6 for k in new):
            wd_factor = new
            break
        wd_factor = new

    resid = []
    for r, m, lg in zip(rows, months, logs):
        resid.append(lg - (a + b * m) - wd_factor.get(r["weekday"], 0.0))
    sigma = stat.pstdev(resid) if len(resid) > 1 else 0.25

    # weather sensitivity of the TOTAL, estimated on those residuals
    X, y, w = [], [], []
    for r, e in zip(rows, resid):
        if not r["has_wx"]:
            continue
        X.append([1.0, temp_z(r["tmax"]), wet_flag(r["precip"])])
        y.append(e)
        w.append(recency_w(r, ref))
    tot_wx = ridge_wls(X, y, w, 6.0) if len(X) >= 6 else [0.0, 0.0, 0.0]

    return {
        "t0": t0, "a": a, "b": b, "wd_factor": wd_factor,
        "sigma": sigma, "pooled": stat.median(logs),
        "temp_coef": tot_wx[1], "wet_coef": tot_wx[2],
        "last_m": months[-1],
        "n_by_wd": {k: sum(1 for r in rows if r["weekday"] == k) for k in wd_factor},
    }


def clamp_months(m, last_m):
    """Never extrapolate the trend more than EXTRAPOLATE_MONTHS past the data."""
    return min(m, last_m + EXTRAPOLATE_MONTHS)


def predict_total(fit, target, weekday, tmax, precip):
    m = clamp_months((target - fit["t0"]).days / 30.4375, fit["last_m"])
    trend = fit["a"] + fit["b"] * m
    wd = fit["wd_factor"].get(weekday)
    known = wd is not None
    if not known:
        wd = 0.0
    wx = fit["temp_coef"] * temp_z(tmax) + fit["wet_coef"] * wet_flag(precip)
    return math.exp(trend + wd + wx), known, math.exp(wx)


# --- layer 2: dish mix -------------------------------------------------
def fit_mix(dishes, rows, ref):
    t0 = rows[0]["d"]
    out = {}
    usable = [r for r in rows if r["has_wx"]]
    for dish in dishes:
        shares = [(r, r["qty"][dish] / r["total"]) for r in usable]
        nonzero = sum(1 for _, s in shares if s > 0)
        flat = stat.mean([s for _, s in shares]) if shares else 0.0
        if nonzero < MIN_SHARE_SAMPLES:
            out[dish] = {"beta": None, "flat": flat, "n": nonzero, "sigma": 0.5}
            continue
        X, y, w = [], [], []
        for r, s in shares:
            s = max(s, 0.0015)          # floor so log() behaves on zero days
            X.append([1.0,
                      (r["d"] - t0).days / 30.4375,
                      temp_z(r["tmax"]),
                      wet_flag(r["precip"])])
            y.append(math.log(s))
            w.append(recency_w(r, ref) * math.sqrt(r["total"] / 250.0))
        beta = ridge_wls(X, y, w, RIDGE)
        resid = [yi - sum(bj * xj for bj, xj in zip(beta, xi))
                 for xi, yi in zip(X, y)]
        out[dish] = {
            "beta": beta, "flat": flat, "n": nonzero,
            "sigma": stat.pstdev(resid) if len(resid) > 1 else 0.4,
            "t0": t0,
            "last_m": (usable[-1]["d"] - t0).days / 30.4375,
        }
    return out


def predict_share(mix, dish, target, tmax, precip):
    m = mix[dish]
    if m["beta"] is None:
        return m["flat"], 0.0, 0.0
    b = m["beta"]
    months = clamp_months((target - m["t0"]).days / 30.4375, m["last_m"])
    tz, wf = temp_z(tmax), wet_flag(precip)
    logs = b[0] + b[1] * months + b[2] * tz + b[3] * wf
    # neutral-weather counterfactual, so we can report the weather delta
    neutral = b[0] + b[1] * months
    return math.exp(logs), math.exp(logs - neutral) - 1.0, b[2]


# --- public API --------------------------------------------------------
def forecast(target, tmax, precip, reserved_guests=0, event_factor=1.0,
             holiday=False, buffer=BUFFER):
    dishes, rows = load()
    ref = target
    ft = fit_total(rows, ref)
    mx = fit_mix(dishes, rows, ref)
    weekday = target.strftime("%A")

    base, wd_known, wx_mult = predict_total(ft, target, weekday, tmax, precip)

    resv = 1.0
    if reserved_guests >= 8:
        resv = 1.0 + min(0.01 * reserved_guests, 0.25)
    hol = 1.20 if holiday else 1.0
    total = base * resv * hol * event_factor

    per_dish, notes = {}, []
    for d in dishes:
        share, wx_delta, tcoef = predict_share(mx, d, target, tmax, precip)
        pt = total * share
        lo = pt * math.exp(-mx[d]["sigma"])
        hi = pt * math.exp(mx[d]["sigma"])
        per_dish[d] = {
            "portions": int(math.ceil(pt * buffer)),
            "low": int(math.floor(lo)),
            "high": int(math.ceil(hi * buffer)),
            "share_pct": round(share * 100, 2),
            "weather_delta_pct": round(wx_delta * 100, 1),
            "n_samples": mx[d]["n"],
        }
        if abs(wx_delta) >= 0.08:
            notes.append(f"{DISH_LABEL[d]} {wx_delta*100:+.0f}% on this weather")

    # "normal" = the same weekday, same point in the trend, neutral weather,
    # no bookings and no events. So HEAVY means busy for a current Friday,
    # not merely busier than Fridays were back in July.
    wd_avg, _, _ = predict_total(ft, target, weekday, TEMP_REF, 0.0)
    ratio = total / wd_avg if wd_avg else 1.0
    level = "HEAVY" if ratio >= 1.12 else ("LIGHT" if ratio <= 0.90 else "NORMAL")

    return {
        "date": target.isoformat(),
        "weekday": weekday,
        "inputs": {"tmax": tmax, "precip_mm": precip, "wet": bool(wet_flag(precip)),
                   "reserved_guests": reserved_guests, "event_factor": event_factor,
                   "holiday": holiday},
        "total_units": int(round(total * buffer)),
        "total_low": int(round(total * math.exp(-ft["sigma"]))),
        "total_high": int(round(total * math.exp(ft["sigma"]) * buffer)),
        "prep_level": level,
        "vs_weekday_avg_pct": round((ratio - 1) * 100, 1),
        "weekday_samples": ft["n_by_wd"].get(weekday, 0),
        "weekday_known": wd_known,
        "weather_multiplier_on_total": round(wx_mult, 3),
        "reservation_multiplier": round(resv, 3),
        "dishes": per_dish,
        "behaviour_notes": notes,
        "model": {
            "trend_pct_per_month": round((math.exp(ft["b"]) - 1) * 100, 1),
            "total_temp_coef": round(ft["temp_coef"], 4),
            "total_wet_coef": round(ft["wet_coef"], 4),
            "sigma_log": round(ft["sigma"], 3),
            "n_samples": len(rows),
        },
    }


def render(f):
    L = []
    d = dt.date.fromisoformat(f["date"])
    L.append(f"PREP {d.strftime('%a %d.%m')}  ~{f['total_units']} units "
             f"({f['total_low']}-{f['total_high']}), {f['prep_level']} "
             f"({f['vs_weekday_avg_pct']:+.0f}% vs {f['weekday']} avg, "
             f"{f['weekday_samples']} samples)")
    for k, v in f["dishes"].items():
        wx = f"  [{v['weather_delta_pct']:+.0f}% weather]" if abs(v["weather_delta_pct"]) >= 5 else ""
        L.append(f"  {DISH_LABEL[k]:<15} {v['portions']:>4}   "
                 f"(range {v['low']}-{v['high']}){wx}")
    if f["behaviour_notes"]:
        L.append("Weather behaviour: " + "; ".join(f["behaviour_notes"]))
    m = f["model"]
    L.append(f"model: trend {m['trend_pct_per_month']:+.1f}%/month, "
             f"wet coef {m['total_wet_coef']:+.3f}, temp coef {m['total_temp_coef']:+.3f}, "
             f"n={m['n_samples']}")
    return "\n".join(L)


def explain():
    dishes, rows = load()
    ref = rows[-1]["d"]
    ft = fit_total(rows, ref)
    mx = fit_mix(dishes, rows, ref)
    print(f"samples: {len(rows)}  ({rows[0]['date']} .. {rows[-1]['date']})")
    print(f"level trend: {(math.exp(ft['b'])-1)*100:+.1f}% per month")
    print(f"total vs weather: temp {ft['temp_coef']:+.3f}/z, wet {ft['wet_coef']:+.3f}")
    print("\nweekday factors (x the trend line):")
    for k, v in sorted(ft["wd_factor"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} x{math.exp(v):.2f}   n={ft['n_by_wd'][k]}")
    print("\ndish mix sensitivities (effect on the dish's SHARE of the day):")
    print(f"  {'dish':<15}{'share%':>8}{'per +5C':>10}{'if wet':>9}{'per month':>11}{'n':>4}")
    for d in dishes:
        m = mx[d]
        if m["beta"] is None:
            print(f"  {DISH_LABEL[d]:<15}{m['flat']*100:8.2f}{'--':>10}{'--':>9}{'--':>11}{m['n']:>4}")
            continue
        b = m["beta"]
        print(f"  {DISH_LABEL[d]:<15}{math.exp(b[0])*100:8.2f}"
              f"{(math.exp(b[2])-1)*100:+9.0f}%{(math.exp(b[3])-1)*100:+8.0f}%"
              f"{(math.exp(b[1])-1)*100:+10.0f}%{m['n']:>4}")
    print("\nNote: temperature and calendar month are correlated in this sample "
          "(Sept is both cooler and busier), so the two right-hand columns split "
          "credit imperfectly. Treat direction as real, magnitude as indicative.")


def backtest():
    dishes, rows = load()
    print(f"{'date':12}{'wd':10}{'actual':>7}{'pred':>7}{'err%':>7}")
    errs = []
    for i, r in enumerate(rows):
        held = rows[:i] + rows[i + 1:]
        if not any(h["weekday"] == r["weekday"] for h in held):
            continue
        ft = fit_total(held, r["d"])
        pred, known, _ = predict_total(ft, r["d"], r["weekday"], r["tmax"], r["precip"])
        e = (pred - r["total"]) / r["total"] * 100
        errs.append(abs(e))
        print(f"{r['date']:12}{r['weekday'][:9]:10}{r['total']:7.0f}{pred:7.0f}{e:+7.1f}")
    if errs:
        print(f"\nMAPE {stat.mean(errs):.1f}%   median |err| {stat.median(errs):.1f}%   n={len(errs)}")
        print("(leave-one-out on total units, before reservation/event uplift)")


def add_actual(date, units, qty, tmax=None, precip=None):
    """Append a real trading day to stats.json (and its weather) so the model learns."""
    stats = json.load(open(STATS))
    weekday = dt.date.fromisoformat(date).strftime("%A")
    row = {"date": date, "total_units": int(units)}
    for d in stats["dishes"]:
        row[d] = int(qty.get(d, 0))
    bucket = stats["samples"].setdefault(weekday, [])
    bucket[:] = [s for s in bucket if s.get("date") != date]   # replace, don't duplicate
    bucket.append(row)
    bucket.sort(key=lambda s: s["date"])
    json.dump(stats, open(STATS, "w"))

    if tmax is not None or precip is not None:
        w = json.load(open(WEATHER)) if os.path.exists(WEATHER) else {}
        e = w.get(date, {})
        if tmax is not None:
            e["tmax"] = tmax
        if precip is not None:
            e["precip_mm"] = precip
        w[date] = e
        json.dump(w, open(WEATHER, "w"), indent=0)
    return f"stored {date} ({weekday}) = {units} units, {len(bucket)} sample(s) for {weekday}"


def parse_qty(s):
    out = {}
    for part in s.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        out[k.strip().upper()] = float(v)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date")
    p.add_argument("--tmax", type=float)
    p.add_argument("--precip-mm", type=float, default=0.0)
    p.add_argument("--reserved-guests", type=int, default=0)
    p.add_argument("--event-factor", type=float, default=1.0)
    p.add_argument("--holiday", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--add-actual", metavar="YYYY-MM-DD",
                   help="record a real trading day; needs --units and --qty")
    p.add_argument("--units", type=int)
    p.add_argument("--qty", default="",
                   help="PLOV=52,MANTY_MEAT=66,SAMSA_BEEF=54,LAGMAN_SOUP=30 ...")
    a = p.parse_args()

    if a.add_actual:
        print(add_actual(a.add_actual, a.units, parse_qty(a.qty),
                         a.tmax, a.precip_mm if a.precip_mm is not None else None))
    elif a.explain:
        explain()
    elif a.backtest:
        backtest()
    else:
        target = dt.date.fromisoformat(a.date) if a.date else dt.date.today()
        f = forecast(target, a.tmax, a.precip_mm, a.reserved_guests,
                     a.event_factor, a.holiday)
        print(json.dumps(f, indent=2) if a.json else render(f))
