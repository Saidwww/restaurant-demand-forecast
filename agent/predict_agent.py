#!/usr/bin/env python3
"""
Restaurant Predict Agent — daily per-dish prep forecast sent to Telegram.

Data sources: open-meteo (weather), date.nager.at (DE holidays),
an optional reservations API, stats.json (sales history).
Optional LLM commentary: Ollama (local) or Groq (cloud), both Llama 3.1.

Usage:
  python predict_agent.py            # forecast for today, send to Telegram
  python predict_agent.py --tomorrow # forecast for tomorrow
  python predict_agent.py --dry-run  # print instead of sending
  python predict_agent.py --collect  # only ingest actuals replied to the bot
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

# ---------------- config ----------------
TZ = ZoneInfo("Europe/Berlin")
BOT_TOKEN = os.environ.get("PREDICT_BOT_TOKEN", "")
CHAT_ID = os.environ.get("PREDICT_CHAT_ID", "")
RESERVATIONS_URL = os.environ.get("RESERVATIONS_URL", "")
LAT, LON = 52.52, 13.405  # restaurant location (set your own)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "")          # e.g. http://localhost:11434
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")      # free key from console.groq.com
STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.json")

DISH_LABELS = {
    "PLOV": "🍚 Plov", "MANTY_MEAT": "🥟 Manty (meat)", "MANTY_VEG": "🥟 Manty (veg)",
    "SAMSA_BEEF": "🥐 Samsa beef", "SAMSA_CHICKEN": "🥐 Samsa chicken", "SAMSA_VEG": "🥐 Samsa veg",
    "LAGMAN": "🍜 Lagman", "LAGMAN_SOUP": "🍜 Lagman soup",
}
SAFETY_BUFFER = 1.10
WMO_RAINY = set(range(51, 100))  # WMO weather codes: drizzle/rain/snow/showers/storms


# ---------------- helpers ----------------
def http_get(url, timeout=20, retries=1, retry_wait=30):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.text.strip():
                return r
        except requests.RequestException as e:
            print(f"  ! {url.split('/')[2]}: {e}")
        if attempt < retries:
            time.sleep(retry_wait)
    return None


def load_stats():
    with open(STATS_FILE) as f:
        return json.load(f)


def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


# ---------------- data collection ----------------
def get_weather(day):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
           f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode"
           f"&timezone=Europe/Berlin&start_date={day}&end_date={day}")
    r = http_get(url)
    if not r:
        return None
    d = r.json()["daily"]
    return {"tmax": d["temperature_2m_max"][0], "tmin": d["temperature_2m_min"][0],
            "rain_prob": d["precipitation_probability_max"][0] or 0,
            "rainy": d["weathercode"][0] in WMO_RAINY}


def get_holiday(day):
    r = http_get(f"https://date.nager.at/api/v3/PublicHolidays/{day.year}/DE")
    if not r:
        return None
    for h in r.json():
        counties = h.get("counties")
        if h["date"] == str(day) and (not counties or "DE-BE" in counties):
            return h["localName"]
    return ""


def get_reservations(day):
    if not RESERVATIONS_URL:
        return None
    r = http_get(RESERVATIONS_URL, retries=1, retry_wait=30)  # render free tier may sleep
    if not r:
        return None
    todays = [x for x in r.json()
              if x.get("reservationDate") == str(day)
              and x.get("status", "").upper() in ("PENDING", "CONFIRMED")]
    return {"count": len(todays),
            "guests": sum(x.get("guestCount", 0) for x in todays),
            "details": [f"{x.get('reservationTime','')[:5]} {x.get('guestCount')}p {x.get('tableType','')}"
                        for x in todays]}


# ---------------- forecasting ----------------
MAX_SAMPLES = 8  # only the freshest N days per weekday — old data ages out as you grow


def dish_base(samples, dish):
    """Recent-weighted mean, softening outliers (>1.6x median counts half)."""
    samples = samples[-MAX_SAMPLES:]
    vals = [s.get(dish, 0) for s in samples]
    if not vals:
        return 0
    med = statistics.median(vals)
    weights, total, wsum = [], 0.0, 0.0
    for i, v in enumerate(vals):
        w = 1.0 + 0.5 * i                      # newer samples weigh more
        if med > 0 and v > 1.6 * med:
            w *= 0.5                           # soften outlier days
        total += v * w
        wsum += w
    return total / wsum


def make_forecast(stats, weekday, weather, holiday, resv):
    samples = stats["samples"].get(weekday, [])
    if not samples:  # no history for this weekday -> average all weekdays
        samples = [s for lst in stats["samples"].values() for s in lst]

    adj, reasons = 1.0, []
    if holiday:
        adj *= 1.20
        reasons.append(f"holiday: {holiday} +20%")
    if weather:
        if weather["rain_prob"] >= 60 or weather["rainy"]:
            adj *= 0.88
            reasons.append(f"rain {weather['rain_prob']}% -12%")
        elif weather["tmax"] >= 32:
            adj *= 0.90
            reasons.append("very hot -10%")
        elif 22 <= weather["tmax"] <= 28:
            adj *= 1.05
            reasons.append("nice weather +5%")
    if resv and resv["guests"] >= 8:
        adj *= 1.0 + min(resv["guests"], 40) * 0.01
        reasons.append(f"{resv['guests']} reserved guests +{min(resv['guests'],40)}%")

    portions = {d: max(0, math.ceil(dish_base(samples, d) * adj * SAFETY_BUFFER))
                for d in stats["dishes"]}
    avg_total = statistics.mean([s["total_units"] for s in samples])
    exp_total = round(avg_total * adj)
    level = "LIGHT" if exp_total < avg_total * 0.8 else ("HEAVY" if exp_total > avg_total * 1.15 else "NORMAL")
    return portions, exp_total, level, reasons


def backtest(stats):
    """Replay history: for each day, predict it using only the days before it.
    Prints per-day and overall error so training progress is measurable."""
    print(f"{'date':<12}{'weekday':<10}{'actual':>7}{'pred':>7}{'err%':>7}")
    errors = []
    for weekday, samples in sorted(stats["samples"].items()):
        for i in range(1, len(samples)):
            past, actual = samples[:i], samples[i]
            if "total_units" not in actual:
                continue
            pred = round(statistics.mean(s["total_units"] for s in past[-MAX_SAMPLES:]) * SAFETY_BUFFER)
            err = abs(pred - actual["total_units"]) / actual["total_units"] * 100
            errors.append(err)
            print(f"{actual['date']:<12}{weekday:<10}{actual['total_units']:>7}{pred:>7}{err:>6.1f}%")
    if errors:
        mape = statistics.mean(errors)
        print(f"\nMAPE (avg error): {mape:.1f}%  over {len(errors)} days")
        print("Target: <20% decent, <15% good, <10% excellent. More training days = lower error.")
    else:
        print("Not enough history yet — need 2+ days on the same weekday. Keep training.")


# ---------------- LLM (optional) ----------------
def llm_comment(context):
    prompt = ("You advise a small restaurant. Given this data, reply with 2 short, "
              "practical sentences for the kitchen team (no lists): " + json.dumps(context))
    try:
        if OLLAMA_URL:
            r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/generate",
                              json={"model": "llama3.1", "prompt": prompt, "stream": False}, timeout=60)
            return r.json().get("response", "").strip()
        if GROQ_API_KEY:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                              json={"model": "llama-3.1-8b-instant",
                                    "messages": [{"role": "user", "content": prompt}]}, timeout=60)
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ! LLM skipped: {e}")
    return ""


# ---------------- telegram ----------------
def send_telegram(text, dry=False):
    if dry:
        print("\n--- TELEGRAM (dry run) ---\n" + text + "\n--------------------------")
        return True
    r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": text}, timeout=20)
    ok = r.status_code == 200 and r.json().get("ok")
    print("  telegram:", "delivered" if ok else f"FAILED {r.text[:200]}")
    return ok


def collect_actuals(stats, dry=False):
    """Read replies to the prediction bot like 'actual 175' and store yesterday's total."""
    if not BOT_TOKEN:
        return
    r = http_get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={stats.get('telegram_offset', 0) + 1}")
    if not r:
        return
    for u in r.json().get("result", []):
        stats["telegram_offset"] = u["update_id"]
        text = (u.get("message") or {}).get("text", "").lower().strip()
        if text.startswith("actual"):
            try:
                total = int(text.split()[1])
            except (IndexError, ValueError):
                continue
            yday = (datetime.now(TZ).date() - timedelta(days=1))
            wd = yday.strftime("%A")
            stats["samples"].setdefault(wd, []).append({"date": str(yday), "total_units": total})
            print(f"  recorded actual {total} for {yday} ({wd})")
            if not dry:
                send_telegram(f"Recorded: {total} units for {yday}. Thanks!")
    save_stats(stats)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tomorrow", action="store_true")
    ap.add_argument("--collect", action="store_true", help="only collect actuals, no forecast")
    ap.add_argument("--report", action="store_true", help="backtest accuracy on training history")
    args = ap.parse_args()

    stats = load_stats()
    if args.report:
        backtest(stats)
        return
    collect_actuals(stats, dry=args.dry_run)
    if args.collect:
        return

    day = datetime.now(TZ).date() + timedelta(days=1 if args.tomorrow else 0)
    weekday = day.strftime("%A")
    print(f"Forecast for {weekday} {day}")

    weather = get_weather(day)
    holiday = get_holiday(day)
    resv = get_reservations(day)
    portions, exp_total, level, reasons = make_forecast(stats, weekday, weather, holiday, resv)

    lines = [f"🍽 PREP {weekday[:3]} {day.strftime('%d.%m')}: ~{exp_total} units, {level}"]
    lines += [f"{DISH_LABELS[d]}: {n}" for d, n in portions.items()]
    msg1 = "\n".join(lines)

    ctx = []
    if resv is None:
        ctx.append("📋 Reservations: source unavailable")
    else:
        ctx.append(f"📋 Reservations: {resv['count']} ({resv['guests']} guests) " +
                   ("- " + "; ".join(resv["details"]) if resv["details"] else ""))
    ctx.append(f"🌤 {weather['tmin']:.0f}-{weather['tmax']:.0f}C, rain {weather['rain_prob']}%"
               if weather else "🌤 Weather: source unavailable")
    ctx.append(f"🎉 Holiday: {holiday}" if holiday else
               ("🎉 No holiday" if holiday == "" else "🎉 Holiday check unavailable"))
    if reasons:
        ctx.append("💡 " + "; ".join(reasons))
    comment = llm_comment({"day": str(day), "weekday": weekday, "portions": portions,
                           "weather": weather, "holiday": holiday, "reservations": resv})
    if comment:
        ctx.append("🤖 " + comment)
    ctx.append("Reply 'actual <number>' with yesterday's real units to improve forecasts.")
    msg2 = "\n".join(ctx)

    ok = send_telegram(msg1, dry=args.dry_run) and send_telegram(msg2, dry=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
