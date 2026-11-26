"""
postlift — growth-adjusted engagement + transparent content coding for political social posts.

Why it exists
-------------
Raw likes are dominated by audience size: Trump's median tweet went from ~20 likes (2013)
to ~60,000 (2017). Comparing posts on raw counts mostly measures *when* they were posted.
postlift scores each post against what the same account was getting at the same time:

    lift = log(1 + likes) - rolling median of log(1 + likes) over the surrounding ±15 days
    (original posts only, same platform), reported as "x times a typical post that month".

It then codes every post with a transparent, editable keyword lexicon (themes, style and
format), so the same rules run identically on 2009 tweets and 2026 Truth Social posts —
instead of relying on the auto-generated labels shipped with one platform only.

Usage
-----
    python postlift.py --twitter trump_tweets.csv --truth trump_posts_full.csv --out out/

Outputs: posts.parquet (one row per post, harmonised), theme_lift.csv, model_*.csv,
validation_sample.csv (for hand-coding), and summary JSON used by the report.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Theme lexicon — edit here. Regexes are case-insensitive, word-boundary anchored.
# ---------------------------------------------------------------------------
THEMES: dict[str, list[str]] = {
    "Immigration & border": [r"border", r"illegal (?:alien|immigra|migrant)", r"immigra\w*", r"migrants?",
                              r"deport\w*", r"the wall", r"cartels?", r"asylum", r"sanctuary", r"invasion"],
    "Economy, jobs & prices": [r"econom\w*", r"jobs", r"stock market", r"inflation", r"tariffs?", r"trade deal\w*",
                               r"gas prices?", r"taxe?s", r"tax cuts?", r"the dow", r"unemployment", r"manufactur\w*",
                               r"interest rates?", r"the fed", r"groceries", r"prices"],
    "Elections & fraud claims": [r"rigged", r"election", r"ballots?", r"voter fraud", r"fraud", r"stolen",
                                 r"mail[- ]in", r"polls?", r"primary", r"landslide"],
    "Legal fights & 'witch hunt'": [r"witch ?hunt", r"indict\w*", r"prosecut\w*", r"judges?", r"doj", r"fbi",
                                    r"mueller", r"impeach\w*", r"jack smith", r"bragg", r"trial", r"persecut\w*",
                                    r"hoax", r"lawfare", r"weaponiz\w*", r"special counsel"],
    "Attacks on media": [r"fake news", r"cnn", r"msnbc", r"new york times", r"nytimes", r"washington post",
                         r"failing", r"lamestream", r"nbc", r"abc news", r"cbs", r"enemy of the people", r"media"],
    "Attacks on opponents": [r"biden", r"kamala", r"harris", r"obama", r"hillary", r"crooked", r"sleepy",
                             r"democrats?", r"radical left", r"pelosi", r"schumer", r"newsom", r"desantis",
                             r"rinos?", r"liddle", r"lyin", r"loser", r"low iq", r"aoc", r"comrade"],
    "Foreign policy & war": [r"china", r"russia", r"ukraine", r"iran", r"israel", r"nato", r"putin", r"xi",
                             r"north korea", r"kim jong", r"hamas", r"gaza", r"venezuela", r"world war"],
    "Crime & law and order": [r"crime", r"criminals?", r"police", r"law (?:and|&) order", r"murder\w*", r"antifa",
                              r"riots?", r"violent", r"gangs?", r"fentanyl", r"drugs?"],
    "Faith, values & culture war": [r"god(?! bless)", r"christian\w*", r"faith", r"pray\w*", r"church", r"abortion",
                                    r"pro[- ]life", r"woke", r"transgender", r"gender", r"women'?s sports",
                                    r"second amendment", r"2nd amendment", r"guns?", r"religio\w*", r"bible"],
    "MAGA identity & movement": [r"maga", r"make america great again", r"america first", r"patriots?",
                                 r"silent majority", r"our movement", r"forgotten (?:men|man|people)"],
    "Endorsements": [r"complete and total endorsement", r"has my complete", r"strong endorsement",
                     r"full endorsement", r"i fully endorse", r"is running for"],
    "Rallies & crowds": [r"rally", r"crowd\w*", r"packed", r"arena", r"see you (?:soon|tonight|there)"],
    "Health, MAHA & COVID": [r"rfk", r"maha", r"make america healthy", r"vaccin\w*", r"covid", r"virus",
                             r"pandemic", r"health ?care", r"obamacare", r"drug prices"],
    "Tech, crypto & new media": [r"crypto\w*", r"bitcoin", r"\bai\b", r"artificial intelligence", r"elon",
                                 r"musk", r"podcast", r"rogan", r"tiktok", r"ufc", r"truth social"],
    "Celebrity & business": [r"apprentice", r"golf", r"trump tower", r"real estate", r"miss universe",
                             r"letterman", r"trump (?:hotel|international|national|doral|organization)",
                             r"book", r"tune in", r"interview", r"fox (?:and|&) friends"],
}
_THEME_RX = {k: re.compile(r"\b(?:" + "|".join(v) + r")", re.I) for k, v in THEMES.items()}

# Nicknames / insults — a style flag, not a theme.
INSULT_RX = re.compile(r"\b(crooked|sleepy|lyin'?|little|liddle|low iq|loser|dumb(?:est)?|stupid|lightweight|"
                       r"pathetic|sick|corrupt|crazy|deranged|incompetent|disgrace\w*|scum|lunatic)\b", re.I)
URL_RX = re.compile(r"https?://\S+")
MENTION_RX = re.compile(r"@\w+")
WORD_RX = re.compile(r"[A-Za-z']+")


# ---------------------------------------------------------------------------
# 2. Loading & harmonising
# ---------------------------------------------------------------------------
MOJIBAKE_RX = re.compile(r"[\u00c2-\u00f4][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u02dc\u2013-\u203a\u20ac\u2122]{1,3}")


def _fix_run(m: re.Match) -> str:
    chunk = m.group(0)
    for enc in ("latin-1", "cp1252"):
        try:
            return chunk.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return chunk


def _clean_text(s: str) -> str:
    s = html.unescape(str(s)) if isinstance(s, str) else ""
    if "\u00e2" in s or "\u00c2" in s or "\u00c3" in s:  # UTF-8 read as Latin-1/cp1252 (mojibake)
        s = MOJIBAKE_RX.sub(_fix_run, s)
    if len(s) > 1 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].replace('""', '"')  # Twitter export double-escaped quotes
    return re.sub(r"\s+", " ", s).strip()


def load_twitter(path: str | Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    out = pd.DataFrame({
        "platform": "Twitter",
        "post_id": t["id"].astype("Int64").astype(str),
        "ts": pd.to_datetime(t["date"], errors="coerce").dt.tz_localize("UTC"),
        "text": t["text"].map(_clean_text),
        "likes": t["favorites"].astype(float),
        "shares": t["retweets"].astype(float),
        "replies": np.nan,
        "deleted": t["isDeleted"].eq("t"),
        "flagged": t["isFlagged"].eq("t"),
        "device": t["device"],
    })
    native_rt = t["isRetweet"].eq("t")
    manual_rt = out["text"].str.match(r'^"\s*@\w+\s*:') | out["text"].str.match(r"^RT @")  # 2012-15 fan quote-tweets
    reply = out["text"].str.match(r"^\.?@\w+")  # starts with a handle = reply/mention-led, not a broadcast post
    out["post_type"] = np.select([native_rt, manual_rt, reply], ["Repost", "Quoted fan", "Reply"], "Original")
    return out


def load_truth(path: str | Path) -> pd.DataFrame:
    p = pd.read_csv(path, low_memory=False)
    text = p["text"].map(_clean_text)
    out = pd.DataFrame({
        "platform": "Truth Social",
        "post_id": p["id"].astype(str),
        "ts": pd.to_datetime(p["created_at"], utc=True, format="ISO8601", errors="coerce"),
        "text": text,
        "likes": p["like_count"].astype(float),
        "shares": p["retruth_count"].astype(float),
        "replies": p["reply_count"].astype(float),
        "deleted": False,
        "flagged": False,
        "device": np.nan,
        # supplied auto-labels kept only for comparison/validation
        "src_topic": p["topic"],
        "src_rhetoric": p["rhetorical_device"],
        "src_sentiment": p["sentiment_label"],
    })
    # is_retruth is False for every row in the supplied file, so detect reposts from text.
    repost = text.str.match(r"^RT @") | (out["likes"].eq(0) & text.str.match(r"^RT\b"))
    media_only = text.eq("")
    link_only = (~media_only) & text.str.replace(URL_RX, "", regex=True).str.strip().eq("")
    quote = text.str.match(r"^RT:\s*https?://truthsocial")  # Trump's own quote-post of another Truth
    out["post_type"] = np.select([repost, media_only, link_only, quote],
                                 ["Repost", "Media only", "Link only", "Quote-post"], "Original")
    return out


# ---------------------------------------------------------------------------
# 3. Features — computed identically on both platforms
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    txt = df["text"].fillna("")
    body = txt.map(lambda s: MENTION_RX.sub("", URL_RX.sub("", s)))
    words = body.map(WORD_RX.findall)
    n_words = words.map(len)
    caps = words.map(lambda w: sum(1 for x in w if len(x) >= 3 and x.isupper()))
    df["word_count"] = n_words
    df["caps_words"] = caps
    df["caps_share"] = np.where(n_words > 0, caps / n_words.clip(lower=1), 0.0)
    df["exclaims"] = txt.str.count("!")
    # external links only: Truth Social quote-links point inside the platform
    df["has_link"] = txt.str.contains(r"https?://(?!truthsocial\.com)\S+", regex=True)
    df["insult"] = txt.str.replace(URL_RX, " ", regex=True).str.contains(INSULT_RX.pattern, flags=re.I, regex=True)
    df["length_band"] = pd.cut(n_words, [-1, 0, 15, 40, 80, 10_000],
                               labels=["No text", "Very short (1–15)", "Short (16–40)", "Medium (41–80)", "Long (80+)"])
    # Endorsement posts reuse a policy-boilerplate template ("Secure the Border, Grow the Economy,
    # Protect our Second Amendment..."), so they get the Endorsement theme only.
    # Themes are coded on the words the author wrote: URLs are removed first, because article
    # slugs (e.g. foxnews.com/politics/biden-...) otherwise add ~10% spurious theme tags.
    words_only = txt.str.replace(URL_RX, " ", regex=True)
    endorse = words_only.str.contains(_THEME_RX["Endorsements"].pattern, flags=re.I, regex=True)
    for name, rx in _THEME_RX.items():
        hit = words_only.str.contains(rx.pattern, flags=re.I, regex=True)
        df[f"t::{name}"] = hit if name == "Endorsements" else hit & ~endorse
    tcols = [c for c in df.columns if c.startswith("t::")]
    df["n_themes"] = df[tcols].sum(axis=1)
    loc = df["ts"].dt.tz_convert("America/New_York")
    df["hour_et"] = loc.dt.hour
    df["weekday"] = loc.dt.day_name()
    df["year"] = loc.dt.year
    df["era"] = pd.cut(df["ts"], pd.to_datetime(["2009-01-01", "2015-06-16", "2017-01-20", "2021-01-20",
                                                  "2025-01-20", "2030-01-01"], utc=True),
                       labels=["Celebrity (2009–mid 2015)", "Candidate (2015–16)", "President, 1st term (2017–21)",
                               "Out of office (2022–Jan 2025)", "President, 2nd term (2025–26)"], right=False)
    return df


# ---------------------------------------------------------------------------
# 4. Growth-adjusted engagement ("lift")
# ---------------------------------------------------------------------------
def add_lift(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    df["log_likes"] = np.log1p(df["likes"])
    df["lift_log"] = np.nan
    # zero likes on a non-repost = missing data (clustered in early 2013), not a genuine flop
    scored = df["post_type"].isin(["Original", "Media only", "Link only", "Quote-post"]) & ~df["deleted"] & df["likes"].gt(0)
    for plat, g in df[scored].groupby("platform"):
        g = g.sort_values("ts")
        s = pd.Series(g["log_likes"].values, index=g["ts"])
        base = s.rolling(f"{window_days}D", center=True, min_periods=10).median()
        df.loc[g.index, "baseline_log"] = base.values
        df.loc[g.index, "lift_log"] = g["log_likes"].values - base.values
    # implausible outliers (<0.5% of the account's typical likes that month) are capture errors, not flops
    bad = df["lift_log"] < np.log(0.005)
    df["capture_error"] = bad
    df.loc[bad, "lift_log"] = np.nan
    df["lift_x"] = np.exp(df["lift_log"])  # 1.0 = typical post that month; 2.0 = twice the likes
    return df


# ---------------------------------------------------------------------------
# 5. Summaries
# ---------------------------------------------------------------------------
def theme_table(df: pd.DataFrame, by: str = "platform") -> pd.DataFrame:
    rows = []
    tcols = [c for c in df.columns if c.startswith("t::")]
    d = df[df.lift_log.notna() & df.post_type.eq("Original")]  # text posts only, so format doesn't confound
    for key, g in d.groupby(by, observed=True):
        for c in tcols:
            m = g[c]
            if m.sum() < 30:
                continue
            rows.append({by: key, "theme": c[3:], "n": int(m.sum()), "share": m.mean(),
                         "median_lift_x": float(np.exp(g.loc[m, "lift_log"].median())),
                         "mean_lift_x": float(np.exp(g.loc[m, "lift_log"].mean())),
                         "lift_vs_rest": float(np.exp(g.loc[m, "lift_log"].mean() - g.loc[~m, "lift_log"].mean()))})
    return pd.DataFrame(rows)


def fit_model(df: pd.DataFrame, platform: str, eras: list[str] | None = None) -> pd.DataFrame:
    """OLS of log-lift on themes + style + format + timing; HC3 robust SEs. Coefs as % change in likes."""
    import statsmodels.api as sm
    d = df[(df.platform == platform) & df.lift_log.notna()].copy()
    if eras:
        d = d[d.era.isin(eras)]
    tcols = [c for c in d.columns if c.startswith("t::")]
    X = d[tcols].astype(float).rename(columns=lambda c: "Theme: " + c[3:])
    X["Style: insult / nickname"] = d["insult"].astype(float)
    X["Style: ALL-CAPS share (per +10pp)"] = d["caps_share"] * 10
    X["Style: exclamation (any)"] = (d["exclaims"] > 0).astype(float)
    X["Format: contains link"] = d["has_link"].astype(float)
    X["Format: media only (no text)"] = d["post_type"].eq("Media only").astype(float)
    X["Format: link only (no comment)"] = d["post_type"].eq("Link only").astype(float)
    lb = pd.get_dummies(d["length_band"], prefix="Length", dtype=float)
    lb = lb.drop(columns=[c for c in lb.columns if "Short (16–40)" in c or "No text" in c or lb[c].sum() == 0])
    X = pd.concat([X, lb], axis=1)
    X["Timing: late night (11pm–5am ET)"] = d["hour_et"].isin([23, 0, 1, 2, 3, 4]).astype(float)
    X["Timing: weekend"] = d["weekday"].isin(["Saturday", "Sunday"]).astype(float)
    X = X.loc[:, X.std() > 0]  # drop constant columns (e.g. no media-only posts on Twitter export)
    X = sm.add_constant(X)
    res = sm.OLS(d["lift_log"], X).fit(cov_type="HC3")
    ci = res.conf_int()
    out = pd.DataFrame({"term": res.params.index, "coef": res.params.values,
                        "pct": (np.exp(res.params.values) - 1) * 100,
                        "lo": (np.exp(ci[0].values) - 1) * 100, "hi": (np.exp(ci[1].values) - 1) * 100,
                        "p": res.pvalues.values})
    out.attrs.update(n=int(res.nobs), r2=float(res.rsquared))
    return out[out.term != "const"]


def validation_sample(df: pd.DataFrame, n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Stratified sample of original Truth Social + Twitter posts for hand-coding the lexicon."""
    d = df[(df.post_type == "Original") & (df.word_count >= 5)]
    s = pd.concat([g.sample(n // 2, random_state=seed) for _, g in d.groupby("platform")])
    tcols = [c for c in d.columns if c.startswith("t::")]
    s = s.assign(lexicon_themes=s[tcols].apply(lambda r: "; ".join(c[3:] for c, v in r.items() if v), axis=1))
    return s[["platform", "post_id", "ts", "text", "lexicon_themes"]].assign(hand_themes="", hand_notes="")


def run(twitter: str, truth: str, out: str) -> pd.DataFrame:
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load_twitter(twitter), load_truth(truth)], ignore_index=True)
    n0 = len(df); df = df[df.ts.notna()].reset_index(drop=True)
    print(f"dropped {n0-len(df)} rows with unparseable dates")
    df = add_lift(add_features(df))
    df.to_parquet(outp / "posts.parquet")
    theme_table(df).to_csv(outp / "theme_lift_by_platform.csv", index=False)
    theme_table(df, "era").to_csv(outp / "theme_lift_by_era.csv", index=False)
    political = ["Candidate (2015–16)", "President, 1st term (2017–21)", "Out of office (2022–Jan 2025)",
                 "President, 2nd term (2025–26)"]
    for plat in ["Twitter", "Truth Social"]:
        fit_model(df, plat, political).to_csv(outp / f"model_{plat.replace(' ', '_').lower()}.csv", index=False)
    validation_sample(df).to_csv(outp / "validation_sample.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twitter", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    d = run(a.twitter, a.truth, a.out)
    print(d.groupby(["platform", "post_type"]).size())
