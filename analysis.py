"""Builds every number and chart dataset used in the ADI report from postlift output -> out/report_data.json"""
import json
import numpy as np
import pandas as pd
import postlift as pl

d = pd.read_parquet("out/posts.parquet")
R = {}
POL = ["Candidate (2015–16)", "President, 1st term (2017–21)", "Out of office (2022–Jan 2025)",
       "President, 2nd term (2025–26)"]
scored = d[d.lift_log.notna()]
pol = scored[scored.era.isin(POL)]

# --- headline counts -------------------------------------------------------
R["counts"] = d.groupby(["platform", "post_type"]).size().unstack(fill_value=0).to_dict(orient="index")
R["n_capture_error"] = int(d.capture_error.sum()); R["n_zero_likes"] = int(((d.likes == 0) & ~d.post_type.eq("Repost")).sum())
R["n_scored"] = int(len(scored)); R["n_scored_political"] = int(len(pol))

# --- 1. raw likes vs time (monthly median, originals) ----------------------
m = scored.set_index("ts").groupby("platform").likes.resample("MS").median().reset_index()
m = m[m.ts >= "2013-01-01"]
R["monthly_likes"] = [{"platform": r.platform, "month": r.ts.strftime("%Y-%m"), "median_likes": float(r.likes)}
                      for r in m.dropna().itertuples()]
vol = d.set_index("ts").groupby("platform").text.resample("MS").size().reset_index(name="n")
R["monthly_posts"] = [{"platform": r.platform, "month": r.ts.strftime("%Y-%m"), "n": int(r.n)} for r in vol.itertuples()
                      if r.ts >= pd.Timestamp("2013-01-01", tz="UTC")]

# --- 2. era summary ----------------------------------------------------------
era = d.groupby("era", observed=True).agg(
    posts=("text", "size"),
    days=("ts", lambda x: (x.max() - x.min()).days + 1),
    repost_share=("post_type", lambda x: x.eq("Repost").mean()),
    media_or_link_share=("post_type", lambda x: x.isin(["Media only", "Link only"]).mean()),
)
era["posts_per_day"] = era.posts / era.days
med = scored.groupby("era", observed=True).likes.median()
era["median_likes"] = med
oe = d[d.post_type == "Original"].groupby("era", observed=True)
era["caps_share"] = oe.caps_share.mean(); era["insult_share"] = oe.insult.mean(); era["median_words"] = oe.word_count.median()
era["exclaim_share"] = oe.exclaims.apply(lambda x: (x > 0).mean())
R["eras"] = era.reset_index().round(4).to_dict(orient="records")

# --- 3. theme lift by platform (political eras) & by era --------------------
R["theme_platform"] = pl.theme_table(pol).round(4).to_dict(orient="records")
R["theme_era"] = pl.theme_table(scored[scored.era.isin(POL)], "era").round(4).to_dict(orient="records")

# --- 4. models ----------------------------------------------------------------
for plat in ["Truth Social", "Twitter"]:
    mm = pl.fit_model(d, plat, POL)
    R[f"model_{plat}"] = {"n": mm.attrs["n"], "r2": round(mm.attrs["r2"], 4),
                          "terms": mm.round(4).to_dict(orient="records")}

# --- 5. format (Truth Social) ----------------------------------------------
ts = scored[scored.platform == "Truth Social"]
fmt = ts.groupby("post_type").lift_log.agg(["size", "median"])
fmt["median_lift_x"] = np.exp(fmt["median"])
R["format_truth"] = fmt.reset_index().round(4).to_dict(orient="records")
tw = pol[pol.platform == "Twitter"]
R["link_twitter"] = {"with_link": float(np.exp(tw[tw.has_link].lift_log.median())),
                     "no_link": float(np.exp(tw[~tw.has_link].lift_log.median()))}
to = ts[ts.post_type == "Original"]
R["link_truth_text"] = {"with_link": float(np.exp(to[to.has_link].lift_log.median())),
                        "no_link": float(np.exp(to[~to.has_link].lift_log.median()))}

# --- 6. endorsements over time ---------------------------------------------
e = d[(d.platform == "Truth Social") & d.post_type.eq("Original")].copy()
e["q"] = e.ts.dt.tz_convert(None).dt.to_period("Q").astype(str)
eq = e.groupby("q").agg(n=("text", "size"), endorse=("t::Endorsements", "sum"))
eq["share"] = eq.endorse / eq.n
R["endorse_quarterly"] = eq.reset_index().round(4).to_dict(orient="records")
R["endorse_total_truth"] = int(e["t::Endorsements"].sum())

# --- 7. Newsom caps test -----------------------------------------------------
o = d[(d.platform == "Truth Social") & (d.post_type == "Original")]
cut = pd.Timestamp("2025-08-11", tz="UTC")
newsom = {}
for w in [4, 8, 13, 26]:
    b = o[(o.ts < cut) & (o.ts >= cut - pd.Timedelta(weeks=w))]
    a = o[(o.ts >= cut) & (o.ts < cut + pd.Timedelta(weeks=w))]
    newsom[f"{w}wk"] = {"before": round(b.caps_share.mean(), 4), "after": round(a.caps_share.mean(), 4),
                        "n_before": len(b), "n_after": len(a)}
R["newsom_caps"] = newsom
mc = o.set_index("ts").caps_share.resample("MS").mean()
R["caps_monthly_truth"] = [{"month": k.strftime("%Y-%m"), "caps": round(float(v), 4)} for k, v in mc.items()]
# twitter caps monthly too (originals, political era)
mt = d[(d.platform == "Twitter") & (d.post_type == "Original") & (d.ts >= "2015-01-01")].set_index("ts").caps_share.resample("MS").mean()
R["caps_monthly_twitter"] = [{"month": k.strftime("%Y-%m"), "caps": round(float(v), 4)} for k, v in mt.items()]

# --- 8. volume vs lift (daily) -------------------------------------------
dd = ts.copy(); dd["day"] = dd.ts.dt.tz_convert("America/New_York").dt.date
day = dd.groupby("day").agg(n=("lift_log", "size"), lift=("lift_log", "mean"))
day["band"] = pd.cut(day.n, [0, 5, 10, 20, 40, 1000], labels=["1–5", "6–10", "11–20", "21–40", "41+"])
vb = day.groupby("band", observed=True).agg(days=("n", "size"), mean_lift=("lift", "mean"))
vb["lift_x"] = np.exp(vb.mean_lift)
day["sum_lift"] = dd.groupby("day").lift_x.sum()
vb["total_typical"] = day.groupby("band", observed=True).sum_lift.median()
vb["posts_median"] = day.groupby("band", observed=True).n.median()
dd2 = dd.merge(day[["band"]], left_on="day", right_index=True)
vb["housekeeping_share"] = dd2.assign(h=dd2["t::Endorsements"] | dd2.post_type.isin(["Link only", "Media only", "Quote-post"])).groupby("band", observed=True).h.mean()
R["volume_bands_truth"] = vb.reset_index().round(4).to_dict(orient="records")
R["max_posts_day"] = int(d[d.platform == "Truth Social"].groupby(d.ts.dt.tz_convert("America/New_York").dt.date).size().max())

# --- 9. example posts (top/bottom by lift, text originals, political eras) ---
def ex(g, k, asc):
    g = g[(g.post_type == "Original") & (g.word_count >= 6) & (g.likes >= 500)].sort_values("lift_log", ascending=asc).head(k)
    return [{"date": r.ts.strftime("%d %b %Y"), "platform": r.platform, "text": r.text[:280],
             "likes": int(r.likes), "lift_x": round(float(r.lift_x), 2)} for r in g.itertuples()]
EXCLUDE_EXAMPLES_RX = r"E\. Jean Carroll"


def ex_typical_low(g, k):
    """Under-performers near the era's 5th percentile of lift, not the extreme minimum: like counts are
    capture-time snapshots, and the very lowest values cannot be verified (e.g. a widely shared Dec 2020
    tweet recorded at 19k likes when that week's median was ~224k)."""
    g = g[(g.post_type == "Original") & (g.word_count >= 6) & (g.likes >= 500)]
    q = g.lift_log.quantile(0.05)
    # editorial: skip one real post that quotes slurs verbatim, unsuitable for a policy audience
    g = g[~g.text.str.contains(EXCLUDE_EXAMPLES_RX, regex=True)]
    g = g.assign(dist=(g.lift_log - q).abs()).sort_values("dist")
    out, seen = [], set()
    for r in g.itertuples():
        key = r.text[:40]
        if key in seen: continue
        seen.add(key)
        out.append({"date": r.ts.strftime("%d %b %Y"), "platform": r.platform, "text": r.text[:280],
                    "likes": int(r.likes), "lift_x": round(float(r.lift_x), 2)})
        if len(out) == k: break
    return out
R["examples"] = {e_: {"top": ex(pol[pol.era == e_], 3, False), "bottom": ex_typical_low(pol[pol.era == e_], 3)} for e_ in POL}
R["examples_low_quantile"] = {e_: round(float(np.exp(pol[(pol.era == e_) & (pol.post_type == "Original")].lift_log.quantile(0.05))), 2) for e_ in POL}

# --- 10. supplied-label cross-check (Truth Social) ---------------------------
tl = ts[ts.post_type == "Original"]
rd = tl.groupby("src_rhetoric").lift_log.agg(["size", "median"]).query("size>=100")
rd["median_lift_x"] = np.exp(rd["median"])
R["src_rhetoric_lift"] = rd.reset_index().round(4).to_dict(orient="records")
pairs = {"Immigration": "Immigration & border", "Economy": "Economy, jobs & prices", "Elections": "Elections & fraud claims",
         "Legal": "Legal fights & 'witch hunt'", "Media": "Attacks on media", "Foreign Policy": "Foreign policy & war"}
agree = []
for src, mine in pairs.items():
    a = tl.src_topic.eq(src); b = tl[f"t::{mine}"]
    agree.append({"src_topic": src, "lexicon_theme": mine, "src_n": int(a.sum()), "lex_n": int(b.sum()),
                  "lexicon_recall_of_src": round(float((a & b).sum() / max(a.sum(), 1)), 3),
                  "src_recall_of_lexicon": round(float((a & b).sum() / max(b.sum(), 1)), 3)})
R["label_agreement"] = agree
R["src_other_share"] = round(float(tl.src_topic.eq("Other").mean()), 3)
R["unthemed_share"] = {p: round(float((g[[c for c in g if c.startswith("t::")]].sum(1) == 0).mean()), 3)
                       for p, g in pol[pol.post_type == "Original"].groupby("platform")}

# --- 11. nicknames -----------------------------------------------------------
nick = {"Crooked Hillary": r"crooked hillary", "Sleepy/Crooked Joe": r"(?:sleepy|crooked) joe",
        "Lyin' Ted": r"lyin'? ted", "Little Marco": r"little marco", "Ron DeSanctimonious": r"desanctimonious",
        "Comrade Kamala": r"comrade kamala", "Fake News": r"fake news"}
R["nicknames"] = []
for k, rx in nick.items():
    msk = pol.text.str.contains(rx, case=False, regex=True)
    if msk.sum() >= 20:
        R["nicknames"].append({"name": k, "n": int(msk.sum()), "lift_x": round(float(np.exp(pol[msk].lift_log.mean())), 3)})

# --- 12. timing --------------------------------------------------------------
hr = pol.groupby(["platform", "hour_et"]).agg(n=("lift_log", "size"), lift=("lift_log", "mean")).reset_index()
hr["lift_x"] = np.exp(hr.lift)
R["hour"] = hr.round(4).to_dict(orient="records")

json.dump(R, open("out/report_data.json", "w"), default=str, indent=1)
print("ok", list(R))
print(json.dumps({k: R[k] for k in ["newsom_caps", "format_truth", "link_twitter", "link_truth_text", "volume_bands_truth",
                                     "nicknames", "label_agreement", "src_other_share", "unthemed_share", "max_posts_day"]}, indent=0, default=str))
print(pd.DataFrame(R["eras"]).to_string())
print(pd.DataFrame(R["src_rhetoric_lift"]).to_string())

# --- web bundle for the report's post checker -----------------------------------
R["lexicon"] = {k: "|".join(v) for k, v in pl.THEMES.items()}
R["insult_rx"] = pl.INSULT_RX.pattern
R["coefs"] = {t["term"]: t["coef"] for t in R["model_Truth Social"]["terms"]}
json.dump(R, open("out/report_data_web.json", "w"), default=str)
