"""Robustness checks quoted in the report ("Three checks", first two) and the method note.

1. Same Truth Social model with shares, replies and total engagement in place of likes
   (each re-baselined with the same rolling ±15-day median), plus correlation of coefficients with the likes model.
2. Likes model with extreme lows (lift < 0.1×) dropped.
3. Android (Trump's own phone) vs iPhone (staff) tweets, 16 Jun 2015 – 28 Feb 2017: average lift,
   insult and exclamation rates, and the exclamation effect with and without a device control.

Run after postlift.py:  python robustness.py   ->  out/robustness.json
"""
import json
import numpy as np
import pandas as pd
import postlift as pl

d = pd.read_parquet("out/posts.parquet")
POL = ["Candidate (2015–16)", "President, 1st term (2017–21)", "Out of office (2022–Jan 2025)", "President, 2nd term (2025–26)"]

# --- 13. robustness: other engagement metrics, trimmed lows, who wrote it -----------------
import statsmodels.api as sm
KEY = ["Style: exclamation (any)", "Style: insult / nickname", "Theme: Foreign policy & war", "Theme: Economy, jobs & prices",
       "Theme: Attacks on opponents", "Theme: Endorsements", "Format: link only (no comment)", "Format: contains link"]
base_m = pl.fit_model(d, "Truth Social", POL).set_index("term").pct
d_trim = d.copy(); d_trim.loc[d_trim.lift_x < 0.1, "lift_log"] = np.nan
trim_m = pl.fit_model(d_trim, "Truth Social", POL).set_index("term").pct
def relift(df, col):
    x = df.copy(); x["likes"] = x[col]
    return pl.add_lift(x.drop(columns=["lift_log", "lift_x", "baseline_log", "log_likes", "capture_error"], errors="ignore"))
tsd = d[d.platform == "Truth Social"].copy()
tsd["all_eng"] = tsd.likes.fillna(0) + tsd.shares.fillna(0) + tsd.replies.fillna(0)
alt = {m: pl.fit_model(relift(tsd, c), "Truth Social", POL).set_index("term").pct for m, c in [("Shares", "shares"), ("Replies", "replies"), ("All engagement", "all_eng")]}
rob = {"rows": [], "corr": {}}
for k in KEY:
    rob["rows"].append({"term": k, "Likes": round(float(base_m[k]), 1), "Likes, extreme lows dropped": round(float(trim_m[k]), 1),
                        **{m: round(float(v[k]), 1) for m, v in alt.items()}})
for m, v in alt.items():
    rob["corr"][m] = round(float(np.corrcoef(base_m.values, v.loc[base_m.index].values)[0, 1]), 2)
w = d[(d.platform == "Twitter") & d.lift_log.notna() & (d.ts >= "2015-06-16") & (d.ts < "2017-03-01")].copy()
w["android"] = w.device.fillna("").str.contains("Android"); w = w[w.android | w.device.fillna("").str.contains("iPhone")]
w["excl"] = w.exclaims > 0
dev = w.groupby("android").agg(n=("excl", "size"), excl=("excl", "mean"), insult=("insult", "mean"), lift=("lift_log", lambda x: float(np.exp(x.mean()))))
r1 = sm.OLS(w.lift_log, sm.add_constant(w[["excl"]].astype(float))).fit(cov_type="HC3")
r2 = sm.OLS(w.lift_log, sm.add_constant(w[["excl", "android"]].astype(float))).fit(cov_type="HC3")
rob["device"] = {"window": "16 Jun 2015 – 28 Feb 2017, Twitter",
                 "android": {"n": int(dev.loc[True, "n"]), "lift": round(dev.loc[True, "lift"], 2), "insult": round(float(dev.loc[True, "insult"]), 3), "excl": round(float(dev.loc[True, "excl"]), 3)},
                 "iphone": {"n": int(dev.loc[False, "n"]), "lift": round(dev.loc[False, "lift"], 2), "insult": round(float(dev.loc[False, "insult"]), 3), "excl": round(float(dev.loc[False, "excl"]), 3)},
                 "excl_effect_alone": round(float((np.exp(r1.params["excl"]) - 1) * 100), 1),
                 "excl_effect_with_device": round(float((np.exp(r2.params["excl"]) - 1) * 100), 1)}
json.dump(rob, open("out/robustness.json", "w"), indent=1)
print(pd.DataFrame(rob["rows"]).to_string()); print(rob["corr"]); print(json.dumps(rob["device"], indent=1))

