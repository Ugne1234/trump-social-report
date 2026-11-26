"""Robustness: rerun the headline theme model on a 600-post Truth Social subsample (2022–26) coded by an LLM
(Claude, reading each post against the codebook) for four themes, and compare with the keyword lexicon."""
import json
import numpy as np, pandas as pd, statsmodels.api as sm
import postlift as pl

d = pd.read_parquet("out/posts.parquet")
ids = pd.read_csv("out/llm600_ids.csv")
sub = d.loc[ids["index"].values].reset_index(drop=True)
T = {"W": "Foreign policy & war", "E": "Economy, jobs & prices", "O": "Attacks on opponents", "N": "Endorsements"}
llm = pd.DataFrame(False, index=sub.index, columns=list(T.values()))
for line in open("llm600_codes.txt"):
    parts = line.split()
    if not parts: continue
    i = int(parts[0]) - 1
    for c in parts[1:]: llm.loc[i, T[c]] = True

agree = []
for t in T.values():
    l, m = sub["t::" + t].values, llm[t].values
    tp = (l & m).sum(); agree.append({"theme": t, "llm_n": int(m.sum()), "lexicon_n": int(l.sum()),
        "both": int(tp), "lexicon_precision_vs_llm": round(tp / max(1, l.sum()), 2), "lexicon_recall_vs_llm": round(tp / max(1, m.sum()), 2)})

def fit(themes):
    X = themes.astype(float).rename(columns=lambda c: "Theme: " + c)
    X["Style: exclamation (any)"] = (sub.exclaims > 0).astype(float)
    X["Style: insult / nickname"] = sub.insult.astype(float)
    X["Style: ALL-CAPS share (per +10pp)"] = sub.caps_share * 10
    X["Format: contains link"] = sub.has_link.astype(float)
    lb = pd.get_dummies(sub.length_band, prefix="Length", dtype=float)
    lb = lb.drop(columns=[c for c in lb.columns if "Short (16–40)" in c or lb[c].sum() == 0])
    X = sm.add_constant(pd.concat([X, lb], axis=1))
    r = sm.OLS(sub.lift_log, X).fit(cov_type="HC3"); ci = r.conf_int()
    return pd.DataFrame({"pct": (np.exp(r.params) - 1) * 100, "lo": (np.exp(ci[0]) - 1) * 100, "hi": (np.exp(ci[1]) - 1) * 100, "p": r.pvalues})

lex_m = fit(sub[["t::" + t for t in T.values()]].rename(columns=lambda c: c[3:]))
llm_m = fit(llm)
full = pl.fit_model(d, "Truth Social", ["Out of office (2022–Jan 2025)", "President, 2nd term (2025–26)"]).set_index("term")
rows = []
for t in T.values():
    k = "Theme: " + t
    rows.append({"theme": t, "full_lexicon_pct": round(float(full.loc[k, "pct"]), 1),
                 "sub_lexicon_pct": round(float(lex_m.loc[k, "pct"]), 1), "sub_lexicon_ci": [round(float(lex_m.loc[k, "lo"]), 0), round(float(lex_m.loc[k, "hi"]), 0)],
                 "sub_llm_pct": round(float(llm_m.loc[k, "pct"]), 1), "sub_llm_ci": [round(float(llm_m.loc[k, "lo"]), 0), round(float(llm_m.loc[k, "hi"]), 0)],
                 "sub_llm_p": round(float(llm_m.loc[k, "p"]), 3)})
# simple descriptive: mean lift of posts with vs without each LLM theme
desc = {t: round(float(np.exp(sub.lift_log[llm[t]].mean() - sub.lift_log[~llm[t]].mean())), 2) for t in T.values()}
out = {"n": len(sub), "agreement": agree, "models": rows, "llm_lift_vs_rest": desc}
json.dump(out, open("out/llm600_results.json", "w"), indent=1, default=float)
print(pd.DataFrame(agree).to_string()); print(pd.DataFrame(rows).to_string()); print(desc)
