"""Run postlift's coder on a sample of Congress tweets (alexlitel/congresstweets, 2 days/month, Jul 2017–Jul 2023)
and compare style/topic profiles with Trump's own tweets over the same window."""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import postlift as pl

HERE = Path(__file__).resolve().parent

# ---- 1. member -> party map (unitedstates/congress-legislators) ----------------
leg = yaml.safe_load(open(HERE / "legislators-current.yaml")) + yaml.safe_load(open(HERE / "legislators-historical.yaml"))
info = {}
for m in leg:
    terms = [t for t in m["terms"] if t["end"] >= "2017-01-01"]
    if not terms: continue
    t = terms[-1]
    info[m["id"]["bioguide"]] = {"name": m["name"].get("official_full") or f'{m["name"]["first"]} {m["name"]["last"]}',
                                 "party": t.get("party"), "chamber": "Senate" if t["type"] == "sen" else "House"}
sm = yaml.safe_load(open(HERE / "legislators-social-media.yaml"))
by_id, by_name = {}, {}
for s in sm:
    b = s["id"]["bioguide"]
    if b not in info: continue
    if s["social"].get("twitter_id"): by_id[str(s["social"]["twitter_id"])] = b
    if s["social"].get("twitter"): by_name[s["social"]["twitter"].lower()] = b
# well-known personal/campaign accounts not in the official-account file
MANUAL = {"aoc": "O000172", "tedcruz": "C001098", "mtgreenee": "G000596", "realmtgreenee": "G000596",
          "laurenboebert": "B000825", "mattgaetz": "G000578", "hawleymo": "H001089", "berniesanders": "S000033",
          "ewarren": "W000817", "marcorubio": "R000595", "lindseygrahamsc": "G000359", "gopleader": "M001165",
          "speakerpelosi": "P000197", "senschumer": "S000148", "leadermcconnell": "M000355", "jim_jordan": "J000289",
          "replizcheney": "C001109", "ilhanmn": "O000173", "repmtg": "G000596", "speakermccarthy": "M001165"}
for k, b in MANUAL.items():
    if b in info: by_name.setdefault(k, b)

# ---- 2. load sampled days ----------------------------------------------------------
rows = []
for f in sorted(glob.glob(str(HERE / "raw/*.json"))):
    for t in json.load(open(f)):
        b = by_id.get(str(t["user_id"])) or by_name.get(t["screen_name"].lower())
        if not b: continue
        rows.append((t["screen_name"], b, t["time"], t["text"]))
c = pd.DataFrame(rows, columns=["screen_name", "bioguide", "time", "text"])
c["ts"] = pd.to_datetime(c["time"], utc=True, format="ISO8601")
c = c[~c.text.str.match(r"^RT @")].copy()           # own posts only
c["text"] = c.text.map(pl._clean_text)
# the archive inlines quoted tweets ("<link> QT @user <their text>") and media URLs; keep only the member's own words
c["text"] = (c.text.str.replace(r"\s*https?://twitter\.com/\S+\s+QT @\w+.*$", "", regex=True)
                   .str.replace(r"https?://(?:pbs|video)\.twimg\.com/\S+", "", regex=True).str.strip())
c = c[c.text.str.len() > 0]
c = c[~c.text.str.match(r"^\.?@\w+")]                 # reply-led, like Trump's
for k in ("name", "party", "chamber"): c[k] = c.bioguide.map(lambda b: info[b][k])
c = pl.add_features(c.reset_index(drop=True))
c["platform"] = "Congress"

# Symmetric "names the other side" measure (the Trump lexicon only lists Democratic opponents)
OTHER = {"Republican": r"\b(?:biden|kamala|harris|obama|hillary|clinton|pelosi|schumer|democrats?|radical left|the left|socialis\w*|aoc|squad)\b",
         "Democrat": r"\b(?:trump|republicans?|gop|mcconnell|mccarthy|maga|desantis|far[- ]right|mtg|jim jordan|insurrectionists?)\b",
         "Independent": r"\b(?:trump|republicans?|gop|mcconnell|mccarthy|maga)\b"}
_w = c.text.str.replace(pl.URL_RX, " ", regex=True)
c["names_other"] = [bool(pd.Series([w]).str.contains(OTHER.get(p, OTHER["Independent"]), case=False, regex=True)[0]) for w, p in zip(_w, c.party)]

# ---- 3. Trump, same window, same rules ---------------------------------------------
d = pd.read_parquet(HERE.parent / "out/posts.parquet")
tr = d[(d.platform == "Twitter") & (d.post_type == "Original") & (d.ts >= "2017-07-01") & (d.ts < "2021-01-09")].copy()

tr["names_other"] = tr.text.str.replace(pl.URL_RX, " ", regex=True).str.contains(OTHER["Republican"], case=False, regex=True)
STYLE = {"caps": ("caps_share", "Share of words in ALL-CAPS"), "excl": ("exclaim_any", "Posts with an exclamation mark"),
         "insult": ("insult", "Posts with an insult or nickname"), "other": ("names_other", "Posts naming the other side"), "link": ("has_link", "Posts with an outside link")}
for df in (c, tr): df["exclaim_any"] = df.exclaims > 0; df["has_link"] = df.text.str.contains(r"https?://", regex=True)
THEMES = [t for t in pl.THEMES]

def profile(g):
    out = {k: float(g[col].mean()) for k, (col, _) in STYLE.items()}
    for t in THEMES: out["t::" + t] = float(g["t::" + t].mean())
    out["n"] = int(len(g)); return out

T = profile(tr)
# Voice index: mean of (rate / Trump's 2017–20 rate) for ALL-CAPS share, insults/nicknames and exclamation marks; x100,
# so 100 = Trump's own voice. Targets ("names the other side") are reported separately.
def tsi(p): return float(np.mean([p["caps"] / T["caps"], p["insult"] / T["insult"], p["excl"] / T["excl"]]) * 100)

R = {"window": [str(c.ts.min().date()), str(c.ts.max().date())], "n_days": len(glob.glob(str(HERE / "raw/*.json"))),
     "n_congress_posts": int(len(c)), "n_members": int(c.bioguide.nunique()), "n_trump": int(len(tr)), "trump": T}
parties = c[c.party.isin(["Republican", "Democrat"])]
R["party"] = {p: profile(g) | {"tsi": tsi(profile(g))} for p, g in parties.groupby("party")}
R["trump"]["tsi"] = 100.0
# per quarter, by party (Congress) and Trump
parties = parties.assign(q=parties.ts.dt.tz_convert(None).dt.to_period("Q").astype(str))
qq = []
for (q, p), g in parties.groupby(["q", "party"]):
    if len(g) >= 300: qq.append({"q": q, "party": p, "tsi": tsi(profile(g)), "n": len(g), "insult": float(g.insult.mean()), "caps": float(g.caps_share.mean()),
                                 "opp": float(g["t::Attacks on opponents"].mean()), "other": float(g.names_other.mean())})
R["quarterly"] = qq
# named senior figures (members with >= 150 sampled posts)
FIG = ["O000172", "C001098", "G000596", "S000148", "M000355", "P000197", "M001165", "J000289", "S000033", "W000817",
       "R000595", "G000359", "H001089", "B000825", "G000578", "C001109", "O000173"]
figs = []
for b, g in c.groupby("bioguide"):
    if len(g) < 150: continue
    p = profile(g); figs.append({"bioguide": b, "name": info[b]["name"], "party": info[b]["party"], "chamber": info[b]["chamber"],
                                 "n": len(g), "tsi": tsi(p), "caps": p["caps"], "excl": p["excl"], "insult": p["insult"], "opp": p["t::Attacks on opponents"], "other": p["other"],
                                 "named": b in FIG})
figs = pd.DataFrame(figs).sort_values("tsi", ascending=False)
R["members_ranked"] = figs.round(4).to_dict(orient="records")
R["named"] = figs[figs.named].round(4).to_dict(orient="records")
R["share_members_above_50"] = {p: float((g.tsi >= 50).mean()) for p, g in figs.groupby("party")}
R["n_members_ranked"] = int(len(figs))
json.dump(R, open(HERE / "congress_results.json", "w"), indent=1, default=str)

print(R["window"], R["n_days"], R["n_congress_posts"], R["n_members"], R["n_trump"])
print(pd.DataFrame({k: v for k, v in R["party"].items()} | {"Trump": T}).loc[["n", "tsi", "caps", "excl", "insult", "other"]].round(3))
th = pd.DataFrame({k: {t: v["t::" + t] for t in THEMES} for k, v in (R["party"] | {"Trump": T}).items()}).round(3)
print(th.sort_values("Trump", ascending=False))
print(figs.head(15)[["name", "party", "n", "tsi", "caps", "excl", "insult", "opp"]].round(3).to_string())
print(figs[figs.named][["name", "party", "n", "tsi", "caps", "excl", "insult", "other"]].round(3).to_string())
print(pd.DataFrame(qq).pivot(index="q", columns="party", values="tsi").round(1).T.to_string())
print(pd.DataFrame(qq).pivot(index="q", columns="party", values="other").round(3).T.to_string())
print("trump other", T["other"])
