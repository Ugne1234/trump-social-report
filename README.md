# postlift: growth-adjusted engagement analysis for politicians' social posts

Built for the Knox / Atlantic Data Institute assessment (Trump on Twitter 2009–21 and Truth Social 2022–26).

## Why
Raw likes measure audience size, not content. Trump's median tweet went from about 79 likes (2013) to about 165,000 (Nov 2020).
`postlift` scores each post against the same account's typical engagement in the surrounding 30 days:

    lift = log(1 + likes) − rolling median of log(1 + likes), ±15 days, same platform

So 2.0× means twice the likes of a typical post at that point in the account's history.

## Run
```bash
pip install pandas numpy statsmodels pyarrow
python postlift.py --twitter trump_tweets.csv --truth trump_posts_full.csv --out out/
python analysis.py        # main report figures -> out/report_data.json
python robustness.py      # robustness and device checks -> out/robustness.json
python build_report.py    # -> report/index.html
```

## What it does
1. Harmonises both exports and classifies post types: Original, Repost, Quote-post, Reply, Quoted fan (2012–15 manual RTs), Media only, Link only.
2. Repairs data problems: double-escaped quotes, UTF-8 encoding errors, the broken `is_retruth` flag, zero-like capture gaps.
3. Codes 15 themes with an editable keyword lexicon (`THEMES` at the top of `postlift.py`), plus style flags computed identically on both platforms. Endorsement templates are coded only as endorsements.
4. Fits OLS on log lift with HC3 robust errors (`fit_model`) and exports `validation_sample.csv` (200 posts, stratified by platform) for hand-coding.

## Files
- `postlift.py`: the tool (module and command line)
- `analysis.py`: the main numbers and chart datasets in the report
- `robustness.py`: the checks in the report's "Three checks" paragraph: shares / replies / total-engagement models, the extreme-lows drop, and Android vs iPhone (2015–17)
- `llm600_compare.py` + `llm600_codes.txt`: the third check, a 600-post LLM-coded rerun of the four headline themes -> `out/llm600_results.json`
- `llm_codes.py`: the LLM pilot on the 200 hand-coded posts -> `out/llm_vs_lexicon.json`
- `report/template.html` + `build_report.py`: the web report, including the in-browser post checker (same lexicon and coefficients)
- `out/validation_sample.csv`: fill in `hand_themes` to measure the lexicon's precision and recall

## Pointing it at another politician
Write a loader that returns the same columns as `load_truth()` (platform, post_id, ts, text, likes, shares, replies, deleted, flagged, post_type), then call `add_features` → `add_lift` → `theme_table` / `fit_model`.

## Extensions
- `congress/congress.py`: runs the same coder on a sample of Congress tweets (alexlitel/congresstweets, 2 days/month,
  Jul 2017–Jul 2023) matched to party via unitedstates/congress-legislators; compares voice (caps, exclamation, insults)
  and targets (a symmetric "names the other side" measure) with Trump. Download the daily JSON files into `congress/raw/`
  and the three legislators YAML files into `congress/` first.
- `llm_codes.py`: scores an LLM coder and the keyword lexicon against the 200 hand codes (precision, recall, F1).
- Themes are coded on the words written: URLs are stripped first, because article slugs otherwise add ~10% spurious tags.

## Where each claim in the report comes from
| Claim | Script | Output |
|---|---|---|
| Lift, theme/style/format effects, eras, volume, Newsom, coalition figures | `analysis.py` | `out/report_data.json` |
| Shares/replies/total engagement, extreme lows, Android vs iPhone | `robustness.py` | `out/robustness.json` |
| Keyword vs hand codes (P/R/F1), LLM pilot | `llm_codes.py` | `out/llm_vs_lexicon.json` |
| 600-post LLM rerun | `llm600_compare.py` | `out/llm600_results.json` |
| Congress comparison | `congress/congress.py` | `congress/congress_results.json` |
