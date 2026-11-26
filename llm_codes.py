"""Claude (Opus 5.5) coding of the 200 validation posts, done by reading each post against the same codebook,
without seeing the hand codes. Scored against Ugne's hand codes alongside the keyword lexicon."""
import json, openpyxl, pandas as pd, postlift as pl

A = {"IMM": "Immigration & border", "ECO": "Economy, jobs & prices", "ELE": "Elections & fraud claims",
     "LEG": "Legal fights & 'witch hunt'", "MED": "Attacks on media", "OPP": "Attacks on opponents",
     "FOR": "Foreign policy & war", "CRI": "Crime & law and order", "FAI": "Faith, values & culture war",
     "MAGA": "MAGA identity & movement", "END": "Endorsements", "RAL": "Rallies & crowds",
     "HEA": "Health, MAHA & COVID", "TEC": "Tech, crypto & new media", "CEL": "Celebrity & business"}
CODES = {
1:"OPP ELE",2:"LEG OPP",3:"ELE",4:"",5:"LEG ELE OPP",6:"ELE",7:"CRI OPP ECO",8:"FOR",9:"",10:"ECO",
11:"CEL",12:"LEG",13:"ELE",14:"ECO",15:"OPP IMM ECO",16:"LEG",17:"OPP",18:"LEG",19:"",20:"",
21:"MED FOR",22:"CEL",23:"LEG",24:"ECO FOR",25:"TEC OPP",26:"",27:"ELE",28:"OPP ECO",29:"FOR OPP",30:"ELE",
31:"MED FOR",32:"IMM CRI OPP",33:"RAL",34:"ECO",35:"",36:"LEG",37:"TEC ELE",38:"RAL",39:"CEL",40:"OPP ECO",
41:"FOR",42:"",43:"END",44:"",45:"OPP ECO",46:"MED FOR LEG",47:"MED",48:"ELE",49:"",50:"LEG OPP ELE",
51:"IMM",52:"ELE",53:"HEA ECO",54:"ELE",55:"CEL",56:"CEL",57:"OPP MED",58:"CEL OPP",59:"FOR",60:"ELE MED IMM",
61:"MED",62:"FOR",63:"OPP",64:"",65:"OPP LEG",66:"OPP MED",67:"",68:"OPP ELE",69:"",70:"OPP",
71:"LEG",72:"OPP",73:"FOR",74:"CEL",75:"OPP",76:"OPP FOR",77:"ECO",78:"FOR",79:"",80:"ELE",
81:"LEG",82:"RAL",83:"CEL",84:"",85:"CRI IMM",86:"OPP HEA CRI",87:"OPP HEA",88:"",89:"MED",90:"END",
91:"",92:"ECO FOR",93:"RAL",94:"FOR",95:"END OPP IMM",96:"ELE",97:"",98:"OPP ECO ELE",99:"CRI OPP IMM ELE LEG",100:"OPP",
101:"OPP ECO",102:"END ELE",103:"CEL ELE",104:"ECO",105:"",106:"OPP CRI ELE",107:"ECO",108:"OPP",109:"",110:"LEG OPP",
111:"ECO",112:"ELE OPP",113:"",114:"ELE",115:"LEG OPP ELE",116:"MED",117:"RAL",118:"TEC",119:"LEG OPP",120:"CEL",
121:"MED",122:"",123:"ECO OPP ELE",124:"LEG",125:"",126:"OPP",127:"LEG ELE",128:"MED",129:"ECO",130:"FOR ECO",
131:"ELE",132:"",133:"OPP HEA",134:"ELE",135:"END",136:"FOR LEG",137:"MED ELE",138:"MED",139:"",140:"",
141:"CRI",142:"MED",143:"",144:"CEL",145:"LEG",146:"MED OPP",147:"OPP",148:"END",149:"OPP ELE",150:"FOR ECO",
151:"FAI",152:"RAL",153:"CEL",154:"",155:"OPP",156:"ELE",157:"LEG",158:"",159:"ELE OPP",160:"FOR",
161:"",162:"CEL",163:"CEL",164:"ELE",165:"CEL",166:"ELE LEG",167:"LEG",168:"OPP",169:"FOR OPP ECO",170:"IMM CRI OPP",
171:"MED",172:"CEL",173:"END",174:"",175:"ELE OPP",176:"MED ELE",177:"",178:"OPP",179:"END",180:"",
181:"LEG",182:"CRI",183:"HEA",184:"ECO",185:"RAL",186:"MED",187:"FOR",188:"OPP",189:"CEL",190:"END",
191:"OPP",192:"ECO",193:"ELE",194:"MED OPP",195:"RAL",196:"LEG",197:"END",198:"CEL",199:"CEL",200:"CEL ELE"}
assert len(CODES) == 200

wb = openpyxl.load_workbook("out/validation_coding.xlsx", data_only=True); c = wb["Code"]
hdr = [x.value for x in c[1]]
df = pd.DataFrame([[x.value for x in r] for r in c.iter_rows(min_row=2)], columns=hdr)
themes = list(pl.THEMES)
lex = pl.add_features(pd.DataFrame({"text": df["Post text"].fillna("").values, "ts": pd.Timestamp("2020-01-01", tz="UTC")}))
llm = pd.DataFrame(False, index=df.index, columns=themes)
for i, r in df.iterrows():
    for a in CODES[int(r["#"])].split():
        llm.loc[i, A[a]] = True

def score(pred):
    rows, T = [], dict(tp=0, fp=0, fn=0)
    for t in themes:
        h = (df[t] == "x").values; p = pred[t].values
        tp, fp, fn = int((h & p).sum()), int((~h & p).sum()), int((h & ~p).sum())
        for k, v in zip(("tp", "fp", "fn"), (tp, fp, fn)): T[k] += v
        rows.append(dict(theme=t, hand=int(h.sum()), tp=tp, fp=fp, fn=fn, P=tp / max(1, tp + fp), R=tp / max(1, tp + fn),
                         F1=2 * tp / max(1, 2 * tp + fp + fn)))
    P = T["tp"] / (T["tp"] + T["fp"]); R = T["tp"] / (T["tp"] + T["fn"])
    return pd.DataFrame(rows), dict(P=P, R=R, F1=2 * P * R / (P + R), **T)

lr, lm = score(lex.rename(columns=lambda c: c[3:] if c.startswith("t::") else c)[themes])
mr, mm = score(llm)
out = lr[["theme", "hand"]].assign(lex_F1=lr.F1, llm_F1=mr.F1, lex_R=lr.R, llm_R=mr.R, lex_P=lr.P, llm_P=mr.P)
print(out.round(2).to_string())
print("LEXICON", {k: round(v, 3) for k, v in lm.items()})
print("LLM    ", {k: round(v, 3) for k, v in mm.items()})
# post-level exact agreement (all 15 themes identical)
ex_l = ((lex.rename(columns=lambda c: c[3:] if c.startswith("t::") else c)[themes].values) == (df[themes] == "x").values).all(1).mean()
ex_m = (llm.values == (df[themes] == "x").values).all(1).mean()
print("posts coded identically to hand: lexicon %.2f  llm %.2f" % (ex_l, ex_m))
json.dump({"lexicon": lm, "llm": mm, "by_theme": out.round(4).to_dict(orient="records"), "exact": {"lexicon": ex_l, "llm": ex_m}},
          open("out/llm_vs_lexicon.json", "w"), indent=1)
