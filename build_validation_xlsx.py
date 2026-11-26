"""Builds dist/validation_coding.xlsx: blind hand-coding sheet + auto-calculated lexicon agreement."""
import pandas as pd
import postlift as pl
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter as L

themes = list(pl.THEMES)
v = pd.read_csv("out/validation_sample.csv", dtype={"post_id": str})
v = v.sample(frac=1, random_state=11).reset_index(drop=True)  # mix platforms
# Re-run the exact pipeline coder on the sample text (Twitter IDs in the export are rounded, so no ID join)
lex = pl.add_features(pd.DataFrame({"text": v["text"].fillna(""), "ts": pd.to_datetime(v["ts"], utc=True, format="ISO8601")}))
N = len(v); last = N + 1

F = "Arial"
def f(**k): return Font(name=F, **k)
hdr = PatternFill("solid", fgColor="1F3A5F"); inp = PatternFill("solid", fgColor="FFF2CC"); grey = PatternFill("solid", fgColor="F2F2F2")
thin = Side(style="thin", color="D9D9D9"); box = Border(left=thin, right=thin, top=thin, bottom=thin)
wb = Workbook()

# ---- How to -------------------------------------------------------------
h = wb.active; h.title = "How to"
lines = [
    ("Hand-coding the topic lexicon: 200-post validation sample", f(bold=True, size=14)), ("", None),
    ("What you're doing", f(bold=True, size=11)),
    ("Read each post on the 'Code' sheet and mark which themes it is genuinely ABOUT. Use the 'Codebook' sheet for definitions. "
     "Code blind: the keyword coder's answers are on a hidden sheet, so don't unhide it until you've finished.", f()), ("", None),
    ("Which cells to edit (yellow)", f(bold=True, size=11)),
    ("• Theme columns: pick 'x' from the dropdown (or type x) for every theme that applies. A post can have several themes, or none.", f()),
    ("• 'Reviewed' column: set to Y once you've finished a row, even if you ticked no themes. Only rows marked Y count in the results.", f()),
    ("• 'Notes': optional. Jot down tricky cases; these make good material for the method note.", f()), ("", None),
    ("Example row (format only; not part of the sample)", f(bold=True, size=11)),
    ("Post: \"Iran has agreed to a CEASEFIRE. The Fake News won't report it! Peace through strength.\"", f(italic=True)),
    ("→ x under 'Foreign policy & war' and x under 'Attacks on media'; Reviewed = Y; Notes: 'peace deal announcement'.", f()), ("", None),
    ("Results", f(bold=True, size=11)),
    ("The 'Results' sheet updates as you go. Precision = of the posts the lexicon tagged, how many you agreed with. "
     "Recall = of the posts you tagged, how many the lexicon caught. F1 combines the two. Treat themes you tagged fewer than 10 times as too small to judge.", f()),
    ("For the method note, report the overall (micro-average) precision and recall, plus the best and worst themes.", f()), ("", None),
    ("Tip: aim for about 25 seconds a post. Code what the post is about, not single words: 'God bless' is not a faith post.", f(italic=True, color="555555")),
]
for i, (t, ft) in enumerate(lines, 1):
    c = h.cell(row=i, column=1, value=t)
    if ft: c.font = ft
    c.alignment = Alignment(wrap_text=True, vertical="top")
h.column_dimensions["A"].width = 110

# ---- Code ----------------------------------------------------------------
c = wb.create_sheet("Code")
cols = ["#", "Platform", "Date", "Post text"] + themes + ["Reviewed", "Notes"]
T0 = 5; TR = T0 + len(themes) - 1; RV = TR + 1; NT = RV + 1
for j, name in enumerate(cols, 1):
    x = c.cell(row=1, column=j, value=name); x.font = f(bold=True, color="FFFFFF"); x.fill = hdr
    x.alignment = Alignment(wrap_text=True, vertical="bottom", horizontal="center", text_rotation=90 if T0 <= j <= TR else 0)
c.row_dimensions[1].height = 170
for i, r in v.iterrows():
    rr = i + 2
    c.cell(row=rr, column=1, value=i + 1); c.cell(row=rr, column=2, value=r.platform)
    c.cell(row=rr, column=3, value=str(r.ts)[:10]); c.cell(row=rr, column=4, value=r.text)
    for j in range(1, NT + 1):
        x = c.cell(row=rr, column=j); x.font = f(size=10); x.border = box
        x.alignment = Alignment(wrap_text=j in (4, NT), vertical="top", horizontal="center" if T0 <= j <= RV else "left")
        if T0 <= j <= NT: x.fill = inp
    c.row_dimensions[rr].height = max(30, min(150, 14 * (len(str(r.text)) // 75 + 1)))
for j, w in zip(range(1, 5), [5, 12, 11, 70]): c.column_dimensions[L(j)].width = w
for j in range(T0, RV + 1): c.column_dimensions[L(j)].width = 4.5
c.column_dimensions[L(NT)].width = 30
c.freeze_panes = "E2"
dv = DataValidation(type="list", formula1='"x"', allow_blank=True, showErrorMessage=True, errorTitle="Use x", error="Leave blank or enter x")
c.add_data_validation(dv); dv.add(f"{L(T0)}2:{L(TR)}{last}")
dv2 = DataValidation(type="list", formula1='"Y"', allow_blank=True); c.add_data_validation(dv2); dv2.add(f"{L(RV)}2:{L(RV)}{last}")
c.conditional_formatting.add(f"{L(T0)}2:{L(TR)}{last}", FormulaRule(formula=[f'{L(T0)}2="x"'], fill=PatternFill("solid", fgColor="9BC2E6")))
c.conditional_formatting.add(f"A2:{L(NT)}{last}", FormulaRule(formula=[f'${L(RV)}2="Y"'], font=Font(color="7F7F7F")))
c.auto_filter.ref = f"A1:{L(NT)}{last}"

# ---- Codebook ------------------------------------------------------------
defs = {
    "Immigration & border": "Border security, illegal immigration, deportations, migrants, asylum, cartels at the border.",
    "Economy, jobs & prices": "Jobs, inflation, prices, taxes, tariffs and trade, stock market, manufacturing, interest rates.",
    "Elections & fraud claims": "Elections, polls, voting, ballots, claims of rigging or fraud. Not endorsements (separate theme).",
    "Legal fights & 'witch hunt'": "Investigations, indictments, trials, judges, DOJ/FBI, impeachment, 'witch hunt' or 'hoax' framing.",
    "Attacks on media": "Criticism of news outlets or 'Fake News'. Neutral 'watch me on Fox' plugs belong under Celebrity & business.",
    "Attacks on opponents": "Criticism, insults or nicknames aimed at named political opponents or 'Democrats / Radical Left'.",
    "Foreign policy & war": "Other countries, wars, ceasefires, NATO, China/Russia/Iran/Israel/Ukraine.",
    "Crime & law and order": "Crime, police, riots, gangs, drugs and fentanyl, 'law and order'.",
    "Faith, values & culture war": "Religion, abortion, guns, gender or 'woke' issues, traditional values. Not a 'God bless' sign-off on its own.",
    "MAGA identity & movement": "MAGA / America First identity, 'our movement', patriots, the forgotten man.",
    "Endorsements": "Endorsing a candidate ('Complete and Total Endorsement', 'has my strong endorsement').",
    "Rallies & crowds": "Rallies, crowd sizes, event announcements ('see you tonight').",
    "Health, MAHA & COVID": "COVID, vaccines, healthcare, drug prices, RFK Jr / MAHA, Obamacare.",
    "Tech, crypto & new media": "Crypto, AI, tech companies, Musk, podcasts, Rogan, UFC, TikTok, Truth Social itself.",
    "Celebrity & business": "Apprentice, golf, Trump properties and brands, books, TV appearances, interview plugs, self-promotion.",
}
cb = wb.create_sheet("Codebook")
for j, t in enumerate(["Theme", "Code it when the post is about…"], 1):
    x = cb.cell(row=1, column=j, value=t); x.font = f(bold=True, color="FFFFFF"); x.fill = hdr
for i, t in enumerate(themes, 2):
    cb.cell(row=i, column=1, value=t).font = f(bold=True); cb.cell(row=i, column=2, value=defs[t]).font = f()
    for j in (1, 2): cb.cell(row=i, column=j).alignment = Alignment(wrap_text=True, vertical="top")
cb.column_dimensions["A"].width = 28; cb.column_dimensions["B"].width = 90; cb.freeze_panes = "A2"

# ---- Lexicon (hidden) ----------------------------------------------------
lx = wb.create_sheet("Lexicon")
lx.cell(row=1, column=1, value="#").font = f(bold=True)
for j, t in enumerate(themes, 2): lx.cell(row=1, column=j, value=t).font = f(bold=True)
for i in range(N):
    lx.cell(row=i + 2, column=1, value=i + 1)
    for j, t in enumerate(themes, 2): lx.cell(row=i + 2, column=j, value=int(bool(lex.loc[i, f"t::{t}"])))
lx.sheet_state = "hidden"

# ---- Results ---------------------------------------------------------------
rs = wb.create_sheet("Results", 1)
rs["A1"] = "Lexicon vs hand coding"; rs["A1"].font = f(bold=True, size=14)
rs["A2"] = f'=COUNTIF(Code!{L(RV)}2:{L(RV)}{last},"Y")&" of {N} posts reviewed"'; rs["A2"].font = f(italic=True)
heads = ["Theme", "Hand-coded n", "Lexicon n", "Both (TP)", "Lexicon only (FP)", "Hand only (FN)", "Precision", "Recall", "F1"]
for j, t in enumerate(heads, 1):
    x = rs.cell(row=4, column=j, value=t); x.font = f(bold=True, color="FFFFFF"); x.fill = hdr
    x.alignment = Alignment(wrap_text=True, horizontal="center")
rev = f'(Code!${L(RV)}$2:${L(RV)}${last}="Y")'
for i, t in enumerate(themes):
    r = 5 + i
    hc = f'(Code!{L(T0 + i)}$2:{L(T0 + i)}${last}="x")'; lc = f"(Lexicon!{L(2 + i)}$2:{L(2 + i)}${last}=1)"
    rs.cell(row=r, column=1, value=t)
    rs.cell(row=r, column=2, value=f"=SUMPRODUCT({rev}*{hc})")
    rs.cell(row=r, column=3, value=f"=SUMPRODUCT({rev}*{lc})")
    rs.cell(row=r, column=4, value=f"=SUMPRODUCT({rev}*{hc}*{lc})")
    rs.cell(row=r, column=5, value=f"=C{r}-D{r}")
    rs.cell(row=r, column=6, value=f"=B{r}-D{r}")
    rs.cell(row=r, column=7, value=f'=IF(C{r}=0,"–",D{r}/C{r})')
    rs.cell(row=r, column=8, value=f'=IF(B{r}=0,"–",D{r}/B{r})')
    rs.cell(row=r, column=9, value=f'=IF(B{r}+C{r}=0,"–",2*D{r}/(B{r}+C{r}))')
end = 4 + len(themes); tr = end + 1
rs.cell(row=tr, column=1, value="All themes (micro-average)")
for col in "BCDEF": rs[f"{col}{tr}"] = f"=SUM({col}5:{col}{end})"
rs[f"G{tr}"] = f'=IF(C{tr}=0,"–",D{tr}/C{tr})'; rs[f"H{tr}"] = f'=IF(B{tr}=0,"–",D{tr}/B{tr})'
rs[f"I{tr}"] = f'=IF(B{tr}+C{tr}=0,"–",2*D{tr}/(B{tr}+C{tr}))'
for r in range(5, tr + 1):
    for j in range(1, 10):
        x = rs.cell(row=r, column=j); x.border = box; x.font = f(bold=(r == tr))
        if j >= 7: x.number_format = "0%"
        if j > 1: x.alignment = Alignment(horizontal="center")
        if r == tr: x.fill = grey
rs.column_dimensions["A"].width = 30
for j in range(2, 10): rs.column_dimensions[L(j)].width = 12
notes = ["Precision: of the posts the lexicon tagged with a theme, the share you agreed with (low means the keywords over-fire).",
         "Recall: of the posts you tagged, the share the lexicon also caught (low means the keywords miss things).",
         "Only rows marked Y in Code!Reviewed count. Treat themes with Hand-coded n under 10 as too small to judge.",
         "Note: hand-coded n, lexicon n, precision and recall are all entered by formula; nothing on this sheet is typed in."]
for k, t in enumerate(notes): rs.cell(row=tr + 2 + k, column=1, value=t).font = f(size=9, italic=True, color="555555")
rs.freeze_panes = "B5"
wb.save("dist/validation_coding.xlsx")
print("saved; lexicon tags in sample:", int(lex[[f"t::{t}" for t in themes]].values.sum()))
