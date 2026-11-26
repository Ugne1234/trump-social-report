import json
R = json.load(open('out/report_data_web.json'))
R['robustness'] = json.load(open('out/robustness.json'))
C = json.load(open('congress/congress_results.json'))
R['congress'] = {k: C[k] for k in ['window', 'n_days', 'n_congress_posts', 'n_members', 'n_members_ranked', 'trump', 'party', 'quarterly', 'named']}
D = json.dumps(R, default=str)
t = open('report/template.html').read().replace('/*DATA*/null', D.replace('</', '<\\/'))
open('report/index.html', 'w').write(t); print(len(t) // 1024, 'KB')
