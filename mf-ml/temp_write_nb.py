import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.cells = [
    new_markdown_cell('# Monthly IC analysis for mutual fund predictors\n\nThis notebook reads the IC summary and monthly panel from CSVs, selects the top 25 PASS features, computes their pairwise Spearman correlations, and reports the results.'),
    new_code_cell("import os\nimport numpy as np\nimport pandas as pd\nfrom scipy.stats import spearmanr\n"),
    new_markdown_cell('## Read IC summary and monthly panel\n\nEnsure the summary (ic_summary_all_features.csv) and monthly panel (ic_monthly_all_features.csv) are present.'),
    new_code_cell("# paths\nsummary_path = 'ic_summary_all_features.csv'\nmonthly_panel_path = 'ic_monthly_all_features.csv'\n\n# read files\nsummary = pd.read_csv(summary_path)\nmonthly_panel = pd.read_csv(monthly_panel_path, parse_dates=['month_date'])\nprint('Read summary:', summary_path, '->', len(summary), 'rows')\nprint('Read monthly panel:', monthly_panel_path, '->', len(monthly_panel), 'rows')\nsummary.head()"),
    new_markdown_cell('## Select 25 PASS features'),
    new_code_cell("pass_features = summary[summary['ic_pass'] == 'PASS']['predictor'].tolist()[:25]\nprint('Selected PASS features (count=%d):' % len(pass_features))\nprint(pass_features)") ,
    new_markdown_cell('## Prepare feature matrix'),
    new_code_cell("# Keep only columns present in the monthly panel\npass_features = [c for c in pass_features if c in monthly_panel.columns]\nprint('PASS features present in panel:', len(pass_features))\nfeature_matrix = monthly_panel[pass_features].apply(pd.to_numeric, errors='coerce')\nfeature_matrix.head()"),
    new_markdown_cell('## Pairwise Spearman correlation'),
    new_code_cell("pairwise_corr = feature_matrix.corr(method='spearman')\n# full correlation matrix\nprint('Full Spearman correlation matrix (rounded 6):')\nprint(pairwise_corr.round(6).to_string())\n\n# pairs where |corr| > 0.03\nthreshold = 0.03\npairs = []\ncols = pairwise_corr.columns.tolist()\nfor i,a in enumerate(cols):\n    for b in cols[i+1:]:\n        rho = pairwise_corr.loc[a,b]\n        if pd.notna(rho) and abs(rho) > threshold:\n            pairs.append((a,b,float(rho),abs(float(rho))))\n\npairs_sorted = sorted(pairs, key=lambda x: x[3], reverse=True)\nprint('\nPairs with |rho| > {:.3f}: {}\n'.format(threshold, len(pairs_sorted)))\nfor a,b,rho,ab in pairs_sorted:\n    print(f"{a} | {b} | rho={rho:.6f} | abs={ab:.6f}")\n\n# ranked list (abs, a, b, rho)\nprint('\nRanked list of correlated pairs (abs,rho,a,b):')\nfor a,b,rho,ab in pairs_sorted:\n    print(f"{ab:.6f}\t{a}\t{b}\t{rho:.6f}")"),
    new_markdown_cell('## Export results\n\nSave the full matrix and pair list to CSVs.'),
    new_code_cell("pairwise_corr.to_csv('pass_features_pairwise_spearman.csv')\nimport csv\nwith open('pass_feature_pairs_gt_003.csv','w',newline='') as fh:\n    writer = csv.writer(fh)\n    writer.writerow(['feature_a','feature_b','rho','abs_rho'])\n    for a,b,rho,ab in pairs_sorted:\n        writer.writerow([a,b,rho,ab])\nprint('Wrote pass_features_pairwise_spearman.csv and pass_feature_pairs_gt_003.csv')")
]
nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
nb.metadata['language_info'] = {'name':'python','version':'3.11'}

path = r'd:\PycharmProjects\mf-ml\09_IC_ALL_features.ipynb'
with open(path,'w',encoding='utf-8') as f:
    nbformat.write(nb,f)
print('Wrote notebook:',path)
