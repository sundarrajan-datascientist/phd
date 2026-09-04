import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.cells = []

nb.cells.append(new_markdown_cell('# Monthly IC analysis for mutual fund predictors\n\nThis notebook reads the IC summary and monthly panel from CSVs, selects the top 25 PASS features, computes their pairwise Spearman correlations, and reports the results.'))
nb.cells.append(new_code_cell(['import os', 'import numpy as np', 'import pandas as pd', 'from scipy.stats import spearmanr']))
nb.cells.append(new_markdown_cell('## Read IC summary and monthly panel\n\nEnsure the summary (ic_summary_all_features.csv) and monthly panel (ic_monthly_all_features.csv) are present.'))
nb.cells.append(new_code_cell(["# paths", "summary_path = 'ic_summary_all_features.csv'", "monthly_panel_path = 'ic_monthly_all_features.csv'", "", "# read files", "summary = pd.read_csv(summary_path)", "monthly_panel = pd.read_csv(monthly_panel_path, parse_dates=['month_date'])", "print('Read summary:', summary_path, '->', len(summary), 'rows')", "print('Read monthly panel:', monthly_panel_path, '->', len(monthly_panel), 'rows')", "summary.head()"]))
nb.cells.append(new_markdown_cell('## Select 25 PASS features'))
nb.cells.append(new_code_cell(["pass_features = summary[summary['ic_pass'] == 'PASS']['predictor'].tolist()[:25]", "print('Selected PASS features (count=%d):' % len(pass_features))", "print(pass_features)"]))
nb.cells.append(new_markdown_cell('## Prepare feature matrix'))
nb.cells.append(new_code_cell(["# Keep only columns present in the monthly panel", "pass_features = [c for c in pass_features if c in monthly_panel.columns]", "print('PASS features present in panel:', len(pass_features))", "feature_matrix = monthly_panel[pass_features].apply(pd.to_numeric, errors='coerce')", "feature_matrix.head()"]))
nb.cells.append(new_markdown_cell('## Pairwise Spearman correlation'))
cell_lines = []
cell_lines.append('pairwise_corr = feature_matrix.corr(method="spearman")')
cell_lines.append('# full correlation matrix')
cell_lines.append('print("Full Spearman correlation matrix (rounded 6):")')
cell_lines.append('print(pairwise_corr.round(6).to_string())')
cell_lines.append('')
cell_lines.append('# pairs where |corr| > 0.03')
cell_lines.append('threshold = 0.03')
cell_lines.append('pairs = []')
cell_lines.append('cols = pairwise_corr.columns.tolist()')
cell_lines.append('for i,a in enumerate(cols):')
cell_lines.append('    for b in cols[i+1:]:')
cell_lines.append('        rho = pairwise_corr.loc[a,b]')
cell_lines.append('        if pd.notna(rho) and abs(rho) > threshold:')
cell_lines.append('            pairs.append((a,b,float(rho),abs(float(rho))))')
cell_lines.append('')
cell_lines.append('pairs_sorted = sorted(pairs, key=lambda x: x[3], reverse=True)')
cell_lines.append('print("\nPairs with |rho| > {:.3f}: {}\n".format(threshold, len(pairs_sorted)))')
cell_lines.append('for a,b,rho,ab in pairs_sorted:')
cell_lines.append('    print(f"{a} | {b} | rho={rho:.6f} | abs={ab:.6f}")')
cell_lines.append('')
cell_lines.append('# ranked list (abs, a, b, rho)')
cell_lines.append('print("\nRanked list of correlated pairs (abs,rho,a,b):")')
cell_lines.append('for a,b,rho,ab in pairs_sorted:')
cell_lines.append('    print(f"{ab:.6f}\t{a}\t{b}\t{rho:.6f}")')
nb.cells.append(new_code_cell(cell_lines))
nb.cells.append(new_markdown_cell('## Export results\n\nSave the full matrix and pair list to CSVs.'))
nb.cells.append(new_code_cell(["pairwise_corr.to_csv('pass_features_pairwise_spearman.csv')", "import csv", "with open('pass_feature_pairs_gt_003.csv','w',newline='') as fh:", "    writer = csv.writer(fh)", "    writer.writerow(['feature_a','feature_b','rho','abs_rho'])", "    for a,b,rho,ab in pairs_sorted:", "        writer.writerow([a,b,rho,ab])", "print('Wrote pass_features_pairwise_spearman.csv and pass_feature_pairs_gt_003.csv')"]))

nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
nb.metadata['language_info'] = {'name':'python','version':'3.11'}

path = r'd:\PycharmProjects\mf-ml\09_IC_ALL_features.ipynb'
with open(path,'w',encoding='utf-8') as f:
    nbformat.write(nb,f)
print('Wrote notebook:',path)
