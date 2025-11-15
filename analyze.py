import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 1) Load and keep feasible points
res_dir = Path("results")
files = sorted(res_dir.glob("pareto_full*.csv"))

csv_path = files[-1]

df = pd.read_csv(csv_path)
df_feas = df[df['constraint'] <= 0].copy()

# 2) Interactive scatter (energy vs photocurrent)
fig = px.scatter(
    df_feas, x='max_energy_eV', y='max_photocurrent',
    color='quality_factor', size='prominence',
    hover_data=['RW','RQWt','RQBt','MQWt','LW','LQWt','LQBt','constraint'],
    title='Energy vs Photocurrent (colored by Q, sized by prominence)',
)
fig.update_yaxes(type='log', range=[-14.5, -7.5])  # adjust as needed
fig.show()

# 3) Pairwise objective matrix (quick scan of trade-offs)
obj_cols = ['max_energy_eV','max_photocurrent','quality_factor','prominence']
sns.pairplot(df_feas[obj_cols])
plt.show()

# 4) Parallel coordinates across objectives (normalize for visualization)
norm = df_feas[obj_cols].copy()
for c in obj_cols:
    cmin, cmax = norm[c].min(), norm[c].max()
    norm[c] = (norm[c]-cmin)/(cmax-cmin + 1e-12)
norm['label'] = (
    'RW='+df_feas['RW'].astype(str)
    +', RQWt='+df_feas['RQWt'].round(3).astype(str)
)
fig2 = px.parallel_coordinates(norm, dimensions=obj_cols, color='quality_factor')
fig2.show()
