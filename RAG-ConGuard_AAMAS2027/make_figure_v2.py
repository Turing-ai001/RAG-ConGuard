"""Rebuild the descriptive plot for the copy; does not rerun model experiments.

Identical data to make_figure.py -- only the point labels change, from internal
codes to the reporting names used in the rewritten manuscript. Writes a separate
file so the original figures/tradeoff.pdf (used by main.tex) is left untouched.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'pdf.fonttype':42,'ps.fonttype':42})
fig, ax = plt.subplots(figsize=(3.35, 2.3), layout='constrained')
# Counts from the dated revised-evaluation aggregate; each denominator is 60.
points = [('No-filter',1,50,'#666666','o',(5,-7)),
          ('Per-doc-LOO',4,39,'#777777','s',(5,3)),
          ('ConGuard+Floor',6,29,'#0072B2','D',(6,3)),
          ('ConGuard+Cut',27,11,'#D55E00','^',(-62,9)),
          ('TrustRAG-adapt',29,9,'#333333','s',(-62,-11))]
for label,refuse,attack,color,marker,offset in points:
    x,y=refuse/60*100,attack/60*100
    ax.scatter(x,y,c=color,marker=marker,s=32,zorder=3)
    ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=8)
ax.set(xlim=(-2,55),ylim=(0,95),xlabel='Refusal rate (%)',ylabel='Attack success rate (%)')
ax.set_xticks([0,10,20,30,40,50]); ax.set_yticks([0,20,40,60,80])
ax.spines[['top','right']].set_visible(False)
ax.grid(alpha=.18,zorder=0)
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/tradeoff_v2.pdf',metadata={'Title':'Observed safety-availability operating points','Author':''})
plt.close(fig)
print('wrote', ROOT/'figures/tradeoff_v2.pdf')
