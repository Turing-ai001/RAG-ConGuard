"""Rebuild the descriptive plot; does not rerun model experiments."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'pdf.fonttype':42,'ps.fonttype':42})
fig, ax = plt.subplots(figsize=(3.35, 2.3), layout='constrained')
# Counts from the dated revised-evaluation aggregate; each denominator is 60.
points = [('B0',1,50,'#666666','o',(5,-7)),
          ('B6',4,39,'#777777','s',(5,3)),
          ('C–G1',6,29,'#0072B2','D',(6,3)),
          ('C–G0',27,11,'#D55E00','^',(-41,9)),
          ('B5 adaptation',29,9,'#333333','s',(-78,-11))]
for label,refuse,attack,color,marker,offset in points:
    x,y=refuse/60*100,attack/60*100
    ax.scatter(x,y,c=color,marker=marker,s=32,zorder=3)
    ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=8)
ax.set(xlim=(-2,55),ylim=(0,95),xlabel='Refusal rate (%)',ylabel='Attack success rate (%)')
ax.set_xticks([0,10,20,30,40,50]); ax.set_yticks([0,20,40,60,80])
ax.spines[['top','right']].set_visible(False)
ax.grid(alpha=.18,zorder=0)
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/tradeoff.pdf',metadata={'Title':'Observed safety–availability operating points','Author':''})
plt.close(fig)
