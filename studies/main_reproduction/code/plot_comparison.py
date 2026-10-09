from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=Path('.tmp/main-reproduction-20261003');out=Path('studies/main_reproduction/docs/figures')
b=json.loads((s/'full-comparison.json').read_text(encoding='utf-8'))
colors={'linear':'#2563eb','mlp':'#ea580c','cnn':'#059669','resnet18':'#9333ea'}
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for column,data in enumerate(['MNIST','CIFAR10']):
    for row in b['rows']:
        if row['data']!=data:continue
        for index,metric in enumerate(['test/Accuracy','test/Loss']):
            ax=axes[index,column]
            for source,style,marker in [('original','--','o'),('current','-','x')]:
                ax.plot([r['step'] for r in row['steps']],[r[source][metric] for r in row['steps']],linestyle=style,marker=marker,color=colors[row['model']],label=f"{row['model']} / {source}")
            ax.set(title=f'{data} / {metric.split("/")[1]}',xlabel='optimizer step',ylabel='Accuracy (%)' if index==0 else 'Cross-entropy loss',xticks=[30,60])
            ax.grid(alpha=.25)
    axes[1,column].legend(fontsize=8,ncol=2)
fig.suptitle('Pinned main base: original vs RPipe | seed 0 | benchmark enabled | numeric gate 4/8')
fig.savefig(out/'original_comparison.png',dpi=160)
plt.close(fig)
