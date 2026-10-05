# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sns.set_style("whitegrid")
sns.set_style({"font.family": "serif", "font.serif": ["Times New Roman"]})
palette = sns.color_palette("muted")
colors = {
    "tone55": palette[0],
    "tone51": palette[2],
    "tone35": palette[1],
    "tone33": palette[3],
    "tone21": palette[4],
}

# Create custom legend handles (dot markers)
legend_elements = [
    Patch(facecolor=color, label=tone[-2:])
    for tone, color in colors.items()
]

# Then add the legend to your figure or axis
plt.legend(handles=legend_elements, title='', loc='upper right')

position = "carryover"

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)

for idx, lang in enumerate(("man", "man_bilingual", "min")):

    df = pd.read_csv(f"lmm_fit_{lang}.csv")
    for tone in df['tone'].unique():
        
        for participant in df['participant'].unique():
            temp = df[(df['tone'] == tone) & (df['participant'] == participant) & (df['position'] == position)]

            sns.scatterplot(ax=axes[idx], data=temp, x='x', y='y', color=colors["tone"+str(tone)], alpha=0.1, s=10)
            sns.lineplot(ax=axes[idx], data=temp, x='x', y='fitted', color=colors["tone"+str(tone)], lw=1)
            axes[idx].set_xlabel("")

            axes[idx].set_yticklabels([round(i, 1) for i in axes[idx].get_yticks()], size = 12)
            axes[idx].set_xticklabels([int(i) for i in axes[idx].get_xticks()], size = 12)

plt.legend(handles=legend_elements, title='Following tone', loc='upper right', bbox_to_anchor=(1.63, 1.05), frameon=False, fontsize=12, title_fontsize=12)
fig.subplots_adjust(wspace=0.05)
axes[0].set_ylabel("F0 onset of the following tone\n(z-scored)", fontsize=12)
fig.text(0.5, -0.01, 'F0 offset of the preceding tone (z-scored)', va='center', rotation='horizontal', ha='center', fontsize=12)
fig.tight_layout()
fig.savefig(f"lmm_fit_{position}.png", dpi=300, bbox_inches='tight', )

# %%
position = "anticipatory"

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)

for idx, lang in enumerate(("man", "man_bilingual", "min")):

    df = pd.read_csv(f"lmm_fit_{lang}.csv")
    for tone in df['tone'].unique():
        
        for participant in df['participant'].unique():
            temp = df[(df['tone'] == tone) & (df['participant'] == participant) & (df['position'] == position)]

            sns.scatterplot(ax=axes[idx], data=temp, x='x', y='y', color=colors["tone"+str(tone)], alpha=0.1, s=10)
            sns.lineplot(ax=axes[idx], data=temp, x='x', y='fitted', color=colors["tone"+str(tone)], lw=1)
            axes[idx].set_xlabel("")

            axes[idx].set_yticklabels([round(i, 1) for i in axes[idx].get_yticks()], size = 12)
            axes[idx].set_xticklabels([int(i) for i in axes[idx].get_xticks()], size = 12)

plt.legend(handles=legend_elements, title='Preceding tone', loc='upper right', bbox_to_anchor=(1.63, 1.05), frameon=False, fontsize=12, title_fontsize=12)
fig.subplots_adjust(wspace=0.05)
axes[0].set_ylabel("F0 offset of the preceding tone\n(z-scored)", fontsize=12)
fig.text(0.5, -0.01, 'F0 onset of the following tone (z-scored)', va='center', rotation='horizontal', ha='center', fontsize=12)
fig.tight_layout()
fig.savefig(f"lmm_fit_{position}.png", dpi=300, bbox_inches='tight', )


