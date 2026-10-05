# %% [markdown]
# ## Participant Info

# %%
import pandas as pd

participant_info_df = pd.read_csv("../participant_info.csv")

# %%
participant_info_df.head(5)

# %%
import tgt
import pytiers
import pandas as pd
import parselmouth as plm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import warnings
import tqdm
import math
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def hz_to_semitone(hz, ref): return math.log10(hz / ref) * 39.86

warnings.simplefilter("ignore")

# %% [markdown]
# ## PsychoPy data processing

# %% [markdown]
# ### Get slopes

# %%
def get_maxslope(x, y, absolute=True, return_xy=False):
    slopes = []
    xs = []
    ys = []
    real_slopes = []

    for x1, y1, x2, y2 in zip(x[:-1], y[:-1], x[1:], y[1:]):
        slope = (y2 - y1) / (x2 - x1)
        real_slopes.append(slope)
        slope = abs(slope)
        xs.append((x2 + x1) / 2)
        ys.append((y2 + y1) / 2)
        slopes.append(slope)
    if absolute:
        max_slope = max(slopes)
    else:
        if len([i for i in real_slopes if i >= 0]) == 0:
            max_slope = -max(slopes)
        else:
            max_slope = max(slopes)
    if return_xy:
        return max_slope, (xs[slopes.index(max(slopes))], ys[slopes.index(max(slopes))])
    return max_slope

# %%
data = {
    "tone": [],
    "lang": [],
    "level": [],
    "acceptance": [],
    "participant": [],
    "bilingual": [],
    "strict": [],
    "advanced": []
}

# %%
for lang in ("man", "min"):
    if lang=="man":
        participants = participant_info_df[participant_info_df.group=="monolingual"].code.to_list() + \
                    participant_info_df[participant_info_df.group=="bilingual"].code.to_list()
    if lang=="min":
        participants = participant_info_df[participant_info_df.group=="bilingual"].code.to_list()

    pbar = tqdm.tqdm(participants)
    for participant in pbar: # iterate through all participants

        if lang == "man":
            stimuli = {
                "tone21": ["桌腳", "喝酒", "吸管", "發展", "今晚"],
                "tone51": ["壓力", "飛彈", "登記", "功課", "衣物"],
            }
        elif lang == "min":
            stimuli = {
                "tone21": ["狡怪", "買票", "早暗", "短褲", "你看"],
                "tone51": ["水果", "米酒", "飽滿", "滾水", "補繳"],
            }


        df = pd.read_csv(f'raw_data/{participant}/{["Mandarin", "Min"][lang=="min"]}/{participant}_{["Mandarin", "Min"][lang=="min"]}.csv')

        acceptances = [
            clicked_name == "yes_text_trial_test2"
            for clicked_name in df["mouse_test2.clicked_name"]
            if type(clicked_name) == str
        ]
        words = [
            file.split("/")[-1].split("_")[0]
            for file in list(
                df[
                    [
                        type(clicked_name) == str
                        for clicked_name in df["mouse_test2.clicked_name"]
                    ]
                ]["WAV_file"]
            )
        ]

        levels = [
            int(file.split("level")[-1].replace(".wav", ""))
            for file in list(
                df[
                    [
                        type(clicked_name) == str
                        for clicked_name in df["mouse_test2.clicked_name"]
                    ]
                ]["WAV_file"]
            )
        ]
        
        tones = [
            "tone21" if file.split("WAV/")[-1][:2] in stimuli["tone21"] else "tone51"
            for file in list(
                df[
                    [
                        type(clicked_name) == str
                        for clicked_name in df["mouse_test2.clicked_name"]
                    ]
                ]["WAV_file"]
            )
        ]

        df_result = pd.DataFrame({"acceptance": acceptances, "tone": tones, "level": levels})

        data["tone"]+=tones
        data["level"]+=levels
        data["acceptance"]+=acceptances
        data["participant"]+=[participant]*len(acceptances)
        data["lang"]+=[lang]*len(acceptances)
        data["bilingual"]+=[participant_info_df.group[list(participant_info_df.code).index(participant)]=="bilingual"]*len(acceptances)
        strict = (participant_info_df.group[list(participant_info_df.code).index(participant)] == "bilingual" and lang == "min") or (participant_info_df.group[list(participant_info_df.code).index(participant)]== "monolingual"and lang == "man")
        data["strict"]+=[strict]*len(acceptances)
        data["advanced"]+=[participant_info_df.fluency[list(participant_info_df.code).index(participant)]>=8]*len(acceptances)

df = pd.DataFrame(data)

# %%
acceptance_slope_data = {
    "tone": [],
    "lang": [],
    "acceptance_slope": [],
    "participant": [],
    "bilingual": [],
    "strict": [],
    "advanced": []
}

# %%
for lang in ("man", "min"):
    if lang=="man":
        participants = participant_info_df[participant_info_df.group=="monolingual"].code.to_list() + \
                    participant_info_df[participant_info_df.group=="bilingual"].code.to_list()
    if lang=="min":
        participants = participant_info_df[participant_info_df.group=="bilingual"].code.to_list()

    pbar = tqdm.tqdm(participants)
    for participant in pbar: # iterate through all participants

        df_pred = pd.DataFrame({"X": [], "pred": [], "y": [], "tone": []})

        for tone in ("tone21", "tone51"):

            temp = df[(df.tone==tone) & (df.lang==lang) & (df.participant==participant)]

            X = list(temp["level"])
            X = sm.add_constant(X)
            X = np.array(X)
            y = np.array(list(temp["acceptance"]))

            model = LogisticRegression(solver="liblinear", random_state=0, C=100)

            model.fit(X, y)

            this_df_pred = pd.DataFrame(
                {
                    "X": X[:, 1],
                    "pred": model.predict_proba(X)[:, 1],
                    "y": y,
                    "tone": [tone for _ in range(len(X[:, 1]))],
                }
            )
            df_pred = pd.concat([df_pred, this_df_pred])

        df_pred = df_pred.reset_index(drop=True)
        df_pred = df_pred.sort_values(by=["tone"])


        for tone in ("tone51", "tone21"):
            ax = sns.lineplot(x="X", y="pred", data=df_pred[df_pred.tone == tone])

            line = ax.get_lines()[0]

            slope, (x, y) = get_maxslope(
                line.get_xdata(), line.get_ydata(), return_xy=True, absolute=True
            )
            plt.close("all")

            acceptance_slope_data["tone"] += [tone]
            acceptance_slope_data["lang"] += [lang]
            acceptance_slope_data["acceptance_slope"] += [slope]
            acceptance_slope_data["participant"] += [participant]
            acceptance_slope_data["bilingual"] += [
                participant_info_df.group[list(participant_info_df.code).index(participant)] == "bilingual"
            ]
            strict = (participant_info_df.group[list(participant_info_df.code).index(participant)] == "bilingual" and lang == "min") or (participant_info_df.group[list(participant_info_df.code).index(participant)]== "monolingual"and lang == "man")
            acceptance_slope_data["strict"] += [strict]
            acceptance_slope_data["advanced"] += [
                participant_info_df.fluency[list(participant_info_df.code).index(participant)] >= 8
            ]

df_acceptance_slope = pd.DataFrame(acceptance_slope_data)
df_acceptance_slope.to_csv("concatenated_data/experiment3_acceptance_slope.csv", index=False)

# %% [markdown]
# ### Plot an example

# %%
sns.set_style({"font.family": "serif", "font.serif": ["Times New Roman"]})
palette = sns.color_palette("muted")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, )

example_df = df[(df.tone=="tone51") & (df.lang=="man") & (df.participant=="P3")]
X = list(example_df["level"])
X = sm.add_constant(X)
X = np.array(X)
y = np.array(list(example_df["acceptance"]))
model = LogisticRegression(solver="liblinear", random_state=0, C=100)
model.fit(X, y)
pred_df = pd.DataFrame(
    {
        "level": X[:, 1],
        "pred": model.predict_proba(X)[:, 1],
        "y": y,
        "tone": [tone for _ in range(len(X[:, 1]))],
    }
)

palette = sns.color_palette("muted")
line = sns.lineplot(
    ax=axes[0],
    x="level",
    y="pred",
    data=pred_df,
    alpha=.5,
    lw=2,
    color="black"
)
line = line.get_lines()[0]
slope, (x, y) = get_maxslope(
    line.get_xdata(), line.get_ydata(), return_xy=True, absolute=True
)

# Bootstrap CI function
def bootstrap_ci(data, n_boot=1000, ci=95):
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

# Compute manually
rows = []
for level, group in example_df.groupby("level"):
    mean = group["acceptance"].mean()
    lower, upper = bootstrap_ci(group["acceptance"])
    rows.append({"level": level, "mean": mean, "lower": lower, "upper": upper})

summary = pd.DataFrame(rows)

for i, row in summary.iterrows():
    axes[0].hlines(y=row["mean"], xmin=row["level"] - 0.11, xmax=row["level"] + 0.11, color=palette[0], alpha=.9)
    axes[0].hlines(y=row["lower"], xmin=row["level"] - 0.21, xmax=row["level"] + 0.21, color=palette[0], alpha=.7)
    axes[0].vlines(x=row["level"], ymin=row["lower"], ymax=row["upper"], color="gray", linestyle="dotted", alpha=.7)
    axes[0].hlines(y=row["upper"], xmin=row["level"] - 0.21, xmax=row["level"] + 0.21, color=palette[0], alpha=.7)

axes[0].text(x+.2, y, f"slope: {round(slope, 3)}", verticalalignment="center")
axes[0].scatter(x=x, y=y, s=10, color=palette[1], zorder=1000)

example_df = df[(df.tone=="tone51") & (df.lang=="man") & (df.participant=="P15")]
X = list(example_df["level"])
X = sm.add_constant(X)
X = np.array(X)
y = np.array(list(example_df["acceptance"]))
model = LogisticRegression(solver="liblinear", random_state=0, C=100)
model.fit(X, y)
pred_df = pd.DataFrame(
    {
        "level": X[:, 1],
        "pred": model.predict_proba(X)[:, 1],
        "y": y,
        "tone": [tone for _ in range(len(X[:, 1]))],
    }
)

palette = sns.color_palette("muted")
line = sns.lineplot(
    ax=axes[1],
    x="level",
    y="pred",
    data=pred_df,
    alpha=.5,
    lw=2,
    color="black"
)
line = line.get_lines()[0]
slope, (x, y) = get_maxslope(
    line.get_xdata(), line.get_ydata(), return_xy=True, absolute=True
)
# Compute manually
rows = []
for level, group in example_df.groupby("level"):
    mean = group["acceptance"].mean()
    lower, upper = bootstrap_ci(group["acceptance"])
    rows.append({"level": level, "mean": mean, "lower": lower, "upper": upper})

summary = pd.DataFrame(rows)

for i, row in summary.iterrows():
    axes[1].hlines(y=row["mean"], xmin=row["level"] - 0.11, xmax=row["level"] + 0.11, color=palette[0], alpha=.7)
    axes[1].hlines(y=row["lower"], xmin=row["level"] - 0.21, xmax=row["level"] + 0.21, color=palette[0], alpha=.7)
    axes[1].vlines(x=row["level"], ymin=row["lower"], ymax=row["upper"], color="gray", linestyle="dotted", alpha=.7)
    axes[1].hlines(y=row["upper"], xmin=row["level"] - 0.21, xmax=row["level"] + 0.21, color=palette[0], alpha=.7)
axes[1].text(x+.2, y, f"slope: {round(slope, 3)}", verticalalignment="center")
axes[1].scatter(x=x, y=y, s=10, color=palette[1], zorder=1000)

fig.subplots_adjust(wspace=0.02, hspace=5)

axes[0].set_xlabel("Level")
axes[1].set_xlabel("Level")
axes[0].set_ylabel("Acceptance rate")
fig.savefig("visualization/slope_example.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### Get acceptance threshold through probit analysis

# %%
acceptance_threshold_data = {
    "tone": [],
    "lang": [],
    "acceptance_threshold": [],
    "participant": [],
    "bilingual": [],
    "strict": [],
    "advanced": []
}

# %%
def get_acceptance_threshold(df, tone, participant, lang, threshold=0.5):

    df_filtered = df[(df["tone"] == tone) & (df["participant"] == participant) & (df["lang"] == lang)]
    x = df_filtered["level"].values
    y = df_filtered["acceptance"].values
    try:
        model = sm.Probit(y, sm.add_constant(x)).fit()
        probs = model.predict(sm.add_constant(x))
        decision_boundary_index = np.argmin(np.abs(probs - threshold))
        decision_boundary = x[decision_boundary_index]
    except:
        decision_boundary = np.array(x)[np.array(y) == 0].max() + 0.5 # perfect fit scenario
    
    return decision_boundary

# %%
for lang in ("man", "min"):
    if lang=="man":
        participants = participant_info_df[participant_info_df.group=="monolingual"].code.to_list() + \
                    participant_info_df[participant_info_df.group=="bilingual"].code.to_list()
    if lang=="min":
        participants = participant_info_df[participant_info_df.group=="bilingual"].code.to_list()

    pbar = tqdm.tqdm(participants)
    for participant in pbar: # iterate through all participants

        for tone in ("tone21", "tone51"):
            acceptance_threshold = get_acceptance_threshold(df, tone, participant, lang, threshold=0.5)

            acceptance_threshold_data["tone"].append(tone)
            acceptance_threshold_data["lang"].append(lang)
            acceptance_threshold_data["acceptance_threshold"].append(acceptance_threshold)
            acceptance_threshold_data["participant"].append(participant)
            acceptance_threshold_data["bilingual"].append(participant_info_df.group[list(participant_info_df.code).index(participant)]=="bilingual")
            strict = (participant_info_df.group[list(participant_info_df.code).index(participant)] == "bilingual" and lang == "min") or (participant_info_df.group[list(participant_info_df.code).index(participant)]== "monolingual"and lang == "man")
            acceptance_threshold_data["strict"].append(strict)
            acceptance_threshold_data["advanced"].append(participant_info_df.fluency[list(participant_info_df.code).index(participant)]>=8)
df_acceptance_threshold = pd.DataFrame(acceptance_threshold_data)

# %%
df_acceptance_threshold.to_csv("concatenated_data/experiment3_acceptance_threshold.csv", index=False)

# %%



