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

def hz_to_semitone(hz, ref): return math.log10(hz / ref) * 39.86

warnings.simplefilter("ignore")

# %%
word_frequencies = {
    "man": {
        "tone55": {"tone21": 10/11245330, "tone51": 20/11245330}, # Mandarin word frequencies from AS Balanced Corpus: https://asbc.iis.sinica.edu.tw
        "tone35": {"tone21": 35/11245330, "tone51": 257/11245330},
        "tone51": {"tone21": 45/11245330, "tone51": 2/11245330},
    },
    "min": {
        "tone55": {"tone21": 1e-10, "tone51": 3/711486}, # Southern Min word frequencies from the TAT_MOE Corpus: https://tggl.naer.edu.tw/corpora_applications/new
        "tone51": {"tone21": 1e-10, "tone51": 15/711486}, # 0 frequencty words are set to 1e-10 to avoid log(0) errors
        "tone33": {"tone21": 1e-10, "tone51": 1e-10},
        "tone21": {"tone21": 10/711486, "tone51": 20/711486},
    }
}

# %% [markdown]
# ## PsychoPy data processing

# %%
data = {"preceding_tone": [], "tone21_word_freq": [], "tone51_word_freq": [], "preceding_tone_offset": [], "level": [], "falling_tone_chosen": [], "participant": [], "lang": [], "strict": [], "sex": [], "fluency": []}

for lang in ("man", "min"):
    if lang=="man":
        participants = participant_info_df[participant_info_df.group=="monolingual"].code.to_list() + \
                    participant_info_df[participant_info_df.group=="bilingual"].code.to_list()
    if lang=="min":
        participants = participant_info_df[participant_info_df.group=="bilingual"].code.to_list()

    pbar = tqdm.tqdm(participants)
    for participant in pbar: # iterate through all participants

        df = pd.read_csv(f'raw_data/{participant}/{["Mandarin", "Min"][lang=="min"]}/{participant}_{["Mandarin", "Min"][lang=="min"]}.csv')

        tones_chosen = [
            clicked_name == "word_2_text_test1"
            for clicked_name in df["trial_mouse.clicked_name"]
            if type(clicked_name) == str
        ]
        preceding_tones = [
            file.split("WAV/")[-1].split("_")[0]
            for file in list(
                df[
                    [
                        type(clicked_name) == str
                        for clicked_name in df["trial_mouse.clicked_name"]
                    ]
                ]["WAV_file"]
            )
        ]
        levels = [
            int(file.split("WAV/")[-1].split("_")[1].replace("level", "").replace(".wav", ""))
            for file in list(
                df[
                    [
                        type(clicked_name) == str
                        for clicked_name in df["trial_mouse.clicked_name"]
                    ]
                ]["WAV_file"]
            )
        ]

        for preceding_tone in set(preceding_tones):
            for level in range(1, 11):
                choices = []
                for this_tone_chosen, this_preceding_tone, this_level in zip(
                    tones_chosen, preceding_tones, levels
                ):
                    if this_preceding_tone == preceding_tone and this_level == level:
                        choices.append(this_tone_chosen)

                for choice in choices:

                    data["preceding_tone"].append(preceding_tone)
                    data["tone21_word_freq"].append(
                        word_frequencies[lang][preceding_tone]["tone21"]
                    )
                    data["tone51_word_freq"].append(
                        word_frequencies[lang][preceding_tone]["tone51"]
                    )
                    data["preceding_tone_offset"].append(int(preceding_tone[-1]))
                    data["level"].append(level)
                    data["falling_tone_chosen"].append([0, 1][choice])
                    data["participant"].append(participant)
                    data["lang"].append(lang+"_bilingual" if lang == "man" and participant_info_df.group[list(participant_info_df.code).index(participant)] == "bilingual" else lang)
                    data["strict"].append(
                        (
                            participant_info_df.group[list(participant_info_df.code).index(participant)]
                            == "bilingual"
                            and lang == "min"
                        )
                        or (
                            participant_info_df.group[list(participant_info_df.code).index(participant)]
                            == "monolingual"
                            and lang == "man"
                            )
                            )
                    data["sex"].append(participant_info_df.sex[list(participant_info_df.code).index(participant)])
                    data["fluency"].append(participant_info_df.fluency[list(participant_info_df.code).index(participant)])

# %%
df = pd.DataFrame(data)

df["word_freq_ratio"] = np.log(df["tone51_word_freq"] / df["tone21_word_freq"])

df.to_csv(f"concatenated_data/experiment2_results.csv", index=False)


