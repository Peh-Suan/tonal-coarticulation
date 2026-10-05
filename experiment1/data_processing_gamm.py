# %% [markdown]
# ## Participant Info

# %%
import pandas as pd

participant_info_df = pd.read_csv("../participant_info.csv")

# %%
import tgt # For preprocessing TextGrid files
import parselmouth as plm # Praat implementation in Python
import numpy as np
import glob
import warnings
import tqdm
import math

def hz_to_semitone(hz, ref): return math.log10(hz / ref) * 39.86 # semitone conversion was also considered, but was not used eventually. In previous studies, both raw F0 and semitones were used by different studies. We followed the majority of the studies, which used raw F0.

warnings.simplefilter("ignore")

# %% [markdown]
# ## F0 Processing

# %%
value_type = "f0_z"


dfs = []
for lang in ("man", "min"):
    if lang=="man":
        participants = participant_info_df[participant_info_df.group=="monolingual"].code.to_list() + \
                    participant_info_df[participant_info_df.group=="bilingual"][participant_info_df.fluency>=8].code.to_list()
        participants.remove("P20") # P20 did not participate in Exp. 1
    if lang=="min":
        participants = participant_info_df[participant_info_df.group=="bilingual"][participant_info_df.fluency>=8].code.to_list()
    
    pbar = tqdm.tqdm(participants)
    for participant in pbar: # iterate through all participants

        pbar.set_description(f"Processing participant {participant}")

        f0_df = pd.read_csv(
            f'raw_data/{participant}/{["Mandarin", "Min"][lang=="min"]}/{participant}_F0.csv'
        )
        data = {"pair": [], "token": [], "mean_f0": [], "portion_idx": [], "pos": []}
        for pair in set(
            [position[:3] for position in f0_df.position if "isolated" not in position]
        ):
            if lang=="man" and pair=="3-3": continue
            for token in range(10):

                preceding_tone_values = list(
                    f0_df[f0_df.position == f"{pair}-preceding"][
                        f0_df.token == token
                        ][value_type]
                        )
                following_tone_values = list(
                    f0_df[f0_df.position == f"{pair}-following"][
                        f0_df.token == token
                        ][value_type]
                        )
                if preceding_tone_values==[] or following_tone_values==[]:
                    continue
                data["pair"]+=[pair]*len(preceding_tone_values)
                data["portion_idx"]+=list(range(11))
                data["token"]+=[token]*len(preceding_tone_values)
                data["mean_f0"]+=preceding_tone_values
                data["pos"]+=["preceding"]*len(preceding_tone_values)

                data["pair"]+=[pair]*len(following_tone_values)
                data["portion_idx"]+=list(range(11, 22))
                data["token"]+=[token]*len(following_tone_values)
                data["mean_f0"]+=following_tone_values
                data["pos"]+=["following"]*len(following_tone_values)
                

        df = pd.DataFrame(data)
        df["participant"] = [participant]*len(df)
        df["lang"] = [lang]*len(df)
        dfs+=[df]

# %%
df = pd.concat(dfs, ignore_index=True)

# %% [markdown]
# ### Add information for data analysis

# %%
tones = {
    "man": [(1, "55", "55"), (2, "35", "35"), (3, "21", "35"), (4, "51", "51")],
    "min": [
        (1, "55", "33"),
        (2, "51", "55"),
        (3, "21", "51"),
        (5, "35", "21"),
        (7, "33", "21"),
    ],
}

segments = {
    "min": {
        "1-1": ("V", "N"),
        "1-2": ("V", "S"),
        "1-3": ("V", "V"),
        "1-5": ("V", "N"),
        "1-7": ("N", "V"),
        "2-1": ("N", "N"),
        "2-2": ("V", "S"),
        "2-3": ("V", "V"),
        "2-5": ("V", "V"),
        "2-7": ("V", "L"),
        "3-1": ("N", "V"),
        "3-2": ("V", "N"),
        "3-3": ("V", "L"),
        "3-5": ("V", "L"),
        "3-7": ("N", "S"),
        "5-1": ("V", "L"),
        "5-2": ("V", "L"),
        "5-3": ("V", "V"),
        "5-5": ("V", "N"),
        "5-7": ("N", "S"),
        "7-1": ("V", "L"),
        "7-2": ("V", "S"),
        "7-3": ("N", "V"),
        "7-5": ("V", "L"),
        "7-7": ("V", "S"),
    },
    "man": {
        "1-1": ("V", "N"),
        "1-2": ("V", "S"),
        "1-3": ("V", "N"),
        "1-4": ("N", "V"),
        "2-1": ("N", "N"),
        "2-2": ("V", "S"),
        "2-3": ("V", "V"),
        "2-4": ("V", "V"),
        "3-1": ("N", "V"),
        "3-2": ("V", "N"),
        "3-4": ("V", "L"),
        "4-1": ("V", "L"),
        "4-2": ("V", "L"),
        "4-3": ("V", "V"),
        "4-4": ("V", "N"),
    },
}

# %%
f0_data = {
    "position": [],
    "pair": [],
    "pos": [],
    "tone": [],
    "token": [],
    "mean_f0": [],
    "portion_idx": [],
    "participant": [],
    "onset_segment": [],
    "offset_segment": [],
    "lang": [],
    "strict": [],
    "sex": []
}

for lang in ("min", "man"):

    for position in ("carryover", "anticipatory"):
        for tone in tones[lang]:
            this_tone = tone[0]

            if position == "carryover":
                this_df = df[(df.pair.str.contains(f"-{this_tone}")) & (df.lang == lang)]
                sandhi_tone = tone[1]

            elif position == "anticipatory":
                this_df = df[(df.pair.str.contains(f"{this_tone}-")) & (df.lang == lang)]

            for token, mean_f0, portion_idx, participant, pair, pos in zip(
                this_df.token,
                this_df.mean_f0,
                this_df.portion_idx,
                this_df.participant,
                this_df.pair,
                this_df.pos,
            ):
                if position == "anticipatory": # apply tone sandhi rules for anticipatory condition
                    if lang == "min":
                        sandhi_tone = tone[2]
                        if participant in (
                            "P1",
                            "P7",
                            "P14",
                            "P28",
                            "P35",
                            "P37",
                            "P42",
                        ):
                            if tone[0] == 5:
                                sandhi_tone = "33"

                    elif lang == "man":
                        if pair == "3-3":
                            sandhi_tone = tone[2]
                        else:
                            sandhi_tone = tone[1]

                onset_segment = segments[lang][pair][0]
                offset_segment = segments[lang][pair][1]

                f0_data["position"].append(position)
                f0_data["pair"].append(pair)
                f0_data["pos"].append(pos)
                f0_data["tone"].append(sandhi_tone)
                f0_data["token"].append(token)
                f0_data["mean_f0"].append(mean_f0)
                f0_data["portion_idx"].append(portion_idx)
                f0_data["participant"].append(participant)
                f0_data["onset_segment"].append(onset_segment)
                f0_data["offset_segment"].append(offset_segment)
                f0_data["lang"].append(lang)
                f0_data["strict"].append(
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
                f0_data["sex"].append(participant_info_df.sex[list(participant_info_df.code).index(participant)])

# %%
f0_df = pd.DataFrame(f0_data)

f0_df["lang"] = [
    lang if strict else lang + "_bilingual"
    for lang, strict in zip(f0_df.lang, f0_df.strict)
]

f0_df.to_csv(
    f"concatenated_data/{value_type}_mean_f0_processed_for_gamm.csv"
)


