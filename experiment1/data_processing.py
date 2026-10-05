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
for lang in ("man", "min"):
    if lang=="man":
        participants = participant_info_df[participant_info_df.group=="monolingual"].code.to_list() + \
                    participant_info_df[participant_info_df.group=="bilingual"][participant_info_df.fluency>=8].code.to_list()
        participants.remove("P20") # P20 did not participate in Exp. 1
    if lang=="min":
        participants = participant_info_df[participant_info_df.group=="bilingual"][participant_info_df.fluency>=8].code.to_list()

    pbar = tqdm.tqdm(participants)
    for participant in pbar: # iterate through all participants

        if participant in ("P1", "P7", "P14", "P28", "P35", "P37", "P42"): # these participants speak dialects where T5 -> T7 under tone sandhi
            sandhi_rules = {
                "min": {
                    "1": "7",
                    "2": "1",
                    "3": "2",
                    "4": None,
                    "5": "7",
                    "7": "3",
                    "8": None,
                },
                "man": {"1": "1", "2": "2", "3": "2", "4": "4"},
            }
        else: # other participants speak dialects where T5 -> T3 under tone sandhi
            sandhi_rules = {
                "min": {"1": "7", "2": "1", "3": "2", "4": None, "5": "3", "7": "3", "8": None},
                "man": {"1": "1", "2": "2", "3": "2", "4": "4"},
            }


        pbar.set_description(f"Processing participant {participant}")

        pitch_range = pd.read_csv( # read the pitch range for the participant
            f'raw_data/{participant}/{["Mandarin", "Min"][lang=="min"]}/{participant}_pitch.csv'
            ).to_dict()
        
        fs = [f
            for f in glob.glob(
                f'raw_data/{participant}/{["Mandarin", "Min"][lang=="min"]}/*.wav'
            )
            if not (lang == "min" and ("T4" in f or "T8" in f))
        ]

        data = {
            "position": [],
            "tone": [],
            "word": [],
            "token": [],
            "point": [],
            "f0": [],
            "time": [],
        }

        if lang == "man": tones = "1234"
        if lang == "min": tones = "12357"

        token_counts = {}

        for f in fs:
            cur_dir = f'raw_data/{participant}/{["Mandarin", "Min"][lang=="min"]}'
            sound = plm.Sound(f)

            for tone in tones:

                tgrd = tgt.read_textgrid(f.replace(".wav", "_position.TextGrid"), include_empty_intervals=True)
                tgrd_word = tgt.read_textgrid(f.replace(".wav", "_word.TextGrid"), include_empty_intervals=True)
                pitch = sound.to_pitch(
                    pitch_floor=pitch_range[tone][0],
                    pitch_ceiling=pitch_range[tone][1],
                )

                f0 = pitch.selected_array["frequency"]
                f0[f0 == 0] = np.nan

                for interval_idx, interval in enumerate(tgrd.tiers[0]):
                    assert len(tgrd.tiers[0])== len(tgrd_word.tiers[0])
                    position = interval.text
                    if not ("preceding" in position or "following" in position or "isolated" in position):
                        continue

                    if position not in token_counts:
                        token_counts[position] = 0

                    if "preceding" in position:

                        this_tone = position[0]
                        context_tone = position[2]
                        sandhi_tone = sandhi_rules[lang][this_tone]

                    elif "following" in position:

                        sandhi_tone = this_tone = position[2]
                        context_tone = position[0]

                    elif "isolated" in position: # isolated stimuli were not included in this paper

                        sandhi_tone = this_tone = position[0]
                        context_tone = "0"

                    if sandhi_tone == tone and (
                        lang == "man"
                        or (
                            lang == "min"
                            and tone not in "48"
                            and context_tone not in "48"
                        )
                    ):
                        start_time, end_time = interval.start_time, interval.end_time
                        start_frame, end_frame = round(pitch.time_to_frame_number(start_time)), round(pitch.time_to_frame_number(end_time))
                        f0s = f0[start_frame:end_frame + 1]
                        frames = [i for i in range(start_frame, end_frame + 1)]

                        idx_step = len(f0s) / 11 # calculate the step size for dividing the pitch contour into 11 equal portions
                        time_step = (end_time - start_time) / 11

                        mean_f0s = []
                        times = []

                        for idx in range(11):
                            this_start_time = start_time + time_step * idx
                            this_end_time = start_time + time_step * (idx + 1)
                            this_time = (this_start_time + this_end_time) / 2
                            times.append(this_time)

                            these_f0s = np.array([
                                this_f0 for this_f0 in f0s[round(idx_step * idx):round(idx_step * (idx + 1))]
                                if not np.isnan(this_f0)
                            ])
                            if len(these_f0s) == 0:
                                mean_f0 = np.nan
                            else:
                                mean_f0 = np.array([
                                    this_f0 for this_f0 in these_f0s if not np.isnan(this_f0)
                                ]).mean()
                            mean_f0s.append(mean_f0) # calculate the mean pitch for each of the 11 equal portions

                        for point, (this_f0, this_time) in enumerate(zip(mean_f0s, times)):
                            if lang=="man" and position.startswith("3-3"): continue

                            data["position"].append(position)
                            data["tone"].append(this_tone)
                            data["word"].append(tgrd_word.tiers[0][interval_idx].text)
                            data["token"].append(token_counts[position])
                            data["point"].append(point)
                            data["f0"].append(this_f0)
                            data["time"].append(this_time)

                        token_counts[position] += 1
                        pbar.set_description(f"Processing participant {participant}, position {position}, tone {this_tone}")


        f0_df = pd.DataFrame(data)

        ref = f0_df.f0.quantile(0.01)
        f0_df["semitone"] = [hz_to_semitone(f0, ref) for f0 in f0_df.f0]
        f0_df["semitone_z"] = (
            f0_df.semitone - f0_df.semitone.mean()
        ) / f0_df.semitone.std(ddof=0)
        f0_df["f0_z"] = (
            f0_df.f0 - f0_df.f0.mean()
        ) / f0_df.f0.std(ddof=0)
        f0_df.to_csv(
            f'raw_data/{participant}/{["Mandarin", "Min"][lang=="min"]}/{participant}_F0.csv'
        )


# %%
value_type = "f0_z"

n_discarded = 0
total_n = 0
n_taken = {
    "preceding": {0: 0, 1: 0, 2: 0},
    "following": {0: 0, 1: 0, 2: 0},
}

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
        data = {"pair": [], "token": [], "offset": [], "onset": []}
        for pair in set(
            [position[:3] for position in f0_df.position if "isolated" not in position]
        ):
            for token in range(10):
                total_n+=1
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

                preceding_offsets = [(idx, v) for idx, v in enumerate(preceding_tone_values[-3:][::-1]) if not np.isnan(v)]
                following_onsets = [(idx, v) for idx, v in enumerate(following_tone_values[:3]) if not np.isnan(v)]
                
                if len(preceding_offsets)!=0 and len(following_onsets)!=0:
                    preceding_offset_idx, preceding_offset = preceding_offsets[0]
                    following_onset_idx, following_onset = following_onsets[0]

                    n_taken["preceding"][preceding_offset_idx] += 1
                    n_taken["following"][following_onset_idx] += 1

                    data["pair"].append(pair)
                    data["token"].append(token)
                    data["offset"].append(preceding_offset)
                    data["onset"].append(following_onset)

                else:

                    n_discarded+=1

        onset_offset_df = pd.DataFrame(data)
        onset_offset_df["participant"] = [participant]*len(onset_offset_df)
        onset_offset_df["lang"] = [lang]*len(onset_offset_df)
        onset_offset_df.to_csv(
            f'processed_data/{participant}_{lang}_{value_type}_onset_offset.csv'
        )

print(f"Total number of tokens: {total_n}")
print(f"Number of tokens discarded: {n_discarded} ({n_discarded/total_n*100:.2f}%)")
print(f"Number of tokens taken:")
for position in n_taken:
    print(position)
    for idx in n_taken[position]:
        print(f"{idx}: {n_taken[position][idx]} ({n_taken[position][idx]/(total_n-n_discarded)*100:.2f}%)")

# %% [markdown]
# ### Combine to the final onset-offset data

# %%
dfs = [pd.read_csv(f, index_col=0) for f in glob.glob(f"processed_data/*_*_{value_type}_onset_offset.csv")]

pd.concat(dfs).to_csv(f"concatenated_data/{value_type}_onset_offset.csv")

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
df = pd.read_csv(f"concatenated_data/{value_type}_onset_offset.csv", index_col=0)

# %%
onset_offset_data = {
    "position": [],
    "pair": [],
    "tone": [],
    "token": [],
    "onset": [],
    "offset": [],
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

            for token, offset, onset, participant, pair in zip(
                this_df.token,
                this_df.offset,
                this_df.onset,
                this_df.participant,
                this_df.pair,
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

                onset_offset_data["position"].append(position)
                onset_offset_data["pair"].append(pair)
                onset_offset_data["tone"].append(sandhi_tone)
                onset_offset_data["token"].append(token)
                onset_offset_data["onset"].append(onset)
                onset_offset_data["offset"].append(offset)
                onset_offset_data["participant"].append(participant)
                onset_offset_data["onset_segment"].append(onset_segment)
                onset_offset_data["offset_segment"].append(offset_segment)
                onset_offset_data["lang"].append(lang)
                onset_offset_data["strict"].append(
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
                onset_offset_data["sex"].append(participant_info_df.sex[list(participant_info_df.code).index(participant)])

# %%
onset_offset_df = pd.DataFrame(onset_offset_data)
xs = []
ys = []
for position, onset, offset in zip(
    onset_offset_df.position, onset_offset_df.onset, onset_offset_df.offset
):
    if position == "carryover":
        x, y = offset, onset

    elif position == "anticipatory":
        x, y = onset, offset

    xs.append(x)
    ys.append(y)

onset_offset_df["x"] = xs
onset_offset_df["y"] = ys

onset_offset_df["lang"] = [
    lang if strict else lang + "_bilingual"
    for lang, strict in zip(onset_offset_df.lang, onset_offset_df.strict)
]

onset_offset_df.to_csv(
    f"concatenated_data/{value_type}_onset_offset_processed.csv"
)


