###
author: Po-Hsuan Huang
email: pohsuanh@sas.upenn.edu
###

This repository contains the data, experiment materials and analysis code for the study
"Production and perception of coarticulated tones: The cases of Taiwan Mandarin and
Taiwan Southern Min".

-------------------------------------------------------------------------------
WHICH SCRIPT PRODUCED WHICH RESULT IN THE PAPER
-------------------------------------------------------------------------------
Experiment 1 (production)
  Data extraction .......... experiment1/data_processing.ipynb / .py
                             -> concatenated_data/f0_z_onset_offset_processed.csv
                             Each token contributes the mean F0 of the single portion
                             closest to the syllable boundary; onset_portion_idx and
                             offset_portion_idx record which portion was usable
                             (0 = first/last portion, 1-2 = fallback after F0
                             tracking failure).
  Reported analysis ........ experiment1/data_analysis_filtered.ipynb
                             (Table 2, Table 3, Sections 2.2.1-2.2.3)
  Unfiltered version ....... experiment1/data_analysis.ipynb
                             (same models without residual trimming)
  Attrition robustness ..... experiment1/matched_retention_check.R
                             -> matched_retention_slopes*.csv,
                                matched_retention_contrasts*.csv
  GAMM extraction .......... experiment1/data_processing_gamm.ipynb / .py
                             -> concatenated_data/f0_z_mean_f0_processed_for_gamm.csv
                             (visualized in Appendix A.3)

Experiment 2 (perceptual compensation)
  Data extraction .......... experiment2_experiment3/experiment2_data_processing.ipynb / .py
  Comprehensive model ...... experiment2_experiment3/experiment2_data_analysis.ipynb
  Concise model ............ experiment2_experiment3/experiment2_data_analysis_concise.ipynb
  Fluency follow-up ........ reported within the analysis notebooks

Experiment 3 (tone acceptance ranges)
  Data extraction .......... experiment2_experiment3/experiment3_data_processing.ipynb / .py
  Steepness ................ experiment3_data_analysis_slope_general.ipynb
                             experiment3_data_analysis_slope_advanced.ipynb
                             experiment3_data_analysis_slope_fluency.ipynb
  Threshold ................ experiment3_data_analysis_threshold_general.ipynb
                             experiment3_data_analysis_threshold_advanced.ipynb
                             experiment3_data_analysis_threshold_fluency.ipynb
  Model development ........ experiment3_logistic_regression_slope.ipynb
                             experiment3_probit_analysis_threshold.ipynb

-------------------------------------------------------------------------------
CONTENTS
-------------------------------------------------------------------------------
README.txt (this file)
participant_info.csv ........ anonymized participant information
experiment2_results.csv ..... collated Experiment 2 responses
synthesis/ .................. synthesized stimuli for Experiments 2 and 3
                              (Mandarin/, Min/)

experiment1/
  concatenated_data/ ........ combined processed data for all participants
  processed_data/ ........... per-participant processed data
  stimuli/ .................. stimuli used in the experiment
  visualization/ ............ figure code and fitted data for the paper
      linear_regression_visualization.ipynb/.R  (generates fitted data)
      plot_linear_regression.ipynb/.py          (plots)
      lmm_fit_man.csv, lmm_fit_man_bilingual.csv, lmm_fit_min.csv
      lmm_fit_carryover.png, lmm_fit_anticipatory.png
      gamm_visualization.ipynb/.R, plot_gamm.ipynb/.py
      gamm_predictions_all.csv
      gamm_visualization_{man,min}_{carryover,anticipatory}.png
      README.txt

experiment2_experiment3/
  concatenated_data/ ........ combined processed data for all participants
  processed_data/ ........... per-participant processed data
  raw_data/, raw_data.zip ... raw collected responses
  visualization/ ............ figure code for the paper

-------------------------------------------------------------------------------
DATA NOT INCLUDED HERE
-------------------------------------------------------------------------------
The Experiment 1 production recordings (experiment1/raw_data, ~1.4 GB of audio)
exceed GitHub's file-size limits and are archived separately at:

    [ADD ARCHIVE LINK / DOI]

All processed data needed to reproduce every analysis reported in the paper is
included in this repository.
