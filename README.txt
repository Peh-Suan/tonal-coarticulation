###
author: Po-Hsuan Huang
email: pohsuan@usc.edu
###

This repository contains all the collected data and experiment materials for the study "Production and perception of coarticulated tones: The cases of Taiwan Mandarin and Taiwan Southern Min".

Files:

README.txt (this file)
participant_info.csv: anonymized participant information
- synthesis (the synthesized stimuli for Exp. 2 & 3)
- experiemnt1
    - concatenated_data (the combined processed data of all participants, generated with `data_processing.py/.ipynb`)
    - processed_data (the processed data of all participants, generated with `data_processing.ipynb/.py` )
    - raw data (the production data)
    - stimuli (the stimuli used in the experiment)
    - visualization (implementation of the visualization for the paper)
        linear_regression_visualization.ipynb/.R (for generating the fitted data)
        lmm_fit_anticipatory.png (LMM plot)
        lmm_fit_carryover.png (LMM plot)
        lmm_fit_man_bilingual.csv (fitted data)
        lmm_fit_man.csv (fitted data)
        lmm_fit_min.csv (fitted data)
        gamm_predictions_all.csv (fitted data)
        gamm_visualization_man_anticipatory.png (GAMM plot)
        gamm_visualization_man_carryover.png (GAMM plot)
        gamm_visualization_min_anticipatory.png (GAMM plot)
        gamm_visualization_min_carryover.png (GAMM plot)
        gamm_visualization.ipynb/.R (for generating the fitted data)
        plot_gamm.ipynb/.py (for plotting)
        plot_linear_regression.ipynb/.py (for plotting)
        README.txt
    data_analysis.ipynb/.R (for conducting LMM)
    data_analysis.pdf (the results of the analysis)
    data_processing_different_positions.ipynb/.py (for data processing, with the context tone onset/offset extracted from different portions)
    data_processing.ipynb/.py (for data processing)
    data_processing_gamm.ipynb/.py (for GAMM data processing)
- experiemnt2_experiment3
    - concatenated_data (the combined processed data of all participants, generated with `data_processing.py/.ipynb`)
    - raw data (the raw collected data)
    - visualization (implementation of the visualization for the paper)
    experiemnt2_data_analysis.ipynb/.R (for conducting the logistic regression analysis in Exp. 2)
    experiemnt2_data_analysis.pdf (the results of the analysis)
    experiment2_data_processing.ipynb/.py (for data processing, Exp. 2)
    experiment3_data_analysis_slope_advanced.ipynb/.R (for conducting LMM, advanced speakers only)
    experiment3_data_analysis_slope_advanced.pdf (the results of the analysis)
    experiment3_data_analysis_slope_general.ipynb/.R (for conducting LMM)
    experiment3_data_analysis_slope_general.pdf (the results of the analysis)
    experiment3_data_analysis_threshold_advanced.ipynb/.R (for conducting LMM, advanced speakers only)
    experiment3_data_analysis_threshold_advanced.pdf (the results of the analysis)
    experiment3_data_analysis_threshold_general.ipynb/.R (for conducting LMM)
    experiment3_data_analysis_threshold_general.pdf (the results of the analysis)
    experiment3_data_processing.ipynb/.py (for data processing, Exp. 3)