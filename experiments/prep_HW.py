"""
=============================================================================
 HOMEWORK -- RUN YOUR PREPROCESSING PIPELINE  (Part 1)
=============================================================================

Reads the raw BCIC2a .mat files and writes your features to
    datasets/BCIC2a/HW_prep/2_class/subject_dependent/

Usage
    python prep_HW.py

Prerequisite
    python download_datasets.py --dataset 'BCIC2a'
"""
import mixnet.preprocessing as prep

prep.BCIC2a.HW_prep.subject_dependent_setting(
    k_folds=5, 
    pick_smp_freq=100, 
    save_path='datasets', 
    num_class=2,
    # ---------------------------------------------------------------------------
    #  TODO -- add the hyper-parameters of YOUR pipeline here.
    # ---------------------------------------------------------------------------
    bands=[[4, 8], [8, 12], [12, 16], [16, 20], [20, 24],
           [24, 28], [28, 32], [32, 36], [36, 40]],
    filter_order=4,
    n_csp=8,          # CSP components per band; must match input_shape[1] in configs/HWNet.py
    )

