"""
Epoching module of EEG - Data pipeline:
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

import mne
import numpy as np
from plot_psd import compute_psd_own, plot_subject_psd_qc


def epoch_data(raw_clean, params, subject="", task="", output_dir=""):
    """
    Function to epoch data. Epoching parameters are read from params['epoching'].
    
    :param raw_clean: mne.io.Raw, Preprocessed raw data
    :param params: dict, Full params.json config
    :param subject: str, Subject ID for PSD plotting (optional, default: "")
    :param task: str, Task name for PSD plotting (optional, default: "")
    :param output_dir: str, Output directory for PSD figures (optional, default: "")
    :return: mne.Epochs, Epoched data
    """
    ep = params.get("epoching", {})
    
    print(f"\n  Epoching parameters:")
    print(f"    epoch_duration: {ep.get('epoch_duration', 2.0)}")
    print(f"    overlap: {ep.get('overlap', 1)}")
    print(f"    reject_by_annotation: {ep.get('reject_by_annotation', True)}")
    print(f"    Raw annotations before epoching: {raw_clean.annotations}")
    
    epochs=mne.make_fixed_length_epochs(
                    raw_clean,
                    duration=ep.get("epoch_duration", 2.0),
                    overlap=ep.get("overlap", 1),
                    preload=ep.get("preload", True),
                    reject_by_annotation=ep.get("reject_by_annotation", True))
    
    print(f"    Epochs created: {len(epochs)} epochs")
    
    # Plot PSD from epochs (stage: epochs)
    if subject and task and output_dir:
        # Remove annotations to allow multitaper PSD calculation
        epochs_clean = epochs.copy()
        if len(epochs_clean.annotations) > 0:
            epochs_clean.annotations.delete(np.arange(len(epochs_clean.annotations)))
        psd_epochs = compute_psd_own(epochs_clean, params)
        plot_subject_psd_qc(psd_epochs, subject, output_dir, task=task, stage="epochs")
        print(f"✓ Saved PSD from epochs: {subject}_task-{task}_epochs_psd.png")
    
    return epochs