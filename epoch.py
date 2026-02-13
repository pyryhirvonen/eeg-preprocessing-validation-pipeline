"""
Epoching module of EEG - Data pipeline:
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

import mne


def epoch_data(raw_clean,params):
    """
    Function to epoch data. Epoching parameters are read from params['epoching'].
    
    :param raw_clean: mne.io.Raw, Preprocessed raw data
    :param params: dict, Full params.json config
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
    
    return epochs