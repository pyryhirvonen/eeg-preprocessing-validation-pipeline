"""
Data preprocessing modules of EEG - Data pipeline:
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

import os
import mne

from epoch import epoch_data
from ica import run_ica

def preprocess_raw(raw,params):
    """
    Orchestrates all preprocessing steps.
    Reads all parameters from params['preprocessing'].
    
    :param raw: mne.io.Raw, Raw EEG data
    :param params: dict, Full params.json config
    :return: mne.io.Raw, Preprocessed data
    """
    pp = params.get("preprocessing", {})
    
    # Extract all parameters from params
    notch_freq = pp.get("notch_freq", 50)
    hp_cutoff = pp.get("hp_cutoff", 0.5)
    lp_cutoff = pp.get("lp_cutoff", 100)
    do_bandpass = pp.get("do_bandpass", True)
    do_notch = pp.get("do_notch", True)
    do_ref = pp.get("do_ref", True)
    do_bads_detection = pp.get("do_bads_detection", True)
    do_run_ica = pp.get("do_run_ica", True)
    do_interpolate_bads = pp.get("do_interpolate_bads", True)
    do_bad_segments_detection = pp.get("do_bad_segments_detection", True)
    do_epoching = pp.get("do_epoching", True)
    
    raw_clean = raw.copy()
    bads = []

   
    

    # Apply preprocessing steps

    
    if do_notch:
        raw_clean = notch_filter(raw_clean, notch_freq)
    
    if do_bandpass:
        raw_clean = bandpass_filter(raw_clean, hp_cutoff, lp_cutoff)
    
    
    if do_ref:
        raw_clean = average_reference(raw_clean)
     # Bad channel and segment detection is now skipped, as it needs debugging.
    if do_bads_detection:
        raw_clean = detect_bad_channels(raw_clean, pp)   

    if do_run_ica:
        raw_clean, best_artifact_info = run_ica(raw_clean, pp)

    # Store bad channel info BEFORE interpolation (which clears info['bads'])
    bad_channels = list(raw_clean.info['bads']) if raw_clean.info['bads'] else []
    n_bad_channels = len(bad_channels)
    best_artifact_info['n_bad_channels'] = n_bad_channels
    best_artifact_info['bad_channels'] = bad_channels  # Store names for plotting

    if do_interpolate_bads:
        raw_clean = interpolate_bads(raw_clean)
    
    # Detect bad segments AFTER ICA and interpolation (per DISCOVER-EEG pipeline)
    if do_bad_segments_detection:
        raw_clean = detect_bad_segments(raw_clean, pp)
    
    if do_epoching:
        epochs = epoch_data(raw_clean, pp)

    return raw_clean, epochs, best_artifact_info

def bandpass_filter(raw,l_freq,h_freq):
    """
    function takes parameters from preprocess_raw, returns bandpass
    filtered object raw. Filter frequency limits are l_freq and h_freq, they are
    decided in main.py
    :param raw: obj, EEG-data and metadata that we are processing
    :param l_freq: int, lower frequency limit
    :param h_freq: int, higher frequency limit
    :return: obj, filtered raw
    """
    return raw.filter(l_freq,h_freq)

def notch_filter(raw,freqs):
    """
    function takes parameters from preprocess_raw, returns notch
    filtered object raw. Filter frequencies are in "freqs", decided in main.py
    :param raw: obj, EEG-data and metadata that we are processing
    :param freqs: int, frequencies where notch filter is applied at.
    :return: obj, filtered raw
    """
    return raw.notch_filter(freqs)

def average_reference(raw):
    """
    function takes raw from preprocess_raw, returns object raw with new
    average reference.
    :param raw: obj, EEG-data and metadata that we are processing
    :return: object raw with new average reference.
    """
    return raw.set_eeg_reference(ref_channels="average")





def detect_bad_channels(raw_clean, params):
    """
    Detects bad channels using multiple criteria:
    1. Flatness (signal variability)
    2. Noise-to-Signal ratio (SNR)
    3. RANSAC predictability (requires temporary epoching)
    
    :param raw_clean: mne.io.Raw, Preprocessed EEG data
    :param params: dict, Parameters from params.json
    :return: set, Bad channel names
    """
    from pyprep import NoisyChannels
    from autoreject import Ransac
    import numpy as np

    raw_clean.set_channel_types({
    "VPVA": "misc",
    "VNVB": "misc",
    "HPHL": "eog",
    "HNHR": "eog",
    "OrbOcc": "eog",
    "Erbs": "ecg",
    "Mass": "emg",
})
    montage = mne.channels.make_standard_montage("standard_1020")
    raw_clean.set_montage(montage, match_case=False,match_alias=False,on_missing='raise')
    
    all_bads = set()  # Use a set to avoid duplicates
    
    # Initialize NoisyChannels
    nd = NoisyChannels(
        raw_clean,
        do_detrend=False,
        random_state=params.get('discover_random_state', 42),
        matlab_strict=False,
        ransac=False
    )

    # ===== CRITERION 1: FLATNESS =====
    print(f"    Criterion 1: Flatness (signal variability)...")
    nd.find_bad_by_nan_flat(flat_threshold=params.get('bad_channel_flat', 1e-6))
    bads_flat = nd.bad_by_flat
    if bads_flat:
        all_bads.update(bads_flat)  # ← Use update() for sets
        print(f"  Bad channels detected by flatness: {len(bads_flat)} - {bads_flat}")
    else:
        print(f"  No bad channels detected by flatness")

    # ===== CRITERION 2: NOISE-TO-SIGNAL RATIO =====
    print(f"    Criterion 2: Noise-to-Signal ratio (z-score threshold)...")
    nd.find_bad_by_SNR()
    bads_snr = nd.bad_by_SNR
    if bads_snr:
        all_bads.update(bads_snr)  # ← Use update() for sets
        print(f"  Bad channels detected by SNR: {len(bads_snr)} - {bads_snr}")
    else:
        print(f"  No bad channels detected by SNR")

    # ===== CRITERION 3: RANSAC =====
    print(f"    Criterion 3: RANSAC predictability (≥80% good)...")
    
    try:
        
        from pyprep import NoisyChannels
        nd.find_bad_by_ransac()
        print(f"\n✓ RANSAC analysis complete")
        print(f"Bad channels detected by RANSAC: {nd.bad_by_ransac}")
        print(f"Number of bad channels by RANSAC: {len(nd.bad_by_ransac)}")
        if nd.bad_by_ransac:
            all_bads.update(nd.bad_by_ransac)  # ← Use update() for sets
    except Exception as e:
        print(f"      ⚠ RANSAC failed: {str(e)}")
        print(f"      ⚠ Skipping RANSAC criterion.")

    # ===== SUMMARY =====
    bads = list(all_bads)
    if bads:
        raw_clean.info['bads'] = list(set(raw_clean.info.get('bads', []) + bads))
        print(f"\n  Total unique bad channels detected: {len(bads)} - {bads}")
    else:
        print(f"\n  No bad channels detected by any criterion.")
    print(f"  Current bad channels in info: {raw_clean.info['bads']}")    
    return raw_clean

def interpolate_bads(raw_clean):
    """
    Interpolates bad channels in the raw data.
    
    :param raw_clean: mne.io.Raw, Preprocessed EEG data with bad channels marked
    :return: mne.io.Raw, Raw data with bad channels interpolated
    """
    if raw_clean.info['bads']:
        print(f"  Interpolating {len(raw_clean.info['bads'])} bad channels: {raw_clean.info['bads']}")
        raw_clean.interpolate_bads()
    else:
        print(f"  No bad channels to interpolate.")
    print(f"  Bad channels after interpolation: {raw_clean.info['bads']}")    
    return raw_clean

def detect_bad_segments(raw_clean,params):
    """ Detects bad segments in the raw data using amplitude-based criteria."""

    import numpy as np
    from mne.preprocessing import annotate_amplitude
    
    annotations,bads=annotate_amplitude(raw_clean,peak=params.get('bad_segment_peak', 200e-6),
                                        flat=params.get('bad_segment_flat', 1e-6),
                                        min_duration=params.get('bad_segment_min_duration', 5),
                                        bad_percent=params.get('bad_segment_bad_percent', 20),)
    
    print(f"\n  Bad segments detected by annotate_amplitude:")
    print(f"    Number of bad segments: {len(annotations)}")
    print(f"    Annotations: {annotations}")
    
    raw_clean.set_annotations(annotations)
    
    print(f"    Annotations set on raw object")

    return raw_clean

