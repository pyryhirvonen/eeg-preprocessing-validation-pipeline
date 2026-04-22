"""
Data preprocessing modules of EEG - Data pipeline:
Bachelor's thesis
"""

import os
import mne

from epoch import epoch_data
from ica import run_ica
from plot_psd import compute_psd_own, plot_subject_psd_qc

def preprocess_raw(raw, params, subject="", task="", output_dir=""):
    """
    Orchestrates all preprocessing steps.
    Steps 1-3: Filtering, bad channel detection, re-referencing
    Steps 4-6: ICA + interpolation + bad segments (handled by run_ica)
    Step 7: Epoching
    
    :param raw: mne.io.Raw, Raw EEG data
    :param params: dict, Full params.json config
    :param subject: str, Subject ID for PSD plotting (optional, default: "")
    :param task: str, Task name for PSD plotting (optional, default: "")
    :param output_dir: str, Output directory for PSD figures (optional, default: "")
    :return: tuple (mne.io.Raw, mne.Epochs, dict artifact_info)
    """
    pp = params.get("preprocessing", {})
    channel_setup = params.get("channel_setup", {})
    
    # Extract all parameters from params
    notch_freq = pp.get("notch_freq", 50)
    hp_cutoff = pp.get("hp_cutoff", 1)
    lp_cutoff = pp.get("lp_cutoff", 100)
    do_bandpass = pp.get("do_bandpass", True)
    do_notch = pp.get("do_notch", True)
    do_ref = pp.get("do_ref", True)
    ref_channels = pp.get("ref_channels", "average")
    do_bads_detection = pp.get("do_bads_detection", True)
    do_run_ica = pp.get("do_run_ica", True)
    do_epoching = pp.get("do_epoching", True)
    
    # Create stage-specific subdirectories within output_dir
    reref_dir = os.path.join(output_dir, "reref") if output_dir else ""
    ica_dir = os.path.join(output_dir, "ica") if output_dir else ""
    epoch_dir = os.path.join(output_dir, "epoch") if output_dir else ""
    
    if reref_dir:
        os.makedirs(reref_dir, exist_ok=True)
    if ica_dir:
        os.makedirs(ica_dir, exist_ok=True)
    if epoch_dir:
        os.makedirs(epoch_dir, exist_ok=True)
    
    raw_clean = raw.copy()

    # Step 1: Filtering (notch + bandpass)
    if do_notch:
        raw_clean = notch_filter(raw_clean, notch_freq)
    
    if do_bandpass:
        raw_clean = bandpass_filter(raw_clean, hp_cutoff, lp_cutoff)

    # Step 2: Bad channel detection using multiple criteria (flatness, SNR, RANSAC)
    if do_bads_detection:
        montage_name = channel_setup.get("montage_name", pp.get("montage_name", "standard_1020"))
        raw_clean = detect_bad_channels(raw_clean, pp, montage_name=montage_name)
    
    # Step 3: Re-reference using configured strategy
    if do_ref:
        raw_clean = average_reference(raw_clean, ref_channels=ref_channels)
    
    # Plot PSD after rereferencing (stage: reference)
    if subject and task and reref_dir:
        psd_reference = compute_psd_own(raw_clean, params)
        plot_subject_psd_qc(psd_reference, subject, reref_dir, task=task, stage="reference")
        print(f"✓ Saved PSD after rereferencing: {subject}_task-{task}_reference_psd.png")

    # Steps 4-6: ICA + interpolation + bad segments (DISCOVER-EEG repetition strategy)
    best_artifact_info = {}
    if do_run_ica:
        raw_clean, best_artifact_info = run_ica(raw_clean, params, subject=subject, task=task, output_dir=ica_dir)

    # Step 7: Epoching (includes PSD plot after epoching)
    epochs = None
    if do_epoching:
        epochs = epoch_data(raw_clean, params, subject=subject, task=task, output_dir=epoch_dir)

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

def average_reference(raw, ref_channels="average"):
    """
    function takes raw from preprocess_raw, returns object raw with new
    reference. 
    :param raw: obj, EEG-data and metadata that we are processing
    :param ref_channels: str | list, reference channel setting for MNE (default: "average", can also be list of channel names for specific reference)
    :return: object raw with new reference.
    """
    return raw.set_eeg_reference(ref_channels=ref_channels)





def detect_bad_channels(raw_clean, params, montage_name="standard_1020"):
    """
    Detects bad channels using multiple criteria via PyPrep's NoisyChannels:
    1. Flatness (signal variability)
    2. Noise-to-Signal ratio (SNR)
    3. RANSAC predictability
    
    :param raw_clean: mne.io.Raw, Preprocessed EEG data
    :param params: dict, Preprocessing parameters from params.json
    :param montage_name: str, Electrode montage name (default: "standard_1020")
    :return: mne.io.Raw with bad channels marked in raw.info['bads']
    """
    from pyprep import NoisyChannels

    montage = mne.channels.make_standard_montage(montage_name)
    raw_clean.set_montage(
        montage,
        match_case=False,
        match_alias=False,
        on_missing="ignore",
    )
    
    all_bads = set()  # Use a set to avoid duplicates
    
    # Initialize NoisyChannels
    nd = NoisyChannels(
        raw_clean,
        do_detrend=False,
        random_state=params.get('seed', 42),
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
    print(f"    Criterion 2: Noise-to-Signal ratio")
    nd.find_bad_by_SNR()
    bads_snr = nd.bad_by_SNR
    if bads_snr:
        all_bads.update(bads_snr)  # ← Use update() for sets
        print(f"  Bad channels detected by SNR: {len(bads_snr)} - {bads_snr}")
    else:
        print(f"  No bad channels detected by SNR")

    # ===== CRITERION 3: RANSAC =====
    print(f"    Criterion 3: RANSAC predictability")
    
    try:
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
