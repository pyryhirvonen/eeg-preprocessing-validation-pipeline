"""
ICA module of EEG - Data pipeline:
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi

Implements DISCOVER-EEG steps 4-6 with repetition strategy:
  Step 4: ICA decomposition + ICLabel artifact classification
  Step 5: Interpolation of removed channels
  Step 6: Bad time segment removal
Steps 4-6 are repeated n_repetitions times (default 10).
The run whose bad-segment mask is closest to the average mask is selected.
"""

import numpy as np
from mne.preprocessing import ICA, annotate_amplitude
from mne_icalabel import label_components
from plot_psd import compute_psd_own, plot_subject_psd_qc


def run_ica(raw_clean, params, subject="", task="", output_dir=""):
    """
    Orchestrates ICA artifact removal with DISCOVER-EEG repetition strategy.

    For each repetition:
      1. Fit ICA with different random seed
      2. Classify ICs with ICLabel, flag Muscle/Eye with prob > threshold
      3. Remove flagged ICs
      4. Interpolate bad channels (step 5)
      5. Detect bad time segments (step 6)
      6. Extract binary bad-segment mask

    After all repetitions, select the run whose bad-segment mask is closest
    to the average mask across all runs (DISCOVER-EEG strategy).

    :param raw_clean: mne.io.Raw, Raw EEG data (after filtering + bad channel detection)
    :param params: dict, Full configuration from params.json
    :param subject: str, Subject ID for PSD plotting (optional, default: "")
    :param task: str, Task name for PSD plotting (optional, default: "")
    :param output_dir: str, Output directory for PSD figures (optional, default: "")
    :return: tuple, (mne.io.Raw cleaned data, dict artifact_info)
    """
    ica_params = params.get("ica", {})

    # Extract parameters
    random_state = ica_params.get("random_state", 42)
    n_repetitions = ica_params.get("n_repetitions", 10)
    artifact_threshold = ica_params.get("artifact_threshold", 0.8)

    # Store bad channel info BEFORE any processing (needed for QC)
    bad_channels = list(raw_clean.info['bads']) if raw_clean.info['bads'] else []
    n_bad_channels = len(bad_channels)

    print(f"\n{'='*60}")
    print(f"DISCOVER-EEG Steps 4-6 ({n_repetitions} repetitions)")
    print(f"{'='*60}")

    # Storage for each repetition
    run_results = []
    segment_masks = []

    for run_idx in range(n_repetitions):
        print(f"\n--- Repetition {run_idx + 1}/{n_repetitions} ---")

        # Work on a fresh copy each time (ICA modifies data in-place)
        raw_copy = raw_clean.copy()

        # Step 4: Fit ICA
        ica = fit_ica(raw_copy, params, random_state=random_state + run_idx)

        # Step 4: Classify with ICLabel & identify artifacts
        artifact_info = identify_artifact_components(
            ica, raw_copy, artifact_threshold
        )
        bad_components = artifact_info['bad_components']

        print(f"  ICLabel: {len(bad_components)} bad / {artifact_info['n_total']} total")
        print(f"    • Eye: {artifact_info['artifact_types']['eye']}")
        print(f"    • Muscle: {artifact_info['artifact_types']['muscle']}")

        # Step 4: Remove flagged ICs
        ica.exclude = bad_components
        raw_copy = ica.apply(raw_copy)

        # Step 5: Interpolate bad channels
        if raw_copy.info['bads']:
            print(f"  Interpolating {len(raw_copy.info['bads'])} bad channels")
            raw_copy.interpolate_bads()

        # Step 6: Detect bad time segments
        raw_copy = detect_bad_segments(raw_copy, ica_params)

        # Extract binary mask for segment comparison
        mask = extract_bad_segment_mask(raw_copy)

        run_results.append({
            'ica': ica,
            'artifact_info': artifact_info,
            'raw_cleaned': raw_copy,
            'segment_mask': mask
        })
        segment_masks.append(mask)

    # Select best run based on bad-segment masks (DISCOVER-EEG strategy)
    print(f"\n{'='*60}")
    print("Selecting best run (segment-mask similarity)")
    print(f"{'='*60}")

    best_idx = select_best_run(segment_masks)
    best = run_results[best_idx]
    best_artifact_info = best['artifact_info']

    # Add bad channel info to artifact_info for QC reporting
    best_artifact_info['n_bad_channels'] = n_bad_channels
    best_artifact_info['bad_channels'] = bad_channels

    # Report segment mask stats
    mask = best['segment_mask']
    bad_pct = 100.0 * np.sum(mask) / len(mask) if len(mask) > 0 else 0

    print(f"Selected: Repetition {best_idx + 1}")
    print(f"Components removed: {best_artifact_info['bad_components']}")
    print(f"  • Eye: {len(best_artifact_info['artifact_types']['eye'])}")
    print(f"  • Muscle: {len(best_artifact_info['artifact_types']['muscle'])}")
    print(f"Bad segments: {bad_pct:.1f}% of recording\n")
    
    # Plot PSD after ICA (stage: ica) - before return
    if subject and task and output_dir:
        # Remove annotations to allow multitaper PSD calculation
        raw_for_psd = best['raw_cleaned'].copy()
        if len(raw_for_psd.annotations) > 0:
            raw_for_psd.annotations.delete(np.arange(len(raw_for_psd.annotations)))
        psd_ica = compute_psd_own(raw_for_psd, params)
        plot_subject_psd_qc(psd_ica, subject, output_dir, task=task, stage="ica")
        print(f"✓ Saved PSD after ICA: {subject}_task-{task}_ica_psd.png")

    return best['raw_cleaned'], best_artifact_info


def fit_ica(raw, params, random_state):
    """
    Fits ICA on raw data. Uses infomax (= MATLAB runica) by default with extended infomax.
    ICA is fitted only on clean channels (bad channels already removed).

    :param raw: mne.io.Raw, Raw EEG data
    :param params: dict, Full params.json config
    :param random_state: int, Random state for reproducibility
    :return: mne.preprocessing.ICA, Fitted ICA object
    """
    ica_params = params.get("ica", {})
    n_components = ica_params.get("n_components", None)
    method = ica_params.get("method", "infomax")
    fit_params = ica_params.get("fit_params", {"extended": True})

    ica = ICA(
        n_components=n_components,
        random_state=random_state,
        method=method,
        fit_params=fit_params
    )
    ica.fit(raw)
    print(f"  ICA fitted: {len(ica.ch_names)} components")
    return ica


def identify_artifact_components(ica, raw, artifact_threshold=0.8):
    """
    Classifies ICA components with ICLabel into 7 categories:
    Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other.

    Only components whose probability of being 'muscle artifact' or
    'eye blink' is > threshold are flagged (DISCOVER-EEG default: 80%).

    :param ica: mne.preprocessing.ICA, Fitted ICA object
    :param raw: mne.io.Raw, Raw EEG data
    :param artifact_threshold: float, Probability threshold (default 0.8)
    :return: dict with keys: bad_components, artifact_types, n_total, n_bad_components,
             ic_labels, ic_probs
    """
    # Run ICLabel classification
    ic_result = label_components(raw, ica, method='iclabel')
    labels = ic_result['labels']
    probs = ic_result['y_pred_proba']

    bad_components = []
    artifact_types = {
        'eye': [],
        'muscle': []
    }

    for idx, (label, prob) in enumerate(zip(labels, probs)):
        if label == 'eye blink' and prob > artifact_threshold:
            bad_components.append(idx)
            artifact_types['eye'].append(idx)
        elif label == 'muscle artifact' and prob > artifact_threshold:
            bad_components.append(idx)
            artifact_types['muscle'].append(idx)

    n_total = len(ica.ch_names)
    return {
        'bad_components': bad_components,
        'artifact_types': artifact_types,
        'n_total': n_total,
        'n_bad_components': len(bad_components),
        'ic_labels': labels,
        'ic_probs': probs
    }


def detect_bad_segments(raw, ica_params):
    """
    Detects bad time segments using amplitude-based criteria (DISCOVER-EEG step 6).
    Annotates the raw object with BAD_ annotations.

    :param raw: mne.io.Raw, EEG data after ICA + interpolation
    :param ica_params: dict, ICA parameters (includes bad_segment settings)
    :return: mne.io.Raw with bad-segment annotations
    """
    annotations, _ = annotate_amplitude(
        raw,
        peak=ica_params.get('bad_segment_peak', 200e-6),
        flat=ica_params.get('bad_segment_flat', 1e-6),
        min_duration=ica_params.get('bad_segment_min_duration', 5),
        bad_percent=ica_params.get('bad_segment_bad_percent', 20),
    )

    n_bad = len(annotations)
    total_bad_sec = sum(ann['duration'] for ann in annotations)
    print(f"  Bad segments: {n_bad} annotations, {total_bad_sec:.1f}s total")

    raw.set_annotations(raw.annotations + annotations)
    return raw


def extract_bad_segment_mask(raw):
    """
    Converts raw.annotations into a binary sample-level mask.
    1 = bad sample, 0 = good sample.
    Used for DISCOVER-EEG repetition selection strategy.

    :param raw: mne.io.Raw with annotations
    :return: numpy array of shape (n_samples,), dtype int
    """
    n_samples = len(raw.times)
    mask = np.zeros(n_samples, dtype=int)

    for ann in raw.annotations:
        if str(ann['description']).startswith('BAD'):
            onset_sample = int(ann['onset'] * raw.info['sfreq'])
            end_sample = int((ann['onset'] + ann['duration']) * raw.info['sfreq'])
            # Clip to valid range
            onset_sample = max(0, onset_sample)
            end_sample = min(n_samples, end_sample)
            mask[onset_sample:end_sample] = 1

    return mask


def select_best_run(segment_masks):
    """
    Selects the repetition whose bad-segment mask is most similar to
    the average mask across all repetitions (DISCOVER-EEG strategy).

    This reduces variability of rejected time segments caused by the
    non-deterministic nature of ICA.

    :param segment_masks: list of numpy arrays, binary masks (n_runs × n_samples)
    :return: int, Index of best run
    """
    masks_array = np.array(segment_masks, dtype=float)
    avg_mask = np.mean(masks_array, axis=0)

    distances = [np.linalg.norm(m - avg_mask) for m in masks_array]
    best_idx = np.argmin(distances)

    print(f"  Mask distances: {[f'{d:.1f}' for d in distances]}")
    print(f"  Best run: {best_idx + 1} (distance={distances[best_idx]:.1f})")

    return best_idx

