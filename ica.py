"""
ICA module of EEG - Data pipeline:
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

import os
import os
import numpy as np
from mne.preprocessing import ICA

def run_ica(raw_clean, params):
    """
    Orchestrates ICA artifact removal with DISCOVER-EEG repetition strategy.
    Runs ICA n_repetitions times and selects run closest to average bad-component mask.
    
    :param raw_clean: mne.io.Raw, Raw EEG data
    :param params: dict, Parameters from config (ica settings)
    :return: tuple, (mne.io.Raw cleaned data, mne.preprocessing.ICA fitted object, dict artifact info)
    """
    ica_params = params.get("ica", {})
    
    # Extract parameters
    random_state = ica_params.get("random_state", 42)
    n_repetitions = ica_params.get("n_repetitions", 2)
    
    print(f"\n{'='*60}")
    print(f"Running ICA {n_repetitions} times (DISCOVER-EEG strategy)")
    print(f"{'='*60}")
    
    # Run ICA multiple times
    ica_runs = []
    bad_component_masks = []
    artifact_info_list = []
    
    for run_idx in range(n_repetitions):
        print(f"\nRun {run_idx + 1}/{n_repetitions}")
        
        # Fit with different random state each time
        ica = fit_ica(
            raw_clean,
            params,
            random_state=random_state + run_idx,
        )
        
        # Identify bad components with artifact type breakdown
        artifact_info = identify_artifact_components(ica, raw_clean)
        bad_components = artifact_info['bad_components']
        
        print(f"  → Bad components: {bad_components} ({len(bad_components)}/{ica.n_components})")
        print(f"    • EOG: {artifact_info['artifact_types']['eog']}")
        print(f"    • Muscle: {artifact_info['artifact_types']['muscle']}")
        print(f"    • ECG: {artifact_info['artifact_types']['ecg']}")
        
        # Create binary mask
        print(f"number of components: {len(ica.ch_names)}")
        mask = np.zeros(len(ica.ch_names))
        mask[bad_components] = 1
        
        ica_runs.append(ica)
        bad_component_masks.append(mask)
        artifact_info_list.append(artifact_info)
    
    # Select best run
    print(f"\n{'='*60}")
    print("Selecting best ICA run")
    print(f"{'='*60}")
    
    best_run_idx = select_best_ica_run(ica_runs, np.array(bad_component_masks))
    ica = ica_runs[best_run_idx]
    best_artifact_info = artifact_info_list[best_run_idx]
    bad_components = best_artifact_info['bad_components']
    
    print(f"Selected: Run {best_run_idx + 1}")
    print(f"Components to remove: {bad_components}")
    print(f"Total: {len(bad_components)} / {len(ica.ch_names)}")
    print(f"  • EOG: {len(best_artifact_info['artifact_types']['eog'])}")
    print(f"  • Muscle: {len(best_artifact_info['artifact_types']['muscle'])}")
    print(f"  • ECG: {len(best_artifact_info['artifact_types']['ecg'])}\n")
    
    # Apply ICA
    ica.exclude = bad_components
    raw_clean = ica.apply(raw_clean)
    
    return raw_clean, best_artifact_info


def fit_ica(raw_clean, params, random_state):
    """
    Fits ICA on raw data with specified parameters.
    
    :param raw_clean: mne.io.Raw, Raw EEG data
    :param params: dict, Parameters from config (ica settings)
    :param random_state: int, Random state for reproducibility
    :return: mne.preprocessing.ICA, Fitted ICA object
    """
    ica_params = params.get("ica", {})
    n_components = ica_params.get("n_components", None)
    method = ica_params.get("method", "infomax")
    
    ica = ICA(
        n_components=n_components,
        random_state=random_state,
        method=method
    )
    ica.fit(raw_clean)
    print(f"ica={len(ica.ch_names)}")
    return ica


def identify_artifact_components(ica, raw_clean):
    """
    Identifies artifact-related ICA components using DISCOVER-EEG strategy.
    Checks: EOG, muscle, ECG artifacts.
    Returns detailed breakdown of artifact types.
    
    :param ica: mne.preprocessing.ICA, Fitted ICA object
    :param raw_clean: mne.io.Raw, Raw EEG data
    :return: dict, Contains 'bad_components' list and 'artifact_types' breakdown
    """
    bad_components = []
    artifact_types = {
        'eog': [],
        'muscle': [],
        'ecg': []
    }
    
    # 1. EOG-related components (eye movements)
    try:
        eog_indices, _ = ica.find_bads_eog(raw_clean)
        bad_components.extend(eog_indices)
        artifact_types['eog'] = eog_indices
    except ValueError:
        pass
    
    # 2. Muscle-related components
    try:
        muscle_indices, _ = ica.find_bads_muscle(raw_clean)
        bad_components.extend(muscle_indices)
        artifact_types['muscle'] = muscle_indices
    except ValueError:
        pass
    
    # 3. ECG-related components
    try:
        ecg_indices, _ = ica.find_bads_ecg(raw_clean)
        bad_components.extend(ecg_indices)
        artifact_types['ecg'] = ecg_indices
    except ValueError:
        pass
    
    # Remove duplicates while preserving artifact type information
    bad_components = list(set(bad_components))
    n_total = len(ica.ch_names)
    print(f"Identified {len(bad_components)} bad components out of {n_total}:")
    return {
        'bad_components': bad_components,
        'artifact_types': artifact_types,
        'n_total': n_total,
        'n_bad_components': len(bad_components)  # Unique count (deduplicated)
    }


def select_best_ica_run(ica_runs, bad_component_masks):
    """
    Selects the ICA run closest to the average bad-component mask.
    DISCOVER-EEG strategy for ICA stability.
    
    :param ica_runs: list, List of fitted ICA objects
    :param bad_component_masks: np.ndarray, Binary masks (n_runs × n_components)
    :return: int, Index of best run
    """
    # Compute average mask across all runs
    avg_mask = np.mean(bad_component_masks, axis=0)
    
    # Find run closest to average
    distances = [np.linalg.norm(mask - avg_mask) for mask in bad_component_masks]
    best_run_idx = np.argmin(distances)
    
    return best_run_idx


