"""
Main file of EEG - Data pipeline
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

"""
Main orchestration file for EEG-data pipeline.

This script:
1. Loads raw EEG data from BIDS-compliant dataset
2. Preprocesses data (notch filter, bandpass, average reference)
3. Runs ICA artifact removal (DISCOVER-EEG multi-run strategy)
4. Creates fixed-length overlapping epochs
5. Computes Power Spectral Density (PSD)
6. Generates QC reports and visualizations

Configuration is centralized in params.json for reproducibility.
"""

import json
import os
import matplotlib.pyplot as plt

import mne
from load_data import load_raw
from preprocess import preprocess_raw
from plot_psd import plot_subject_psd_qc, plot_grand_average_psd, compute_psd_own
from qc import orchestrate_qc, generate_qc_summary_csv

# Load centralized configuration (params.json) for reproducibility
with open("params.json", "r") as _pf:
    params = json.load(_pf)

# Extract configuration
bids_root = params.get("bids_root")
subjects = params.get("subjects", [])
sessions = params.get("sessions", [])
tasks = params.get("tasks", [])
datatype = params.get("datatype")
viz_params = params.get("visualization", {})

# Create QC output directory
qc_dir = "derivatives/qc"
os.makedirs(qc_dir, exist_ok=True)

# Store PSD data for grand-average analysis
psd_data = {}  # {subject: {task: {'psd': psd_obj, 'raw': raw_obj}}}
qc_results = []  # Store QC metrics for CSV summary

# Iterate and run pipeline per subject/session/task
for subject in subjects:
    psd_data[subject] = {}
    
    for session in sessions:
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Processing: subject={subject}, session={session}, task={task}")
            print(f"{'='*60}")
            
            # Load raw data
            raw = load_raw(subject, session, task, datatype, bids_root)
            print(raw.info)
            
            # Plot raw data
            fig = raw.plot(title=f"Raw Data - Subject {subject}, Session {session}, Task {task}", show=False)
            raw_plot_path = os.path.join(qc_dir, f"sub-{subject}_ses-{session}_task-{task}_raw.png")
            fig.savefig(raw_plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved raw data plot: {raw_plot_path}")
            # plot PSD of raw data (linear y-axis)
            psd_raw = compute_psd_own(raw, params)
            plot_subject_psd_qc(psd_raw, subject, qc_dir, viz_params, suffix="_raw", task=task)
            print(f"✓ Saved raw data PSD figure for subject {subject}, task {task}")

            
            # Preprocess
            raw_clean, epochs, ica_artifact_info = preprocess_raw(raw, params)

            
            # Compute PSD from epochs (not raw)
            psd = compute_psd_own(epochs, params)
            
            # Store PSD for later (grand-average)
            psd_data[subject][task] = {
                'psd': psd,
                'raw': raw_clean,
                'epochs': epochs
            }
            
            # Plot per-subject QC EEG-timeseries
            fig = raw_clean.plot(title=f"Cleaned Data - Subject {subject}, Session {session}, Task {task}", show=False)
            cleaned_plot_path = os.path.join(qc_dir, f"sub-{subject}_ses-{session}_task-{task}_cleaned.png")
            fig.savefig(cleaned_plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved cleaned data plot: {cleaned_plot_path}")
            
            # Plot per-subject QC PSD (linear y-axis)
            plot_subject_psd_qc(psd, subject, qc_dir, viz_params, task=task)
            print(f"✓ Saved QC PSD figure for subject {subject}, task {task}")
            
            # Orchestrate QC (handles all QC plots and metrics)
            qc_record = orchestrate_qc(
                raw_clean, epochs, ica_artifact_info, subject, session, task, 
             qc_dir, params
            )
            qc_results.append(qc_record)

# Plot grand-average PSD (all subjects combined, EO vs EC with log scale)
print(f"\n{'='*60}")
print("Generating grand-average PSD figure...")
print(f"{'='*60}")
plot_grand_average_psd(psd_data, qc_dir, viz_params)
print(f"✓ Grand-average PSD saved to {qc_dir}/")

# Run permutation cluster test (EO vs EC validation)
from permutation_test import run_permutation_cluster_test
run_permutation_cluster_test(psd_data, params)

# Generate QC summary CSV
if qc_results:
    generate_qc_summary_csv(qc_results, qc_dir)
    print(f"✓ QC summary CSV saved to {qc_dir}/qc_summary.csv")

print(f"\n{'='*60}")
print(f"Pipeline complete. QC outputs saved to {qc_dir}/")
print(f"{'='*60}")
