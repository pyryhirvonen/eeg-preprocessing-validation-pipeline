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
import shutil
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

# Create output directories with new hierarchical structure
qc_root = "derivatives/quality_control"
validation_root = "derivatives/validation"
os.makedirs(qc_root, exist_ok=True)
os.makedirs(os.path.join(qc_root, "overall", "combined"), exist_ok=True)
os.makedirs(validation_root, exist_ok=True)

# Store PSD data for grand-average analysis
psd_data = {}  # {subject: {task: {'psd': psd_obj, 'raw': raw_obj}}}
qc_results = []  # Store QC metrics for CSV summary

# Create error log file
error_log_path = "derivatives/quality_control/error_log.txt"

# Iterate and run pipeline per subject/session/task
for subject in subjects:
    psd_data[subject] = {}
    
    # Create per-subject directory in quality_control
    subject_qc_dir = os.path.join(qc_root, f"sub-{subject}")
    os.makedirs(subject_qc_dir, exist_ok=True)
    
    # Create only raw and final directories (preprocess manages reref/ica/epoch)
    raw_dir = os.path.join(subject_qc_dir, "raw")
    final_dir = os.path.join(subject_qc_dir, "final")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    for session in sessions:
        for task in tasks:
            try:
                print(f"\n{'='*60}")
                print(f"Processing: subject={subject}, session={session}, task={task}")
                print(f"{'='*60}")
                
                # Load raw data
                raw = load_raw(subject, session, task, datatype, bids_root)
                print(raw.info)
                
                # Plot raw data
                fig = raw.plot(title=f"Raw Data - Subject {subject}, Session {session}, Task {task}", show=False)
                raw_plot_path = os.path.join(raw_dir, f"sub-{subject}_ses-{session}_task-{task}_raw.png")
                fig.savefig(raw_plot_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Saved raw data plot: {raw_plot_path}")
                # plot PSD of raw data (linear y-axis)
                psd_raw = compute_psd_own(raw, params)
                plot_subject_psd_qc(psd_raw, subject, raw_dir, suffix="_raw", task=task)
                print(f"✓ Saved raw data PSD figure for subject {subject}, task {task}")

                
                # Preprocess (manages reref/ica/epoch subdirectories and creates PSD plots)
                raw_clean, epochs, ica_artifact_info = preprocess_raw(raw, params, subject=subject, task=task, output_dir=subject_qc_dir)
                
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
                cleaned_plot_path = os.path.join(final_dir, f"sub-{subject}_ses-{session}_task-{task}_cleaned.png")
                fig.savefig(cleaned_plot_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Saved cleaned data plot: {cleaned_plot_path}")
                
                # Orchestrate QC (handles all QC plots and metrics)
                qc_combined_path = orchestrate_qc(
                    raw_clean, epochs, ica_artifact_info, subject, session, task, 
                 final_dir, params
                )
                qc_results.append(qc_combined_path['qc_record'])
                
                # Copy qc_combined to overall/combined/ for easy access
                if qc_combined_path['fig_path']:
                    combined_dir = os.path.join(qc_root, "overall", "combined")
                    os.makedirs(combined_dir, exist_ok=True)
                    shutil.copy2(qc_combined_path['fig_path'], os.path.join(combined_dir, os.path.basename(qc_combined_path['fig_path'])))
                    print(f"✓ Copied qc_combined to overall/combined/")
                    
            except Exception as e:
                error_msg = f"SKIPPED: sub-{subject} ses-{session} task-{task} - {str(e)}"
                print(f"✗ {error_msg}")
                with open(error_log_path, "a") as log:
                    log.write(error_msg + "\n")
                continue

# Plot grand-average PSD (all subjects combined, EO vs EC with log scale)
print(f"\n{'='*60}")
print("Generating grand-average PSD figure...")
print(f"{'='*60}")
qc_overall_dir = os.path.join(qc_root, "overall")
plot_grand_average_psd(psd_data, qc_overall_dir, validation_root)
print(f"✓ Grand-average PSD saved to {qc_overall_dir}/ and {validation_root}/")

# Run permutation cluster test (EO vs EC validation)
from permutation_test import run_permutation_cluster_test
run_permutation_cluster_test(psd_data, params, validation_root, qc_root)

# Generate QC summary CSV
if qc_results:
    generate_qc_summary_csv(qc_results, qc_overall_dir)
    print(f"✓ QC summary CSV saved to {qc_overall_dir}/qc_summary.csv")

print(f"\n{'='*60}")
print(f"Pipeline complete. QC outputs saved to {qc_root}/ and {validation_root}/")
print(f"{'='*60}")
