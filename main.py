"""
Main file of EEG - Data pipeline
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

import json
import os
from load_data import load_raw
from preprocess import preprocess_raw
from plot_psd import plot_subject_psd_qc, plot_grand_average_psd

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
            
            # Preprocess
            pp = params.get("preprocessing", {})
            raw_clean = preprocess_raw(
                raw,
                l_freq=pp.get("hp_cutoff", 0.5),
                h_freq=pp.get("lp_cutoff", 100),
                notch_freqs=pp.get("notch_freq", 50),
                do_bandpass=True,
                do_notch=True,
                do_ref=pp.get("apply_reference", True),
            )
            
            # Compute PSD
            psd_params = params.get("psd", {})
            psd = raw_clean.compute_psd(
                fmin=psd_params.get("fmin", 1),
                fmax=psd_params.get("fmax", 100),
                method=psd_params.get("method", "multitaper")
            )
            
            # Store PSD for later (grand-average)
            psd_data[subject][task] = {
                'psd': psd,
                'raw': raw_clean
            }
            
            # Plot per-subject QC PSD (linear y-axis)
            plot_subject_psd_qc(psd, subject, qc_dir, viz_params)
            print(f"✓ Saved QC PSD figure for subject {subject}, task {task}")

# Plot grand-average PSD (all subjects combined, EO vs EC with log scale)
plot_grand_average_psd(psd_data, qc_dir, viz_params)
print(f"\n✓ Grand-average PSD saved to {qc_dir}/")
print(f"\nPipeline complete. QC outputs saved to {qc_dir}/")
