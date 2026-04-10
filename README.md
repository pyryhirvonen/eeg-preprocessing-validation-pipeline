# EEG Data Pipeline — Bachelor's Thesis

A DISCOVER-EEG–inspired automated EEG processing pipeline implemented in Python, developed as a bachelor's thesis at Tampere University.

**Author:** Pyry Hirvonen  
**Contact:** pyry.hirvonen@tuni.fi  
**License:** MIT — see [LICENSE](LICENSE)

---

## Overview

This pipeline processes resting-state EEG data from the [TD-BRAIN](https://www.nature.com/articles/s41597-022-01413-4) sample dataset in a fully automated, reproducible, BIDS-compliant manner. The primary validation target is reproducing the well-known **eyes-open (EO) vs. eyes-closed (EC) alpha-band attenuation** (8–12 Hz), following the methodology of the [DISCOVER-EEG](https://www.nature.com/articles/s41597-023-02525-0) pipeline.

### Pipeline steps

**Preprocessing:**
1. **Load** raw BIDS-compliant EEG data via `mne-bids`
2. **Line-noise removal** — notch filter (50 Hz) for power-line interference
3. **Band-pass filter** — high-pass (1 Hz) and low-pass (100 Hz)
4. **Bad channel detection** — automated detection using RANSAC (`pyprep`)
5. **Re-reference** — average reference across all channels
6. **ICA** — extended Infomax ICA decomposition with artifact labeling via `mne-icalabel`
7. **Bad channel interpolation** — recover flagged channels using neighboring channels
8. **Bad segment detection** — automated detection of artifactual time segments
9. **Epoching** — create fixed-length overlapping windows (2 s, 50% overlap)
10. **QC plots** — generate per-subject quality control visualizations

**Validation:**
11. **PSD computation** — multitaper power spectral density (1–100 Hz) from epochs
12. **Cluster-based permutation test** — statistical comparison of EO vs EC across frequency spectrum
---

## Repository structure

```
├── main.py                    # Entry point — runs the full pipeline
├── params.json                # Centralized configuration (all parameters)
├── load_data.py               # BIDS data loading via mne-bids
├── preprocess.py              # Filtering, referencing, bad-channel detection, ICA orchestration
├── ica.py                     # ICA decomposition and artifact component identification
├── epoch.py                   # Fixed-length epoch creation
├── plot_psd.py                # PSD computation and visualization (log & linear scales)
├── permutation_test.py        # Cluster-based permutation test (EO vs. EC validation)
├── qc.py                      # QC report generation (combined QC figures and metrics)
├── explore_dataset.py         # Dataset structure exploration utility
├── requirements.txt           # Pinned Python dependencies
└── derivatives/               # Pipeline outputs
    ├── quality_control/       # Per-subject QC figures and metrics
    │   ├── overall/combined/  # Combined QC plots for all subjects
    │   └── sub-*/             # Per-subject subdirectories (raw, final stages)
    ├── validation/            # Permutation test results (CSV) and figures
    └── quality_control_final/ # Final consolidated outputs
```

---

## Requirements

- Python ≥ 3.10
- See [requirements.txt](requirements.txt) for all pinned dependencies

Key packages: `mne`, `mne-bids`, `mne-icalabel`, `pyprep`, `numpy`, `scipy`, `matplotlib`

---

## Installation

```bash
git clone https://github.com/pyryhirvonen/Bachelors-thesis---eeg-pipeline.git
cd Bachelors-thesis---eeg-pipeline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Configuration

All pipeline parameters are centralized in [`params.json`](params.json). Key configuration sections:

| Section | Parameters |
|---|---|
| **Input** | `bids_root`, `subjects`, `sessions`, `tasks`, `datatype` |
| **Preprocessing** | `notch_freq` (50 Hz), `hp_cutoff` (1 Hz), `lp_cutoff` (100 Hz), bad-channel thresholds (`bad_channel_peak`, `bad_channel_flat`, etc.) |
| **ICA** | `method` (infomax), `fit_params` (extended=true), `n_components`, `n_repetitions`, `artifact_threshold`, bad-segment thresholds |
| **Epoching** | `epoch_duration` (2.0 s), `overlap` (1.0 s), `preload`, `reject_by_annotation` |
| **PSD** | `fmin` (1 Hz), `fmax` (100 Hz), `method` (multitaper) |
| **Visualization** | Figure sizes, colors (EC=blue, EO=green), alpha-band highlighting (8–12 Hz), DPI settings, etc. |
| **QC** | Channel/IC/segment visualization parameters |

Edit `params.json` to customize preprocessing, ICA, epoching, and PSD parameters. The configuration is loaded by `main.py` and applied uniformly across all subjects for reproducibility.

---

## Usage

Before running, make sure [`params.json`](params.json) points to your local dataset path:

```json
"bids_root": "/absolute/path/to/TD-BRAIN-SAMPLE"
```

Then run:

```bash
# Activate environment
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Run the full pipeline across all subjects
python main.py

# Explore available subjects/sessions/tasks
python explore_dataset.py
```


### Output organization

Outputs are written to `derivatives/`:
- **`quality_control/`** — Per-subject QC figures organized by stage (raw → final)
  - `overall/combined/` — aggregated QC plots
  - `overall/qc_summary.csv` — per-subject metrics (channels flagged, ICs removed, epochs per condition)
- **`validation/`** — Statistical validation results
  - `permutation_results.csv` — EO vs EC cluster-based permutation test output
  - Grand-average PSD figure with log scale

---

## Data

This pipeline is designed for the **TD-BRAIN sample dataset**, a BIDS-compliant resting-state EEG dataset (20 subjects, 2 conditions: `restEC`, `restEO`).  
Dataset reference: van Dijk *et al.* (2022). *Scientific Data*, 9, 333. https://doi.org/10.1038/s41597-022-01413-4

---

## Implementation notes

### Key design decisions

1. **Configuration-driven**: All parameters live in `params.json` to ensure reproducibility and easy parameter sweeps.
2. **Per-subject organization**: Output directories follow `derivatives/quality_control/sub-{SUBJECT_ID}/` for clean organization.
3. **ICA multi-run strategy** (optional): Set `n_repetitions > 1` in `params.json` to run ICA multiple times and select the most stable decomposition (following DISCOVER-EEG). By default `n_repetitions = 10`.
4. **Integrated QC**: The `qc.py` module generates comprehensive per-subject figures combining channel detection, IC classification, and PSD comparison in a single plot (`qc_combined.png`).
5. **Statistical validation**: The permutation cluster test compares EO vs EC PSD across the full frequency range (1–100 Hz) and identifies significant frequency clusters where alpha attenuation occurs.

### DISCOVER-EEG alignment

This pipeline closely follows the DISCOVER-EEG methodology:
- **Preprocessing**: Notch (50 Hz) + high-pass (1 Hz) + low-pass (100 Hz) + average reference
- **ICA**: Extended Infomax with artifact component labeling via `mne-icalabel`
- **Epoching**: 2 s fixed-length windows with 50% overlap for robust spectral estimation
- **PSD**: Multitaper method (1–100 Hz) for stable power estimates
- **Validation**: Cluster-based permutation test for EO vs EC comparison

### Error handling

- Errors during processing are logged to `derivatives/quality_control/error_log.txt`
- Subjects with processing failures are skipped; the pipeline continues for other subjects
- Raw and cleaned data plots are saved before/after preprocessing for visual inspection

### Dependencies

Core packages:
- **`mne`** (1.11.0) — EEG processing and analysis
- **`mne-bids`** (0.17.0) — BIDS data I/O
- **`mne-icalabel`** (0.8.1) — Automated ICA component labeling
- **`pyprep`** (0.5.0) — Advanced preprocessing (bad-channel detection)
- **`numpy`** (2.3.5) — Numerical computing
- **`scipy`** (1.16.3) — Scientific computing
- **`matplotlib`** (3.10.7) — Data visualization
- **`pandas`** (3.0.1) — Data frame operations for QC summaries

---


## References

- **DISCOVER-EEG pipeline** (primary methodological reference):  
  Gil Ávila *et al.* (2023). *Scientific Data*, 10, 613. https://doi.org/10.1038/s41597-023-02525-0

- **MNE-Python**:  
  Gramfort *et al.* (2013). *Frontiers in Neuroscience*, 7, 267. https://doi.org/10.3389/fnins.2013.00267

- **MNE-BIDS**:  
  Appelhoff *et al.* (2019). *Journal of Open Source Software*, 4(44), 1896. https://doi.org/10.21105/joss.01896

  - **BIDS standard**:  
  Gorgolewski *et al.* (2016). *Scientific Data*, 3, 160044. https://doi.org/10.1038/sdata.2016.44

- **PyPrep** (bad-channel detection):  
  Appelhoff *et al.* (2021). PyPREP: A Python implementation of the preprocessing pipeline (PREP) for EEG data. https://github.com/sappelhoff/pyprep

- **PREP pipeline** (original methodology):  
  Bigdely-Shamlo *et al.* (2015). *Frontiers in Neuroinformatics*, 9, 16. https://doi.org/10.3389/fninf.2015.00016


