# EEG Data Pipeline — Bachelor's Thesis

A DISCOVER-EEG–inspired automated EEG processing pipeline implemented in Python, developed as a bachelor's thesis at Tampere University.

**Author:** Pyry Hirvonen  
**Contact:** pyry.hirvonen@tuni.fi  
**License:** MIT — see [LICENSE](LICENSE)

---

## Overview

This pipeline processes resting-state EEG data from the [TD-BRAIN](https://www.nature.com/articles/s41597-022-01409-z) sample dataset in a fully automated, reproducible, BIDS-compliant manner. The primary validation target is reproducing the well-known **eyes-open (EO) vs. eyes-closed (EC) alpha-band attenuation** (8–12 Hz), following the methodology of the [DISCOVER-EEG](https://www.nature.com/articles/s41597-023-02525-0) pipeline.



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

All pipeline parameters are centralized in [`params.json`](params.json). 
Edit `params.json` to customize preprocessing, ICA, epoching, and PSD parameters. The configuration is loaded by `main.py` and applied uniformly across all subjects for reproducibility, exception to this is plot_psd.py which also loads params.

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

### Error handling

- Errors during processing are logged to `derivatives/quality_control/error_log.txt`
- Subjects with processing failures are skipped; the pipeline continues for other subjects
- Raw and cleaned data plots are saved before/after preprocessing for visual inspection


## Data

This pipeline is designed for the **TD-BRAIN sample dataset**, a BIDS-compliant resting-state EEG dataset (20 subjects, 2 conditions: `restEC`, `restEO`).  
Dataset reference: van Dijk, H., van Wingen, G., Denys, D. *et al.* (2022). *The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database*. *Scientific Data*, 9, 333. https://doi.org/10.1038/s41597-022-01409-z

---
### Pipeline steps

#### Preprocessing
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

#### Validation
11. **PSD computation** — multitaper power spectral density (1–100 Hz) from epochs
12. **Cluster-based permutation test** — statistical comparison of EC vs EO across frequency spectrum
---

## Repository structure

```
├── main.py                    # Entry point — runs the full pipeline
├── params.json                # Centralized configuration (all parameters)
├── load_data.py               # BIDS data loading via mne-bids
├── preprocess.py              # Filtering, referencing, bad-channel detection
├── ica.py                     # ICA decomposition and artifact component identification, bad time segment detection and bad channel interpolation
├── epoch.py                   # Fixed-length epoch creation
├── plot_psd.py                # PSD computation and visualization
├── permutation_test.py        # Cluster-based permutation test
├── qc.py                      # QC report generation 
├── explore_dataset.py         # Dataset structure exploration utility
├── requirements.txt           # Pinned Python dependencies
└── derivatives/               # Pipeline outputs (generated)
    ├── quality_control/
    │   ├── error_log.txt                    # Processing errors logged during pipeline run
    │   ├── overall/
    │   │   ├── combined/                    # Aggregated QC plots across all subjects
    │   │   │   └── sub-*_task-*_qc_combined.png
    │   │   ├── qc_summary.csv               # Per-subject metrics
    │   │   └── grand_average_psd_eo_vs_ec.png
    │   └── sub-{SUBJECT_ID}/
    │       ├── raw/                         # Raw data plots and PSD
    │       ├── reref/                       # After re-referencing (intermediate)
    │       ├── ica/                         # After ICA artifact removal (intermediate)
    │       ├── epoch/                       # After epoching (intermediate)
    │       └── final/                       # Final cleaned data and QC
    └── validation/
        ├── permutation_results.csv          # EC vs EO cluster-based permutation test
        ├── permutation_test_clusters.png    # Significant frequency clusters visualization
        └── grand_average_psd_eo_vs_ec.png  # Grand-average PSD with statistical highlighting
```

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


