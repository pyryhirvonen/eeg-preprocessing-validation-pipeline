"""
Data preprocessing quality control module of EEG - Data pipeline:
Bachelor's thesis
Generates QC summary CSV and visualization figures following DISCOVER-EEG guidelines.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def orchestrate_qc(raw_clean, epochs, ica_artifact_info, subject, session, task, subject_qc_dir, params):
    """
    Unified QC orchestration: generates all QC plots and returns QC metrics record.
    
    :param raw_clean: mne.io.Raw, Preprocessed raw data
    :param epochs: mne.Epochs or None, Epoched data
    :param subject: str, Subject ID
    :param session: str, Session ID
    :param task: str, Task name (restEC or restEO)
    :param ica_artifact_info: dict, ICA artifact information and removed components
    :param subject_qc_dir: str, Subject-specific output directory in quality_control/sub-{ID}/
    :param params: dict, Full params.json config
    :return: dict, Contains 'qc_record' (dict with metrics for CSV) and 'fig_path' (str path to qc_combined figure)
    """
    
    # Get bad channels count and ICA info
    n_bad_channels = ica_artifact_info.get('n_bad_channels', 0)  # Stored before interpolation
    n_ics_removed = ica_artifact_info.get('n_bad_components', 0)  # Use deduplicated count
    print(f"✓ Detected {n_bad_channels} bad channels and removed {n_ics_removed} ICA components for subject {subject}, task {task}")
    
    # Plot combined QC figure (bad channels, IC classification, bad segments)
    fig_path = plot_qc_combined(raw_clean, epochs, ica_artifact_info, subject, task, subject_qc_dir, params)
    
    # Count epochs for CSV
    n_epochs = 0
    if epochs is not None:
        n_epochs = len(epochs)
    
    # Return QC record for CSV and figure path
    qc_record = {
        'subject_id': subject,
        'session': session,
        'task': task,
        'n_channels_flagged': n_bad_channels,
        'n_ics_removed': n_ics_removed,
        'n_epochs': n_epochs
    }
    
    print(f"✓ QC orchestration complete for subject {subject}, task {task}")
    
    return {
        'qc_record': qc_record,
        'fig_path': fig_path
    }


def plot_qc_combined(raw_clean, epochs, ica_artifact_info, subject, task, subject_qc_dir, params):
    """
    Combined QC plot: bad channels, IC classification, and bad segments in one figure.
    Saves to subject directory and returns paths for copying.
    
    :param raw_clean: mne.io.Raw, Preprocessed raw data
    :param epochs: mne.Epochs or None, Epoched data
    :param ica_artifact_info: dict, ICA artifact information
    :param subject: str, Subject ID
    :param task: str, Task name
    :param subject_qc_dir: str, Subject-specific output directory
    :param params: dict, Parameters from config
    :return: str, Path to saved qc_combined figure (for later copying)
    """
    qc_params = params.get("qc", {})
    
    # Create figure with 3 subplots (stacked vertically)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.4)
    
    # ===== SUBPLOT 1: Bad Channels =====
    ax1 = fig.add_subplot(gs[0])
    ch_names = raw_clean.ch_names
    n_channels = len(ch_names)
    bad_channels = ica_artifact_info.get('bad_channels', [])
    
    for i, ch_name in enumerate(ch_names):
        is_bad = ch_name in bad_channels
        color = qc_params.get("color_bad", "red") if is_bad else "white"
        edge_color = qc_params.get("edge_color", "black")
        edge_width = qc_params.get("edge_width", 1.5)
        
        rect = mpatches.Rectangle((i, 0), 1, 1, linewidth=edge_width, edgecolor=edge_color, facecolor=color)
        ax1.add_patch(rect)
        
        if is_bad:
            ax1.text(i + 0.5, 0.5, ch_name, ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    ax1.set_xlim(0, n_channels)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f"Bad Channels (Subject {subject})", fontsize=12, fontweight='bold', loc='left')
    
    # ===== SUBPLOT 2: IC Classification =====
    ax2 = fig.add_subplot(gs[1])
    
    n_components = ica_artifact_info.get('n_total', 0)
    n_kept = n_components - len(ica_artifact_info.get('bad_components', []))
    artifact_types = ica_artifact_info.get('artifact_types', {})
    n_eye = len(artifact_types.get('eye', []))
    n_muscle = len(artifact_types.get('muscle', []))
    
    categories = ['kept ICs', 'muscle', 'eye']
    values = [n_kept, n_muscle, n_eye]
    colors = [
        qc_params.get("color_kept_ic", "steelblue"),
        qc_params.get("color_muscle", "coral"),
        qc_params.get("color_eye", "gold"),
    ]
    
    edge_color = qc_params.get("edge_color", "black")
    edge_width = qc_params.get("edge_width", 1.5)
    
    bars = ax2.barh(categories, values, color=colors, edgecolor=edge_color, linewidth=edge_width)
    ax2.set_xlabel('Number of Components', fontsize=11)
    ax2.set_title('IC Classification', fontsize=12, fontweight='bold', loc='left')
    ax2.set_xlim(0, max(values) * 1.2 if max(values) > 0 else 1)
    
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax2.text(width, bar.get_y() + bar.get_height()/2., f' {int(width)}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
    
    # ===== SUBPLOT 3: Bad Segments =====
    ax3 = fig.add_subplot(gs[2])
    
    duration_sec = raw_clean.times[-1]
    annotations = raw_clean.annotations
    
    color_good = qc_params.get("color_good", "steelblue")
    color_bad = qc_params.get("color_bad", "red")
    
    # Draw good segments (baseline)
    ax3.barh(0, duration_sec, height=0.5, color=color_good, edgecolor=edge_color, linewidth=edge_width)
    
    # Draw bad segments (annotate_amplitude creates descriptions like 'BAD_peak', 'BAD_flat', etc.)
    for ann in annotations:
        # Match any annotation starting with 'BAD' (handles BAD_peak, BAD_flat, BAD_unknown, etc.)
        if str(ann['description']).startswith('BAD'):
            onset = ann['onset']
            duration = ann['duration']
            ax3.barh(0, duration, left=onset, height=0.5, color=color_bad, edgecolor=edge_color, linewidth=edge_width)
    
    ax3.set_xlim(0, duration_sec)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_xlabel('Time (minutes)', fontsize=11)
    ax3.set_yticks([])
    ax3.set_title('Bad Segments', fontsize=12, fontweight='bold', loc='left')
    
    # Format x-axis to show minutes
    ax3.set_xticks(np.arange(0, duration_sec + 1, 60))
    ax3.set_xticklabels([int(x/60) for x in np.arange(0, duration_sec + 1, 60)])
    
    fig.suptitle(f"QC Report - Subject {subject}, Task {task}", fontsize=14, fontweight='bold', y=0.995)
    
    # Save to subject directory
    dpi = qc_params.get("dpi", 100)
    fig_path = os.path.join(subject_qc_dir, f"sub-{subject}_task-{task}_qc_combined.png")
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved combined QC figure: {fig_path}")
    
    return fig_path




def generate_qc_summary_csv(qc_results, output_dir):
    """
    Generate QC summary CSV with per-subject/task statistics.

    :param qc_results: list, List of dicts with QC metrics per subject/task
    :param output_dir: str, Output directory
    :return: None (saves CSV)
    """
    csv_path = os.path.join(output_dir, 'qc_summary.csv')
    
    fieldnames = ['subject_id', 'session', 'task', 'n_channels_flagged', 'n_ics_removed', 'n_epochs']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qc_results)
    
    print(f"✓ Saved QC summary CSV: {csv_path}")


