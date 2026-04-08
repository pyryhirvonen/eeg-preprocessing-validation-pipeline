"""
PSD visualization and analysis module.
Per-subject QC plot and grand-average EO vs EC PSD comparison.
"""

import json
import mne
import numpy as np
import matplotlib.pyplot as plt

# Load centralized configuration at module level for visualization parameters
with open("params.json", "r") as _params_file:
    _params = json.load(_params_file)
viz_params = _params.get("visualization", {})

def compute_psd_own(data,params):
    """
    Computes Power Spectral Density (PSD) using specified parameters.
    
    :param data: mne.io.Raw or mne.Epochs, Input EEG data
    :param params: dict, PSD computation parameters from config
    :return: mne.time_frequency.Spectrum, Computed PSD object
    """
    psd_params = params.get("psd", {})
    psd = data.compute_psd(
        fmin=psd_params.get("fmin", 1),
        fmax=psd_params.get("fmax", 100),
        method=psd_params.get("method", "multitaper")
    )
    # Average across channels and epochs
    if isinstance(data, mne.Epochs):
        psd_avg = psd.average()
    else:
        psd_avg = psd
    
    return psd_avg


def plot_subject_psd_qc(psd, subject, output_dir, suffix="", task="", stage="", ylim=None):
    """
    Function plots per-subject PSD (QC figure) with linear y-axis.
    
    :param psd: mne.time_frequency.Spectrum, Computed PSD object
    :param subject: str, Subject ID
    :param output_dir: str, Output directory for figure
    :param suffix: str, Optional suffix for filename (default: "", used for backward compatibility)
    :param task: str, Task name for filename (default: "")
    :param stage: str, Processing stage for filename (default: "", options: "reference", "ica", "epochs")
    :param ylim: tuple or list, Y-axis limits [ymin, ymax] (default: None, auto-scale)
    """
    fig, ax = plt.subplots(figsize=tuple(viz_params.get('figsize_qc', [10, 6])))
    
    # Extract PSD data
    psd_array = psd.get_data(exclude="bads")
    freqs = psd.freqs
    
    # Determine color based on task
    if "EO" in task or "eo" in task:
        line_color = viz_params.get('color_eo', 'green')
        task_label = "Eyes Open"
    elif "EC" in task or "ec" in task:
        line_color = viz_params.get('color_ec', 'blue')
        task_label = "Eyes Closed"
    else:
        line_color = viz_params.get('color_qc', 'green')
        task_label = task
    
    # Plot mean PSD across channels (linear scale)
    psd_mean = np.mean(psd_array, axis=0)
    ax.plot(freqs, psd_mean, linewidth=viz_params.get('linewidth_qc', 1.2), color=line_color)
    
    # Highlight alpha band
    alpha_min = viz_params.get('alpha_band_min', 8)
    alpha_max = viz_params.get('alpha_band_max', 13)
    ax.axvspan(alpha_min, alpha_max, 
               alpha=viz_params.get('alpha_transparency', 0.15), color=viz_params.get('color_alpha_band', 'gray'), 
               label=f"Alpha band ({alpha_min}-{alpha_max} Hz)")
    
    task_str = f" - {task_label}" if task else ""
    # Title with stage indicator
    stage_title = ""
    if stage:
        stage_title = f" ({stage})"
    
    ax.set_title(f"PSD QC: Subject {subject}{task_str}{stage_title}", fontsize=viz_params.get('fontsize_title', 14), fontweight='bold')
    ax.set_xlabel("Frequency (Hz)", fontsize=viz_params.get('fontsize_label', 12))
    ax.set_ylabel("Power (V²/Hz)", fontsize=viz_params.get('fontsize_label', 12))
    ax.set_xlim([1, 100])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=viz_params.get('grid_alpha', 0.3))
    ax.legend()
    
    # Save figure - handle both old (suffix-based) and new (stage-based) naming
    task_str_file = f"_task-{task}" if task else ""
    
    if stage:
        # New naming: sub-{ID}_task-{TASK}_{stage}_psd.png
        filename = f"sub-{subject}{task_str_file}_{stage}_psd.png"
    else:
        # Old naming (backward compatibility): sub-{ID}_task-{TASK}_psd_qc{suffix}.png
        filename = f"sub-{subject}{task_str_file}_psd_qc{suffix}.png"
    
    fig.savefig(f"{output_dir}/{filename}", dpi=viz_params.get('dpi', 300), bbox_inches="tight")
    plt.close(fig)

def plot_grand_average_psd(psd_data, qc_overall_dir, validation_root):
    """
    Function plots grand-average PSD across all subjects for EO vs EC comparison with log scale y-axis and EO/EC overlaid.
    Saves to both quality_control/overall/ and validation/ directories.
    
    :param psd_data: dict, {subject: {task: {'psd': psd_obj, 'raw': raw_obj}}}
    :param qc_overall_dir: str, Quality control overall directory
    :param validation_root: str, Validation root directory
    """
    eo_psds = []
    ec_psds = []
    eo_freqs_list = []
    ec_freqs_list = []
    
    # Collect PSDs by task (EO vs EC)
    for subject, tasks_dict in psd_data.items():
        for task, data in tasks_dict.items():
            psd = data['psd']
            
            psd_array = psd.get_data(exclude="bads")
            psd_mean = np.mean(psd_array, axis=0)
            freqs = psd.freqs
            
            if "EO" in task or "eo" in task:
                eo_psds.append(psd_mean)
                eo_freqs_list.append(freqs)
            elif "EC" in task or "ec" in task:
                ec_psds.append(psd_mean)
                ec_freqs_list.append(freqs)
    
    # Interpolate all PSDs onto a common frequency grid to handle
    # subjects with different sampling rates / frequency resolutions
    common_freqs = np.linspace(1, 100, 500)
    
    if ec_psds:
        ec_interp = [np.interp(common_freqs, f, p) for f, p in zip(ec_freqs_list, ec_psds)]
        ec_grand_avg = np.mean(ec_interp, axis=0)
        ec_freqs = common_freqs
    else:
        ec_grand_avg = None
        ec_freqs = None
    
    if eo_psds:
        eo_interp = [np.interp(common_freqs, f, p) for f, p in zip(eo_freqs_list, eo_psds)]
        eo_grand_avg = np.mean(eo_interp, axis=0)
        eo_freqs = common_freqs
    else:
        eo_grand_avg = None
        eo_freqs = None
    
    # Plot grand-average PSD
    fig, ax = plt.subplots(figsize=tuple(viz_params.get('figsize_grand', [12, 7])))
    
    if ec_grand_avg is not None and ec_freqs is not None:
        ax.semilogy(ec_freqs, ec_grand_avg, label=f"Eyes Closed (n={len(ec_psds)})", 
                    linewidth=viz_params.get('linewidth_grand', 1.5), color=viz_params.get('color_ec', 'blue'))
    if eo_grand_avg is not None and eo_freqs is not None:
        ax.semilogy(eo_freqs, eo_grand_avg, label=f"Eyes Open (n={len(eo_psds)})", 
                    linewidth=viz_params.get('linewidth_grand', 1.5), color=viz_params.get('color_eo', 'green'))
    
    # Highlight alpha band
    alpha_min = viz_params.get('alpha_band_min', 8)
    alpha_max = viz_params.get('alpha_band_max', 13)
    ax.axvspan(alpha_min, alpha_max, 
               alpha=viz_params.get('alpha_transparency', 0.15), color=viz_params.get('color_alpha_band', 'gray'), 
               label=f"Alpha band ({alpha_min}-{alpha_max} Hz)")
    
    ax.set_xlabel("Frequency (Hz)", fontsize=viz_params.get('fontsize_grand_label', 12))
    ax.set_ylabel("Power (V²/Hz, log scale)", fontsize=viz_params.get('fontsize_grand_label', 12))
    ax.set_title("Grand-Average PSD: Eyes Closed vs Eyes Open", fontsize=viz_params.get('fontsize_grand_title', 14), fontweight='bold')
    ax.legend(loc='upper right', fontsize=viz_params.get('fontsize_legend', 11))
    ax.grid(True, alpha=viz_params.get('grid_alpha', 0.3), which='both')
    ax.set_xlim([1, 100])
    
    # Save figure to both quality_control/overall/ and validation/
    import os
    qc_fig_path = os.path.join(qc_overall_dir, "grand_average_psd_eo_vs_ec.png")
    val_fig_path = os.path.join(validation_root, "grand_average_psd_eo_vs_ec.png")
    
    fig.savefig(qc_fig_path, dpi=viz_params.get('dpi', 300), bbox_inches="tight")
    fig.savefig(val_fig_path, dpi=viz_params.get('dpi', 300), bbox_inches="tight")
    plt.close(fig)
    
    print(f"Grand-average PSD saved: EC={len(ec_psds)} subjects, EO={len(eo_psds)} subjects")
