"""
PSD visualization and analysis module.
Per-subject QC plot and grand-average EO vs EC PSD comparison.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt

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


def plot_subject_psd_qc(psd, subject, output_dir, viz_params, suffix="", task=""):
    """
    Function plots per-subject PSD (QC figure) with linear y-axis, averaging across both EO and EC tasks.
    
    :param psd: mne.time_frequency.Spectrum, Computed PSD object
    :param subject: str, Subject ID
    :param output_dir: str, Output directory for figure
    :param viz_params: dict, Visualization parameters from params.json
    :param suffix: str, Optional suffix for filename (default: "")
    :param task: str, Task name for filename (default: "")
    """
    fig, ax = plt.subplots(figsize=tuple(viz_params['figsize_qc']))
    
    # Extract PSD data
    psd_array = psd.get_data(exclude="bads")
    freqs = psd.freqs
    
    # Determine color based on task
    if "EO" in task or "eo" in task:
        line_color = viz_params['color_eo']
        task_label = "Eyes Open"
    elif "EC" in task or "ec" in task:
        line_color = viz_params['color_ec']
        task_label = "Eyes Closed"
    else:
        line_color = viz_params['color_qc']
        task_label = task
    
    # Plot mean PSD across channels (linear scale)
    psd_mean = np.mean(psd_array, axis=0)
    ax.plot(freqs, psd_mean, linewidth=viz_params['linewidth_qc'], color=line_color)
    
    # Highlight alpha band
    ax.axvspan(viz_params['alpha_band_min'], viz_params['alpha_band_max'], 
               alpha=viz_params['alpha_transparency'], color=viz_params['color_alpha_band'], 
               label=f"Alpha band ({viz_params['alpha_band_min']}-{viz_params['alpha_band_max']} Hz)")
    
    task_str = f" - {task_label}" if task else ""
    ax.set_title(f"PSD QC: Subject {subject}{task_str}", fontsize=viz_params['fontsize_title'], fontweight='bold')
    ax.set_xlabel("Frequency (Hz)", fontsize=viz_params['fontsize_label'])
    ax.set_ylabel("Power (V²/Hz)", fontsize=viz_params['fontsize_label'])
    ax.set_xlim([1, 100])
    ax.grid(True, alpha=viz_params['grid_alpha'])
    ax.legend()
    
    # Save figure
    task_str = f"_task-{task}" if task else ""
    fig.savefig(f"{output_dir}/sub-{subject}{task_str}_psd_qc{suffix}.png", dpi=viz_params['dpi'], bbox_inches="tight")
    plt.close(fig)

def plot_grand_average_psd(psd_data, output_dir, viz_params):
    """
    Function plots grand-average PSD across all subjects for EO vs EC comparison with log scale y-axis and EO/EC overlaid.
    
    :param psd_data: dict, {subject: {task: {'psd': psd_obj, 'raw': raw_obj}}}
    :param output_dir: str, Output directory for figure
    :param viz_params: dict, Visualization parameters from params.json
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
    fig, ax = plt.subplots(figsize=tuple(viz_params['figsize_grand']))
    
    if ec_grand_avg is not None and ec_freqs is not None:
        ax.semilogy(ec_freqs, ec_grand_avg, label=f"Eyes Closed (n={len(ec_psds)})", 
                    linewidth=viz_params['linewidth_grand'], color=viz_params['color_ec'])
    if eo_grand_avg is not None and eo_freqs is not None:
        ax.semilogy(eo_freqs, eo_grand_avg, label=f"Eyes Open (n={len(eo_psds)})", 
                    linewidth=viz_params['linewidth_grand'], color=viz_params['color_eo'])
    
    # Highlight alpha band
    ax.axvspan(viz_params['alpha_band_min'], viz_params['alpha_band_max'], 
               alpha=viz_params['alpha_transparency'], color=viz_params['color_alpha_band'], 
               label=f"Alpha band ({viz_params['alpha_band_min']}-{viz_params['alpha_band_max']} Hz)")
    
    ax.set_xlabel("Frequency (Hz)", fontsize=viz_params['fontsize_grand_label'])
    ax.set_ylabel("Power (V²/Hz, log scale)", fontsize=viz_params['fontsize_grand_label'])
    ax.set_title("Grand-Average PSD: Eyes Closed vs Eyes Open", fontsize=viz_params['fontsize_grand_title'], fontweight='bold')
    ax.legend(loc='upper right', fontsize=viz_params['fontsize_legend'])
    ax.grid(True, alpha=viz_params['grid_alpha'], which='both')
    ax.set_xlim([1, 100])
    
    # Save figure
    fig.savefig(f"{output_dir}/grand_average_psd_eo_vs_ec.png", dpi=viz_params['dpi'], bbox_inches="tight")
    plt.close(fig)
    
    print(f"Grand-average PSD saved: EC={len(ec_psds)} subjects, EO={len(eo_psds)} subjects")
