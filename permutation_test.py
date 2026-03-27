"""
Permutation test module of EEG - Data pipeline.
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy import stats
import mne


def run_permutation_cluster_test(psd_data, params, validation_root, qc_root):
    """
    Orchestrator function: Runs cluster-based permutation test on EO vs EC PSD data.
    
    Workflow:
    1. Extract PSD power values and interpolate to common frequency grid
    2. Compute paired t-statistics (EO - EC) across subjects
    3. Run cluster-based permutation test with MNE
    4. Format and save results CSV
    5. Generate validation figure with significant clusters highlighted
    
    :param psd_data: dict, {subject: {task: {'psd': psd_obj, ...}}}
    :param params: dict, Full configuration from params.json
    :param validation_root: str, Path to validation directory
    :param qc_root: str, Path to quality_control root directory
    :return: None (saves results to validation/ directory)
    """
    print("\n" + "="*60)
    print("Cluster-Based Permutation Test (EO vs EC)")
    print("="*60)
    
    # Create validation directory if not exists
    os.makedirs(validation_root, exist_ok=True)
    
    # Extract permutation test parameters from config
    perm_params = params.get("permutation_test", {})
    freq_min = perm_params.get("freq_min", 1)
    freq_max = perm_params.get("freq_max", 100)
    n_freqs = perm_params.get("n_freqs", 500)
    n_permutations = perm_params.get("n_permutations", 1000)
    seed = perm_params.get("seed", None)
    alpha = perm_params.get("alpha", 0.05)  # Significance level for threshold calculation
    
    # Create common frequency grid
    common_freqs = np.linspace(freq_min, freq_max, n_freqs)
    
    # Loop through subjects, extract and interpolate PSD data
    eo_powers = []
    ec_powers = []
    subjects_list = []
    
    for subject, tasks_dict in psd_data.items():
        eo_psd = tasks_dict.get("restEO", {}).get("psd")
        ec_psd = tasks_dict.get("restEC", {}).get("psd")
        
        if eo_psd and ec_psd:
            # Extract power (average across channels)
            eo_power = extract_and_interpolate(eo_psd, common_freqs)
            ec_power = extract_and_interpolate(ec_psd, common_freqs)
            
            eo_powers.append(eo_power)
            ec_powers.append(ec_power)
            subjects_list.append(subject)
    
    if len(subjects_list) < 2:
        print("⚠ Warning: Fewer than 2 subjects with complete EO/EC data. Skipping permutation test.")
        return
    
    n_subjects = len(subjects_list)
    
    # Convert to arrays
    eo_power_array = np.array(eo_powers)      # shape: (n_subjects, n_freqs)
    ec_power_array = np.array(ec_powers)      # shape: (n_subjects, n_freqs)
    
    print(f"✓ Extracted PSD data from {len(subjects_list)} subjects")
    print(f"  Frequency range: {freq_min}-{freq_max} Hz ({n_freqs} points)")
    print(f"  Threshold: Auto-computed by MNE (α={alpha}, two-tailed)")
    
    # Run permutation cluster test
    T_obs, clusters, cluster_p_values, H0 = run_cluster_permutation(
        eo_power_array, ec_power_array, common_freqs, n_permutations, seed
    )
    
    print(f"✓ Permutation test complete (n_permutations={n_permutations})")
    print(f"  Found {len(clusters)} cluster(s)")
    
    # Format and save results
    results_df = format_cluster_results(
        T_obs, clusters, cluster_p_values, common_freqs, 
        eo_power_array, ec_power_array, alpha=alpha
    )
    
    csv_path = os.path.join(validation_root, "permutation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {validation_root}/permutation_results.csv")
    
    # Generate visualization
    plot_psd_with_clusters(
        psd_data, clusters, cluster_p_values, common_freqs, params, validation_root
    )
    print(f"✓ Validation figure saved to {validation_root}/permutation_test_clusters.png")
    
    # Print summary
    if not results_df.empty:
        significant_clusters = results_df[results_df["significant"] == True]
        if len(significant_clusters) > 0:
            print(f"\n{len(significant_clusters)} significant cluster(s) found (p ≤ {alpha}):")
            for _, row in significant_clusters.iterrows():
                print(f"  Cluster {row['cluster_id']}: {row['freq_min_hz']:.1f}-{row['freq_max_hz']:.1f} Hz, p={row['p_value']:.4f}, {row['direction']}")
        else:
            print(f"\nNo significant clusters found (p ≤ {alpha})")
    else:
        print("\nNo clusters detected in permutation test.")
    
    print("="*60 + "\n")


def extract_and_interpolate(psd_obj, common_freqs):
    """
    Extract mean power from PSD object and interpolate to common frequency grid.
    
    :param psd_obj: mne.time_frequency.Spectrum object
    :param common_freqs: numpy array of target frequencies
    :return: numpy array of interpolated power values
    """
    power = psd_obj.get_data(exclude="bads")
    power_mean = power.mean(axis=0)  # Average across channels
    freqs = psd_obj.freqs
    power_interp = np.interp(common_freqs, freqs, power_mean)
    return power_interp


def build_1d_adjacency(n_freqs):
    """
    Build 1D frequency adjacency matrix for cluster detection.
    Adjacent frequency bins are considered connected.
    
    :param n_freqs: Number of frequency points
    :return: scipy.sparse.csr_matrix of shape (n_freqs, n_freqs)
    """
    # Diagonal adjacency: freq[i] is adjacent to freq[i+1]
    adjacency = csr_matrix(
        (np.ones(n_freqs - 1), (np.arange(n_freqs - 1), np.arange(1, n_freqs))),
        shape=(n_freqs, n_freqs)
    )
    # Make symmetric
    adjacency = adjacency + adjacency.T
    return adjacency


def run_cluster_permutation(eo_power, ec_power, common_freqs, 
                           n_permutations, seed):
    """
    Run cluster-based permutation test on paired EO vs EC differences.
    Uses optimized 1-sample t-test for paired design with automatic threshold.
    
    :param eo_power: numpy array of shape (n_subjects, n_freqs)
    :param ec_power: numpy array of shape (n_subjects, n_freqs)
    :param common_freqs: numpy array of frequency values
    :param n_permutations: Number of permutations
    :param seed: Random seed
    :return: T_obs, clusters, cluster_p_values, H0
    """
    # Compute paired differences (EO - EC for each subject across frequencies)
    differences = eo_power - ec_power  # shape: (n_subjects, n_freqs)
    
    # Build 1D frequency adjacency matrix
    adjacency = build_1d_adjacency(len(common_freqs))
    
    # Run optimized 1-sample permutation cluster test
    # threshold=None allows MNE to compute optimal threshold automatically
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        differences,
        threshold=None,  # MNE auto-computes for p=0.05
        tail=0,  # Two-tailed test
        adjacency=adjacency,
        n_permutations=n_permutations,
        seed=seed
    )
    
    return T_obs, clusters, cluster_p_values, H0


def format_cluster_results(T_obs, clusters, cluster_p_values, common_freqs,
                          eo_power, ec_power, alpha=0.05):
    """
    Format permutation test results into a DataFrame.
    
    :param T_obs: Observed t-statistics, shape (n_freqs,)
    :param clusters: List of cluster indices (frequency arrays)
    :param cluster_p_values: Cluster p-values
    :param common_freqs: Frequency array
    :param eo_power: Eyes open power, shape (n_subjects, n_freqs)
    :param ec_power: Eyes closed power, shape (n_subjects, n_freqs)
    :param alpha: Significance level (default 0.05, read from params)
    :return: pandas DataFrame with columns: cluster_id, freq_min_hz, freq_max_hz, 
             n_freq_points, p_value, direction, t_stat_max, significant
    """
    results = []
    
    for cluster_id, cluster in enumerate(clusters):
        # MNE returns clusters as tuples of arrays, e.g. (array([3, 4, 5]),)
        # Unpack the tuple to get the actual index array
        cluster_indices = cluster[0] if isinstance(cluster, tuple) else cluster
        cluster_indices = np.asarray(cluster_indices, dtype=int)
        
        if len(cluster_indices) == 0:
            continue
        
        # Get frequency range for this cluster
        freq_min = common_freqs[cluster_indices.min()]
        freq_max = common_freqs[cluster_indices.max()]
        n_freq_points = len(cluster_indices)
        p_value = cluster_p_values[cluster_id]
        
        # Get maximum t-statistic in cluster
        t_stat_max = np.max(np.abs(T_obs[cluster_indices]))
        
        # Determine direction: EO > EC (positive t) or EC > EO (negative t)
        mean_t = T_obs[cluster_indices].mean()
        if mean_t > 0:
            direction = "EO > EC"
        else:
            direction = "EC > EO"
        
        # Compute mean power difference in cluster
        mean_diff = (eo_power[:, cluster_indices].mean() - ec_power[:, cluster_indices].mean()).mean()
        
        # Determine significance (p <= alpha; permutation p-values are discrete,
        # so the boundary value is included as significant)
        significant = p_value <= alpha
        
        results.append({
            "cluster_id": cluster_id,
            "freq_min_hz": freq_min,
            "freq_max_hz": freq_max,
            "n_freq_points": n_freq_points,
            "p_value": p_value,
            "direction": direction,
            "t_stat_max": t_stat_max,
            "mean_power_diff": mean_diff,
            "significant": significant
        })
    
    return pd.DataFrame(results)


def plot_psd_with_clusters(psd_data, clusters, cluster_p_values, 
                          common_freqs, params, output_dir):
    """
    Generate validation figure: Grand-average PSD with significant clusters highlighted.
    
    :param psd_data: dict, PSD data from pipeline
    :param clusters: Cluster indices
    :param cluster_p_values: Cluster p-values
    :param common_freqs: Common frequency array
    :param params: Full configuration
    :param output_dir: Output directory
    """
    # Extract visualization parameters
    viz_params = params.get("visualization", {})
    perm_params = params.get("permutation_test", {})
    p_threshold = perm_params.get("alpha", 0.05)
    
    # Collect and interpolate EO/EC PSDs (similar to plot_grand_average_psd)
    eo_psds = []
    ec_psds = []
    
    for _, tasks_dict in psd_data.items():
        for task, data in tasks_dict.items():
            psd = data["psd"]
            psd_array = psd.get_data(exclude="bads")
            psd_mean = np.mean(psd_array, axis=0)
            freqs = psd.freqs
            
            psd_interp = np.interp(common_freqs, freqs, psd_mean)
            
            if "EO" in task or "eo" in task:
                eo_psds.append(psd_interp)
            elif "EC" in task or "ec" in task:
                ec_psds.append(psd_interp)
    
    # Compute grand-averages
    eo_grand_avg = np.mean(eo_psds, axis=0) if eo_psds else None
    ec_grand_avg = np.mean(ec_psds, axis=0) if ec_psds else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=tuple(viz_params.get("figsize_grand", [12, 7])))
    
    # Plot grand-average PSDs
    if ec_grand_avg is not None:
        ax.semilogy(common_freqs, ec_grand_avg, label=f"Eyes Closed (n={len(ec_psds)})",
                   linewidth=viz_params.get("linewidth_grand", 1.5),
                   color=viz_params.get("color_ec", "steelblue"))
    
    if eo_grand_avg is not None:
        ax.semilogy(common_freqs, eo_grand_avg, label=f"Eyes Open (n={len(eo_psds)})",
                   linewidth=viz_params.get("linewidth_grand", 1.5),
                   color=viz_params.get("color_eo", "lightgreen"))
    
    # Highlight alpha band
    ax.axvspan(viz_params.get("alpha_band_min", 8), viz_params.get("alpha_band_max", 12),
              alpha=viz_params.get("alpha_transparency", 0.15),
              color=viz_params.get("color_alpha_band", "gray"),
              label=f"Alpha band ({viz_params.get('alpha_band_min', 8)}-{viz_params.get('alpha_band_max', 12)} Hz)")
    
    # Highlight significant clusters (unpack MNE's tuple format)
    sig_p_values = []
    for _, (cluster, p_value) in enumerate(zip(clusters, cluster_p_values)):
        cluster_indices = cluster[0] if isinstance(cluster, tuple) else cluster
        cluster_indices = np.asarray(cluster_indices, dtype=int)
        if p_value <= p_threshold:
            freq_min = common_freqs[cluster_indices.min()]
            freq_max = common_freqs[cluster_indices.max()]
            ax.axvspan(freq_min, freq_max, alpha=0.2, color="red")
            sig_p_values.append(f"p={p_value:.3f}")
    
    ax.set_xlabel("Frequency (Hz)", fontsize=viz_params.get("fontsize_grand_label", 12))
    ax.set_ylabel("Power (V²/Hz, log scale)", fontsize=viz_params.get("fontsize_grand_label", 12))
    ax.set_title("Permutation Test: EO vs EC (Significant Clusters Highlighted)",
                fontsize=viz_params.get("fontsize_grand_title", 14), fontweight="bold")
    ax.set_xlim([common_freqs.min(), common_freqs.max()])
    
    # Add legend with p-values
    legend_labels = [
        f"Eyes Closed (n={len(ec_psds)})",
        f"Eyes Open (n={len(eo_psds)})",
        f"Alpha band ({viz_params.get('alpha_band_min', 8)}-{viz_params.get('alpha_band_max', 12)} Hz)"
    ]
    if sig_p_values:
        legend_labels.append(f"Significant clusters: {', '.join(sig_p_values)}")
    ax.legend(legend_labels, loc="upper right", fontsize=viz_params.get("fontsize_legend", 11))
    ax.grid(True, alpha=viz_params.get("grid_alpha", 0.3))
    
    # Save figure
    fig_path = os.path.join(output_dir, "permutation_test_clusters.png")
    fig.savefig(fig_path, dpi=viz_params.get("dpi", 100), bbox_inches="tight")
    plt.close(fig)
