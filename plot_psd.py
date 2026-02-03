import matplotlib.pyplot as plt


#, fmax=40, color="blue", title="PSD plot"
def plot_psd(dataEC,dataEO,fmax):

def compute_psd()    # compute PSD
    psd = dataEC.compute_psd(fmax=fmax)
    psd_data = psd.get_data(picks="data").mean(axis=0)
    freqs = psd.freqs
    # piirretään matplotlibilla
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, psd_data, color=color)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax.set_title(title)
    ax.grid(True)
    plt.show()

    return psd_data, freqs, fig