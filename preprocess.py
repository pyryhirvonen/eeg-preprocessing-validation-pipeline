"""
Data preprocessing modules of EEG - Data pipeline:
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

def bandpass_filter(raw,l_freq,h_freq):
    """
    function takes parameters from preprocess_raw, returns bandpass
    filtered object raw. Filter frequency limits are l_freq and h_freq, they are
    decided in main.py
    :param raw: obj, EEG-data and metadata that we are processing
    :param l_freq: int, lower frequency limit
    :param h_freq: int, higher frequency limit
    :return: obj, filtered raw
    """
    return raw.filter(l_freq,h_freq)

def notch_filter(raw,freqs):
    """
    function takes parameters from preprocess_raw, returns notch
    filtered object raw. Filter frequencies are in "freqs", decided in main.py
    :param raw: obj, EEG-data and metadata that we are processing
    :param freqs: int, frequencies where notch filter is applied at.
    :return: obj, filtered raw
    """
    return raw.notch_filter(freqs)

def average_reference(raw):
    """
    function takes raw from preprocess_raw, returns object raw with new
    average reference.
    :param raw: obj, EEG-data and metadata that we are processing
    :return: object raw with new average reference.
    """
    return raw.set_eeg_reference(ref_channels="average")


def preprocess_raw(raw,params):
    """
    Orchestrates all preprocessing steps.
    Reads all parameters from params['preprocessing'].
    
    :param raw: mne.io.Raw, Raw EEG data
    :param params: dict, Full params.json config
    :return: mne.io.Raw, Preprocessed data
    """
    pp = params.get("preprocessing", {})
    
    # Extract all parameters (defaults matching DISCOVER-EEG)
    notch_freq = pp.get("notch_freq", 50)
    hp_cutoff = pp.get("hp_cutoff", 0.5)
    lp_cutoff = pp.get("lp_cutoff", 100)
    do_bandpass = pp.get("do_bandpass", True)
    do_notch = pp.get("do_notch", True)
    do_ref = pp.get("do_ref", True)
    do_bads_detection = pp.get("do_bads_detection", False)
    
    raw_clean = raw.copy()
    bads = []

    # Bad channel and segment detection
    if do_bads_detection:
        bads = detect_bad_channels(raw_clean, pp)
    #bad_segments = detect_bad_segments(raw_clean, pp)
    
    # Apply preprocessing steps

    
    if do_notch:
        raw_clean = notch_filter(raw_clean, notch_freq)
    
    if do_bandpass:
        raw_clean = bandpass_filter(raw_clean, hp_cutoff, lp_cutoff)
    
    
    if do_ref:
        raw_clean = average_reference(raw_clean)
    

    return raw_clean, bads#, bad_segments


def detect_bad_channels(raw_clean,params):
    """
    function detects bad channels based on the parameters given in params.json
    :param raw: obj, EEG-data and metadata that we are processing
    :param params: dict, parameters from params.json
    :return: list, bad channel names
    """
    from mne.preprocessing import annotate_amplitude
    annotations,bads=annotate_amplitude(raw_clean,peak=params['bad_channel_peak'],
                                        flat=params['bad_channel_flat'],
                                        min_duration=params['bad_channel_min_duration'],
                                        bad_percent=params['bad_channel_bad_percent'],)
    raw_clean.set_annotations(annotations)
    if bads:
        raw_clean.info['bads'].extend(bads)
        print(f"  Bad channels detected: {len(bads)} - {bads}")
    else:
        print(f"  No bad channels detected")
    
    return bads

