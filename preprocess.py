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


def preprocess_raw(raw,l_freq,h_freq,notch_freqs,do_bandpass=True,do_notch=True,do_ref=True):
    """
    function takes parameters, and calls for each preprocessing function that
    gets True-bool from main.py
    :param raw: obj, EEG-data and metadata that we are processing
    :param l_freq: int, lower frequency limit
    :param h_freq: int, higher frequency limit
    :param notch_freqs:  int, frequencies where notch filter is applied at.
    :param do_bandpass: bool, If True, apply band-pass filter to the data.
    :param do_notch: bool, If True, apply notch filter to the data.
    :param do_ref: bool, If True, apply average reference to the data.
    :return: obj, preprocessed data
    """
    raw_clean=raw.copy()
    if do_bandpass:
        raw_clean = bandpass_filter(raw_clean,l_freq,h_freq)
    if do_notch:
        raw_clean=notch_filter(raw_clean,notch_freqs)
    if do_ref:
        raw_clean=average_reference(raw_clean)
    return raw_clean


#raw.set_eeg_reference("average", projection=True)
#print(raw.info["projs"])