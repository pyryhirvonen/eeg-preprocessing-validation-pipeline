"""
Main file of EEG - Data pipeline
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

# import necessary tools from mne and mne_bids.
import mne
from mne.viz.utils import plt_show
from mne_bids import BIDSPath, read_raw_bids

from load_data import load_raw
from preprocess import preprocess_raw

# Path to the BIDS dataset root on local computer
bids_root = "/Users/pyryhirvonen/Desktop/Opiskelu/2025-2026/Kandidaatin tutkinto/Discover EEG - Pyry/TD-BRAIN-SAMPLE"
# choose patient, session... etc
subject="87966293"
session="3"
task="restEC"
datatype="eeg"
# call for load_raw function in load_data.py file,
# loads data from BIDS sample dataset to object: "raw"
raw = load_raw(subject,session,task,datatype,bids_root)
# print information about the object raw
print(raw.info)
# copy raw, crop it to 10 sec, plot the data
raw.copy().crop(tmin=0, tmax=10).plot()

# compute a psd from object raw
fig_raw_psd = raw.compute_psd(fmax=250).plot(
    average=True, amplitude=False, picks="data", exclude="bads",
)
# call for preprocess_raw function in preprocess.py file. filters and filter
# limits are chosen in parametres
raw_clean=preprocess_raw(
    raw,1,40,50,True,True,True)
# copy raw_clean (filtered raw), crop it to 10 sec, plot the data
raw_clean.copy().crop(tmin=0,tmax=10).plot()
# info about raw_clean
print(raw_clean.info)
# compute a psd from clean_raw
fig_clean_raw_psd = raw_clean.compute_psd(fmax=250).plot(
    average=True, amplitude=False, picks="data", exclude="bads",
)

plt_show()
