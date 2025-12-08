"""
Main file of EEG - Data pipeline
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

# import necessary tools from mne and mne_bids.
import mne
from mne_bids import BIDSPath, read_raw_bids

from load_data import load_raw

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

# copy raw, crop it to 10 sec, plot the data
raw.copy().crop(tmin=0, tmax=10).plot(block=True)
# print information about the object raw
print(raw.info)
