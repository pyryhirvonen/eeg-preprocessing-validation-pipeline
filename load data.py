"""
Bachelor's thesis
EEG - Data pipeline
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

#import necessary tools from mne and mne_bids.
from mne_bids import BIDSPath, read_raw_bids
#Path to the BIDS dataset root on local computer
bids_root = "/Users/pyryhirvonen/Desktop/Opiskelu/2025-2026/Kandidaatin tutkinto/Discover EEG - Pyry/TD-BRAIN-SAMPLE"

#Assign chosen patient(subject id), session number, task, datatype and the root
#to bids_path
bids_path = BIDSPath(
    subject="87966293",
    session="3",
    task="restEC",
    datatype="eeg",
    root=bids_root
)

#Load the EEG data to "raw"
raw = read_raw_bids(bids_path)

#Plot the data
raw.plot(block=True)
