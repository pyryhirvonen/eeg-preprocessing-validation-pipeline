"""
script to help understand the data structure
EEG - Data pipeline
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""
from pathlib import Path

import mne
from load_data import load_raw
import matplotlib.pyplot as plt
# path of the data
bids_root = Path("/Users/pyryhirvonen/Desktop/Opiskelu/2025-2026/Kandidaatin tutkinto/Discover EEG - Pyry/TD-BRAIN-SAMPLE")

#list the subject folder names from TD-BRAIN-SAMPLE folder and pick one of them
subjects = [d.name for d in bids_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-")]
print("Subjects of TD-BRAIN-SAMPLE:", subjects)
subject= subjects[8]
subject_path = bids_root / subject

#list the session folder names from the subject folder and pick one of them
sessions = [d.name for d in subject_path.iterdir()
            if d.is_dir() and d.name.startswith("ses")]
flipped_sessions=sessions[::-1]
print("sessions of",subject ,flipped_sessions)
session=flipped_sessions[2]
session_path = subject_path / session / "eeg"

#list tasks and choose one
tasks = set()
for f in session_path.iterdir():
    if "_task-" in f.name:
        task = f.name.split("_task-")[1].split("_")[0]
        tasks.add(task)
tasks = list(tasks)
print("Tasks of ",session, tasks)
task=tasks[1]
task_path=session_path/task

datatype="eeg"
#use the "load_raw" function from load_data.py file
raw=load_raw(subject.split("-")[1],session.split("-")[1],task,datatype,bids_root)

#print the sampling rate and number of channels
print("Sampling rate of signal:",raw.info["sfreq"])
print("number of channels:",raw.info["nchan"])

print(f"Channel names: {raw.ch_names}")


montage = mne.channels.make_standard_montage("standard_1020")
#montage.plot(show=True)
#plt.tight_layout()
#plt.show()
raw.set_channel_types({
    "VPVA": "misc",
    "VNVB": "misc",
    "HPHL": "eog",
    "HNHR": "eog",
    "OrbOcc": "eog",
    "Erbs": "emg",
    "Mass": "emg",
})
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, match_case=False,match_alias=False,on_missing='raise')
from pyprep import NoisyChannels
nd = NoisyChannels(
    raw,
    do_detrend=False,
    random_state=42,)
nd.find_bad_by_ransac()
print(f"\n✓ RANSAC analysis complete")
print(f"Bad channels detected: {nd.bad_by_ransac}")
print(f"Number of bad channels: {len(nd.bad_by_ransac)}")

