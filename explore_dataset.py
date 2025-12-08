"""
script to help understand the data structure
EEG - Data pipeline
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""
from pathlib import Path
bids_root = Path("/Users/pyryhirvonen/Desktop/Opiskelu/2025-2026/Kandidaatin tutkinto/Discover EEG - Pyry/TD-BRAIN-SAMPLE")

subjects = [d.name for d in bids_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-")]
print("Subjects of TD-BRAIN-SAMPLE:", subjects)

subject= subjects[8]
subject_path = bids_root / subject
sessions = [d.name for d in subject_path.iterdir()
            if d.is_dir() and d.name.startswith("ses")]
flipped_sessions=sessions[::-1]
print("sessions of",subject ,flipped_sessions)

session=flipped_sessions[2]
session_path = subject_path / session / "eeg"

tasks = set()
for f in session_path.iterdir():
    if "_task-" in f.name:
        task = f.name.split("_task-")[1].split("_")[0]
        tasks.add(task)
tasks = list(tasks)
print("Tasks of ",session, tasks)

