"""
Dataset exploration module of EEG - Data pipeline:
Bachelor's thesis
"""

from pathlib import Path
import json
from load_data import load_raw


def _extract_tasks(session_eeg_path: Path):
    """
    Return sorted unique task names from BIDS EEG filenames.

    :param session_eeg_path: pathlib.Path, Path to session's eeg/ directory
    :return: list[str], Sorted unique BIDS task names
    """
    tasks = set()
    for f in session_eeg_path.glob("*_task-*_eeg.*"):
        if "_task-" in f.name:
            task = f.name.split("_task-")[1].split("_")[0]
            tasks.add(task)
    return sorted(tasks)


def main():
    """
    Explore BIDS dataset contents and run a deterministic raw-data load check.

    Reads pipeline configuration from params.json, lists available
    subject/session/task combinations, and loads the first available
    recording to print basic metadata.

    :return: None
    """
    with open("params.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    bids_root = Path(params.get("bids_root", ""))
    if not bids_root.exists():
        raise FileNotFoundError(
            f"BIDS root not found: {bids_root}. Update bids_root in params.json."
        )

    subjects = sorted(
        d.name for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")
    )
    print(f"BIDS root: {bids_root}")
    print(f"Found {len(subjects)} subject(s)")

    if not subjects:
        print("No subjects found. Check dataset contents.")
        return

    # Explore all subject/session/task combinations quickly
    for subject in subjects:
        subject_path = bids_root / subject
        sessions = sorted(
            d.name for d in subject_path.iterdir() if d.is_dir() and d.name.startswith("ses-")
        )
        print(f"\n{subject}: sessions={sessions}")

        for session in sessions:
            eeg_path = subject_path / session / "eeg"
            if not eeg_path.exists():
                continue
            tasks = _extract_tasks(eeg_path)
            print(f"  {session}: tasks={tasks}")

    # Deterministic load test: first subject, first session, first task
    first_subject = subjects[0]
    first_subject_path = bids_root / first_subject
    first_sessions = sorted(
        d.name
        for d in first_subject_path.iterdir()
        if d.is_dir() and d.name.startswith("ses-")
    )

    if not first_sessions:
        print("No sessions found for first subject. Skipping load test.")
        return

    first_session = first_sessions[0]
    first_eeg = first_subject_path / first_session / "eeg"
    first_tasks = _extract_tasks(first_eeg)

    if not first_tasks:
        print("No tasks found for first subject/session. Skipping load test.")
        return

    first_task = first_tasks[0]
    raw = load_raw(
        subject=first_subject.split("-")[1],
        session=first_session.split("-")[1],
        task=first_task,
        datatype=params.get("datatype", "eeg"),
        bids_root=bids_root,
    )

    print("\nMetadata:")
    print(f"  sampling rate={raw.info['sfreq']} Hz")
    print(f"  channels={raw.info['nchan']}")


if __name__ == "__main__":
    main()

