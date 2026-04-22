"""
Data loading module of EEG - Data pipeline:
Bachelor's thesis
"""

import csv

#import necessary tools from mne and mne_bids.
import mne
from mne_bids import BIDSPath, read_raw_bids


def check_channel_positions(bids_path, montage_name="standard_1020"):
    """
    Function checks if channels marked as EEG in channels.tsv are found
    in selected montage and prints warning if some are missing.

    :param bids_path: mne_bids.BIDSPath, path of current recording
    :param montage_name: str, montage name used for channel position check
    :return: None
    """
    channels_path = bids_path.copy().update(suffix="channels", extension=".tsv").fpath
    electrodes_path = bids_path.copy().update(suffix="electrodes", extension=".tsv").fpath
    coordsystem_path = bids_path.copy().update(suffix="coordsystem", extension=".json").fpath

    if not channels_path.exists():
        print(f"⚠ channels.tsv not found: {channels_path}")
        return

    eeg_channels = []
    with open(channels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ch_name = str(row.get("name", "")).strip()
            ch_type = str(row.get("type", "")).strip().upper()
            if ch_name and ch_type == "EEG":
                eeg_channels.append(ch_name)

    montage = mne.channels.make_standard_montage(montage_name)
    missing_from_montage = [ch for ch in eeg_channels if ch not in montage.ch_names]

    if missing_from_montage:
        print(
            f"⚠ {len(missing_from_montage)} channel(s) marked EEG are not in {montage_name}: "
            f"{missing_from_montage}"
        )
    else:
        print(f"✓ All channels marked EEG map to montage {montage_name}.")

    if not electrodes_path.exists() or not coordsystem_path.exists():
        print(
            "⚠ electrodes.tsv/coordsystem.json not fully available for this recording; "
            "position checks rely on montage name matching."
        )


def normalize_channel_types(raw, channel_setup_params=None):
    """
    Function normalizes channel types right after loading raw data.

    :param raw: mne.io.Raw, EEG data and metadata
    :param channel_setup_params: dict | None, params["channel_setup"]
    :return: obj, raw with normalized channel types
    """
    pp = channel_setup_params or {}

    overrides = pp.get("channel_type_overrides", {})
    if overrides:
        valid_overrides = {ch: ch_type for ch, ch_type in overrides.items() if ch in raw.ch_names}
        if valid_overrides:
            raw.set_channel_types(valid_overrides)
            print(f"✓ Applied channel_type_overrides for {len(valid_overrides)} channel(s)")

    if pp.get("set_eeg_not_in_montage_to_misc", True):
        montage_name = pp.get("montage_name", "standard_1020")
        montage = mne.channels.make_standard_montage(montage_name)
        montage_names = {name.lower() for name in montage.ch_names}

        eeg_channels = [
            ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) if ch_type == "eeg"
        ]
        to_misc = [ch for ch in eeg_channels if ch.lower() not in montage_names]

        if to_misc:
            raw.set_channel_types({ch: "misc" for ch in to_misc})
            print(f"✓ Converted EEG channels not in {montage_name} to misc: {to_misc}")

    return raw


#Path to the BIDS dataset root on local computer
def load_raw(subject, session, task, datatype, bids_root, channel_setup_params=None):
    """
    Function takes parameters of data root and subject information. loads the
    data,of assigned parameters, and returns it.

    :param subject: str, id number of subject
    :param session: str, eeg session number of the subject
    :param task: str, rest and eyes closed/open (eo/ec)
    :param datatype: str, eeg
    :param bids_root: str, root of the data
    :param channel_setup_params: dict | None, params["channel_setup"]
    :return: raw: obj, data and the metadata of the chosen session
    """
    # Assigned patient(subject id), session number, task, datatype and the root
    # to bids_path - given as parameters of the function in main.py.
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        datatype=datatype,
        root=bids_root
    )
    # Load the EEG data to "raw"
    raw = read_raw_bids(bids_path,verbose=True)

    pp = channel_setup_params or {}
    montage_name = pp.get("montage_name", "standard_1020")

    # Early sanity check at pipeline start
    check_channel_positions(bids_path, montage_name=montage_name)

    # Load data into memory
    raw.load_data()

    # Normalize channel types at load stage
    raw = normalize_channel_types(raw, channel_setup_params=pp)

    return raw
