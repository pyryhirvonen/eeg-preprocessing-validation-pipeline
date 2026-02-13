"""
Data loading module of EEG - Data pipeline:
Bachelor's thesis
Author: Pyry Hirvonen
Student number: 152165990
Mail: pyry.hirvonen@tuni.fi
"""

#import necessary tools from mne and mne_bids.
import mne
from mne_bids import BIDSPath, read_raw_bids
#Path to the BIDS dataset root on local computer
def load_raw(subject, session, task, datatype, bids_root):
    """
    Function takes parameters of data root and subject information. loads the
    data,of assigned parameters, and returns it.

    :param subject: str, id number of subject
    :param session: str, eeg session number of the subject
    :param task: str, rest and eyes closed/open (eo/ec)
    :param datatype: str, eeg
    :param bids_root: str, root of the data
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
    raw = read_raw_bids(bids_path,verbose=False)

    # Load data into memory
    raw.load_data()
    return raw
