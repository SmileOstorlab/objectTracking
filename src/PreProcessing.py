import os
import pandas as pd


def preprocess() -> pd.DataFrame:
    """
    Preprocesses object tracking data by loading and formatting a CSV file.

    This function reads a CSV file containing object tracking data from a specified path, formats the data into a
    pandas DataFrame, and assigns appropriate column names. The path is derived from an environment variable.

    Returns:
       pd.DataFrame: A DataFrame containing the preprocessed object tracking data with columns for frame number, ID,
                     bounding box coordinates ('bb_left', 'bb_top', 'bb_width', 'bb_height'), confidence score ('conf'),
                     and three additional columns ('x', 'y', 'z') representing other attributes.
   """
    base_path = os.environ.get('OBJECT_TRACKING_PATH')
    csv_path = os.path.join(base_path, 'ADL-Rundle-6/det/det.txt')
    df = pd.read_csv(csv_path, sep=',', header=None)
    df.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    return df
