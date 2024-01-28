import pandas as pd
import os


def preprocess() -> pd.DataFrame:
    base_path = os.environ.get('OBJECT_TRACKING_PATH')
    csv_path = os.path.join(base_path, 'ADL-Rundle-6/det/det.txt')
    df = pd.read_csv(csv_path, sep=',', header=None)
    df.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    return df
