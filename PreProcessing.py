import pandas as pd


def preprocess() -> pd.DataFrame:
    df = pd.read_csv('ADL-Rundle-6/det/det.txt', sep=',', header=None)
    df.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    return df
