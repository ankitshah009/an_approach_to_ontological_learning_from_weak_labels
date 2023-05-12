import os

import librosa
import numpy as np
import pandas as pd

# DATA_DIR = 'data/msos'

# LABEL_NAME_MAP_L1_DF = pd.read_csv('data/msos/label_name_map_l1.csv')
# LABEL_NAME_MAP_L1 = dict(zip(LABEL_NAME_MAP_L1_DF['label'],
#                              LABEL_NAME_MAP_L1_DF['name']))
# LABEL_NAME_MAP_L2_DF = pd.read_csv('data/msos/label_name_map_l2.csv')
# LABEL_NAME_MAP_L2 = dict(zip(LABEL_NAME_MAP_L2_DF['label'],
#                              LABEL_NAME_MAP_L2_DF['name']))


def generate_label_name_map(filepath):
    logsheet_df = pd.read_csv(filepath)
    label_name_map_l1 = dict(zip(range(len(logsheet_df['Event'].unique())),
                                 logsheet_df['Event'].unique()))
    label_name_map_l2 = dict(zip(range(len(logsheet_df['Category'].unique())),
                                 logsheet_df['Category'].unique()))
    name_label_map_l1 = dict(zip(logsheet_df['Event'].unique(),
                                 range(len(logsheet_df['Event'].unique()))))
    name_label_map_l2 = dict(zip(logsheet_df['Category'].unique(),
                                 range(len(logsheet_df['Category'].unique()))))
    logsheet_df['label_l1'] = logsheet_df['Event']\
        .map(lambda c: name_label_map_l1[c])
    logsheet_df['label_l2'] = logsheet_df['Category']\
        .map(lambda c: name_label_map_l2[c])
    return logsheet_df, label_name_map_l1, label_name_map_l2


def mel_spectro_msos(logsheet, data_dir, eval=False):
    """
    Load all wav files under the folder and make mel_spectrograms with their labels
    """
    data = []
    for i in range(len(logsheet)):
        data_path = os.path.join(data_dir, 'Evaluation'
                                 if eval else 'Development')
        if not eval:
            data_path = os.path.join(data_path, logsheet.loc[i]['Category'])
        data_path = os.path.join(data_path, logsheet.loc[i]['File'])
        # Use original sampling rate
        loaded, sr = librosa.load(data_path, sr=None)
        mel_spectro = librosa.feature.melspectrogram(y=loaded,
                                                     sr=sr,
                                                     n_fft=1024,
                                                     n_mels=128)
        data.append(mel_spectro.T)

    return np.array(data), logsheet[['label_l1', 'label_l2']].to_numpy()
