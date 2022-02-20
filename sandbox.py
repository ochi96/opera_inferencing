import os
from joblib import load, dump
import subprocess
import librosa
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras


audio_main_dir = "E:\Projects\Freelancing\Opera audio files\opera_inferencing/Data/serving_data/output_dir/Wagnerian Contralto Kerstin Thorborg Sings _So ist es denn,_ from Die Walküre, Act II.  1940-sFGGaEr08-M"
model_dir = r"E:\Projects\Freelancing\Opera audio files\opera_inferencing/Models/Females_B/ANN/Females_B"

# # subprocess.call([r"splitting.bat", audio_main_dir, desired_time, segmented_file_name, dir_name])
# def create_mfccs():
#     def extract_features(file_path):
#         audio, sample_rate = librosa.load(file_path) 
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
#         mfccs_processed = np.mean(mfccs.T,axis=0)
#         mfccs_processed = list(mfccs_processed)
#         return mfccs_processed

#     file_paths = ['{0}/{1}'.format(audio_main_dir, file_item) for file_item in os.listdir(audio_main_dir)]
#     mfcc_features = list(map(extract_features, file_paths))
#     random.shuffle(mfcc_features)
#     return mfcc_features

# mfccs_features = create_mfccs()

# with 
print('lol0')
model = tf.keras.models.load_model("./Females_A")
print('lol')