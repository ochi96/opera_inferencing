# import libraries
import os
import logging
import subprocess

import numpy as np
import pandas as pd

from ast import literal_eval
from joblib import load, dump
from pathlib import Path

import sys
import time
import shutil

# %matplotlib inline
import urllib.request as urllib2 # For python3
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import librosa
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

# import pysoundfile
import ffmpeg
import spleeter
from PIL import Image

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG, filename="info.log", filemode='w')

class Config():
    '''class that loads the initial configuration'''
    def __init__(self, audio_file, artist_gender):
        self.base_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        self.spleeting_dir = self.base_dir + 'Data/serving_data/'
        self.audio_file = self.spleeting_dir + audio_file #will change this during production
        self.audio_file_name = os.path.basename(self.audio_file)
        self.audio_dir_name = self.audio_file_name.rsplit('.', 1)[0]
        self.artist_gender = artist_gender

        self.mfccs_file = self.base_dir + "Csv_Files/males_mfccs.csv"
        self.label_target = load(self.base_dir +  'Data/label_target/males_label_target/males_label_target.joblib')
        self.model_names = ['Males_A', 'Males_B']
        self.selected_models = [dict({'pca_model_path': "Models/{0}/PCA/{0}_pca.joblib".format(model_name),
                                'scaler_model_path': "Models/{0}/Scaler/{0}_scaler.joblib".format(model_name),
                                'label_mapping_path': "Models/{0}/Reports/{0}_label_mapping.joblib".format(model_name),
                                'keras_model_dir_path': "Models/{0}/ANN/{0}".format(model_name)}) for model_name in self.model_names]

        if self.artist_gender=='female':
            self.mfccs_file = self.base_dir + "Csv_Files/females_mfccs.csv"
            self.label_target = load(self.base_dir + 'Data/label_target/females_label_target/females_label_target.joblib')
            self.model_names = ['Females_A', 'Females_B']
            self.selected_models = [dict({'pca_model_path': "Models/{0}/PCA/{0}_pca.joblib".format(model_name),
                                'scaler_model_path': "Models/{0}/Scaler/{0}_scaler.joblib".format(model_name),
                                'label_mapping_path': "Models/{0}/Reports/{0}_label_mapping.joblib".format(model_name),
                                'keras_model_dir_path': "Models/{0}/ANN/{0}".format(model_name)}) for model_name in self.model_names]

    def check_paths(self):
        all_paths = [self.base_dir + model[path] for model in self.selected_models for path in model]
        all_paths.extend([self.spleeting_dir, self.mfccs_file, self.audio_file])
        for path in all_paths:
            if os.path.exists(path)==False:
                logging.debug("Path '{}' does not exist. Do not proceed".format(path))
                break
            else:
                logging.debug("Path '{}' exists: OK".format(path))
        pass


class Preprocessing(Config):

    def __init__(self):
        Config.__init__(self, audio_file = audio_file, artist_gender=gender)#figure out how to pass the instatiated config file here.
    
    def spleet_audio_files(self):
        #copy to the speeting directory
        logging.debug("Moving file to spleeting directory...")
        shutil.copy2(self.audio_file, self.spleeting_dir)
        logging.debug("Spleeting files into components, results in output_dir")
        self.audio_main_dir =  '{0}output_dir/{1}'.format(self.spleeting_dir, self.audio_dir_name)
        subprocess.call([r'spleeting.bat', self.spleeting_dir, self.audio_file, self.audio_file_name, self.audio_dir_name, self.audio_main_dir])
        pass

    def remove_silence(self):
        logging.debug("Removing periods of silence from sleeted audio file...")
        subprocess.call([r'silence_remove.bat', self.audio_main_dir, self.audio_file_name])
        pass

    def split_into_six_seconds(self, desired_time):
        logging.debug("spliting the 'silence removed' file into segments of {} seconds each".format(desired_time))
        segment_duration = r"{}".format(str(desired_time))
        segmented_file_name = r"%03d_.wav"
        subprocess.call([r'splitting.bat', self.audio_main_dir, segment_duration, segmented_file_name, self.audio_dir_name])
        for item in os.listdir(self.audio_main_dir):
            segment_file = '{0}/{1}'.format(self.audio_main_dir, item)
            audio_length = librosa.get_duration(filename=segment_file)
            if audio_length<(desired_time-0.02):
                os.remove(segment_file)
                print("Removed {} because audio length of {} is less than {}". format(item, audio_length, desired_time))
        pass

    def create_mfccs(self):

        def extract_features(file_path):
            audio, sample_rate = librosa.load(file_path) 
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
            mfccs_processed = np.mean(mfccs.T,axis=0)
            mfccs_processed = list(mfccs_processed)
            return mfccs_processed

        file_paths = ['{0}/{1}'.format(self.audio_main_dir, file_item) for file_item in os.listdir(audio_main_dir)]
        mfcc_features = list(map(extract_features, file_paths))
        random.shuffle(mfcc_features)

        return mfcc_features


class ModelInferencing(Preprocessing):

    def __init__(self):
        Preprocessing.__init__(self)

    def load_models(self):
        print(self.selected_models)
        print(self.base_dir)
        pca_model_A, pca_model_B = [load(self.base_dir + model['pca_model_path']) for model in self.selected_models]
        scaler_model_A, scaler_model_B = [load(self.base_dir + model['scaler_model_path']) for model in self.selected_models]
        model_A, model_B = [keras.models.load_model(self.base_dir + model['keras_model_dir_path']) for model in self.selected_models]
        model_A_mapping, model_B_mapping = [load(self.base_dir + model['label_mapping_path']) for model in self.selected_models]
        print('completed')

    # scaled_features_model_A, scaled_features_model_B = scaled_features

    
        # np.array(mfcc_features).shape

audio_file = 'testing_audio/Kerstin Thorborg/Wagnerian Contralto Kerstin Thorborg Sings _So ist es denn,_ from Die Walküre, Act II.  1940-sFGGaEr08-M.wav'
gender = 'female'
config = Config(audio_file = audio_file, artist_gender = gender)
config.check_paths()

prep = Preprocessing()
# prep.spleet_audio_files()
# prep.remove_silence()
# prep.split_into_six_seconds(6)
# mfccs = prep.create_mfccs

lol = ModelInferencing()
lol.load_models()





