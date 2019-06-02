import numpy as np
import tensorflow as tf
import pandas as pd
import librosa
import pickle
import PySimpleGUI as sg

import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2

from Utilities_GUI import *

class Segmenter:
    def __init__(self):
        """
        Initialize a waveform segmenter
        
        """
        # Load onset NN models
        self.narrow_temp_model = load_NN("../Segmentation/Guitar_onset_300", verbose=False)
        self.broad_temp_model = load_NN("../Segmentation/Guitar_onset_600", verbose=False)
        self.narrow_spectral_model = load_NN("../Segmentation/Guitar_onset_spectral_300", verbose=False)
        self.broad_spectral_model = load_NN("../Segmentation/Guitar_onset_spectral_600", verbose=False)
        
        # Load note dection NN
        self.note_model = load_NN("../Single_note_models/Guitar/Guitar", verbose=False)
        
        # Initialize prediction attributes
        self.preds_narrow_temp = None
        self.preds_broad_temp = None
        self.preds_narrow_spectral = None
        self.preds_broad_spectral = None
        self.preds_env = None
    
    def onset_preds_NN(self, wave, model_name):
        """
        Calculate note onset predictions using a NN model
        
        Args:
            wave (waveform): Waveform object to segment
            model_name (str): Name of the neural network model
            
        Returns:
            Array: Predicted onsets
        
        """
        # Check which model is being asked for and set apply_window parameters accordingly
        if "narrow" in model_name:
            size = 300
        elif "broad" in model_name:
            size = 600
            
        if "spectral" in model_name:
            temp = False
        else:
            temp = True
            
        results = wave.apply_window(size=size, disp=100, function=getattr(self, model_name).predict, convert=True, temp=temp)
        results = np.array(results)
        final = []
        for x in results:
            final.append([x[0], x[1][0][0]])
        return np.array(final)
    
    def find_onset_candidates_env(self, wave, env, k, use_desc=False):
        """
        Find onset candidates using the envelope of the waveform
        
        Args:
            wave (array): Wave
            env (array): Envelope
            k (int): Minimum current bin amplitude to previous bin amplitude to be considered a frontier
            use_desc (bool): Set to True to count amplitude descents as frontiers
        
        Returns:
            Array: Frontier locations

        """
        previous = np.array(env)[:-1]
        current = np.array(env)[1:]
        if use_desc:
            front = np.argwhere(((current >= k*previous) | (previous >= k*current)) & (current > 0.025)).flatten()
        else:
            front = np.argwhere((current >= k*previous) & (current > 0.025)).flatten()

        return front
    
    def onset_preds_env(self, wave, bins=80, k=1.5, size=500, disp=800, use_desc=False):
        """
        Find onset candidates using the envelope of the waveform, note checking and filtering out unusually short notes
        
        Args:
            wave (waveform): Wave
            bins (int): Number of samples per bin
            k (int): Minimum current bin amplitude to previous bin amplitude to be considered a frontier
            size (int): Size of the windows for note checking
            disp (int): Number of samples that the current window will be displaced from the previous one (note checking)
            use_desc (bool): Set to True to count descending amplitude values as onsets
        
        Returns:
            Array: Onsets
            
        """        
        # Find candidates
        env_pos, env_neg = wave.envelope(bins)
        onsets = self.find_onset_candidates_env(wave.y, env_pos, k, use_desc)
        filtered_onsets = []
        final_onsets = []

        # Remove unusually short candidates
        for t, o in zip((onsets[1:]-onsets[:-1]), onsets):
            if t >= 0.25 * np.mean(onsets[1:]-onsets[:-1]):
                filtered_onsets.append(o)
        
        # Check whether there are multiple notes between two candidates
        for i in range(len(filtered_onsets) - 1):
            sg.OneLineProgressMeter('Segmenter', i+1, len(filtered_onsets) - 1, 'key', 'Calculating...', orientation="h")
            chunk = wave.y[filtered_onsets[i]:filtered_onsets[i+1]]
            chunk = waveform(chunk[size:len(chunk)-size])
            predictions = chunk.apply_window(size=size, disp=disp, convert=True, temp=True, 
                                             norm=False, function=self.note_model.predict)

            final_onsets.append(filtered_onsets[i])

            # Check whether the predicted note changes
            for j in range(0, len(predictions) - 1):
                n = predictions[j][0]
                current_note = np.argmax(predictions[j][1][0])
                next_note = np.argmax(predictions[j+1][1][0])

                # ... excluding those too close to the end of the chunk
                if (current_note != next_note) and ((len(chunk.y) - n - size/2) >= (0.05 * len(chunk.y))):
                    final_onsets.append(int(filtered_onsets[i] + n + 3*size/2))

        if len(filtered_onsets) > 0:
            final_onsets.append(filtered_onsets[-1])

        return final_onsets
    
    def middle_check_env(self, chunk, center, std, final_onsets):
        """
        Run onset prediction with a more sensitive envelope
        
        Args:
            chunk (array): Window from a longer waveform
            center (int): Center of the chunk
            std (float): Standard deviation of note durations in the original waveform
            final_onsets (array): Onsets found by the onsets_preds_env method
        
        Returns:
            Float: New onset
            
        """
        new_candidates = np.array(self.onset_preds_env(waveform(chunk), bins=20, use_desc=True))
        return np.mean(new_candidates[round_to_base(new_candidates, std) == center])
    
    def onset_correction_env(self, wave, final_onsets):
        """
        Perform a final check on onset predictions to find missed notes
        
        Args:
            wave (waveform): Audio wave
            final_onsets (array): Onsets found by the onsets_preds_env method
            
        Returns:
            Array: Updated onsets
            
        """
        # Round onset locations to the standard deviation of note durations and find the most common value
        std = int(np.std(np.array(final_onsets[1:]) - np.array(final_onsets[:-1])))
        rounded = np.array([round_to_base(_, std) for _ in (np.array(final_onsets[1:]) - np.array(final_onsets[:-1]))])
        center = np.median(rounded)

        # Check the location between two onsets for a missed onset
        if center != 0:
            new = np.array(wave.apply_window(int(2*center), int(2*center), self.middle_check_env, False, False, False,
                                             center, std, final_onsets))
            new = new[:,0] + new[:,1]
            new = new[~np.isnan(new)]
            rounded_new = round_to_base(new, 2*std)

            for a, b in zip(new, rounded_new):
                if b not in round_to_base(final_onsets, 2*std):
                    final_onsets.append(int(a))

        return final_onsets
    
    def min_dist(self, x, arr):
        """
        Find the minimum distance to the values of an array
        
        Args:
            x (float): Number to be compared
            arr (array): Array to be compared
        
        Returns:
            Float: Minimum distance from x to the array values
            
        """
        arr = np.array(arr)
        return np.min(np.abs(arr - x))
    
    def predict(self, wave):
        """
        Return onset predictions of an audio wave using NN and envelope info
        
        Args:
            wave (waveform): Input audio wave
            
        Returns:
            List: Onset predictions
        """
        # Calculate NN-based predictions
        self.preds_narrow_temp = self.onset_preds_NN(wave, "narrow_temp_model")
        self.preds_broad_temp = self.onset_preds_NN(wave, "broad_temp_model")
        self.preds_narrow_spectral = self.onset_preds_NN(wave, "narrow_spectral_model")
        self.preds_broad_spectral = self.onset_preds_NN(wave, "broad_spectral_model")
        
        # Calculate envelope-based predictions
        self.preds_env = self.onset_correction_env(wave, self.onset_preds_env(wave))
        
        previous = np.array([])
        before_previous = np.array([])
        all_onsets = []
        
        # Iterate through all NN predictions
        for narrow_temp, broad_temp, narrow_spectral, broad_spectral in zip(self.preds_narrow_temp, 
                                                                            self.preds_broad_temp, 
                                                                            self.preds_narrow_spectral,
                                                                            self.preds_broad_spectral):
            preds = np.append(before_previous, previous)
            preds = np.append(preds, np.array([broad_temp[1], 
                                               narrow_temp[1], 
                                               broad_spectral[1], 
                                               narrow_spectral[1]]))
            before_previous = previous
            previous = np.array([broad_temp[1], 
                                 narrow_temp[1], 
                                 broad_spectral[1], 
                                 narrow_spectral[1]])
            
            # Add the onset location if either:
            # 1. There are 2 NN predictions, which may be from the same NN, above 0.75 and an envelope prediction close
            # 2. There are 2 NN predictions, which may NOT be from the same NN, above 0.85
            if (((preds >= 0.75).sum() >= 2) and (self.min_dist(int(narrow_temp[0]+150), self.preds_env) <= 200) or
               (((preds >= 0.85).sum() >= 2) and ((np.argwhere(preds >= 0.85)[1] - np.argwhere(preds >= 0.85)[0]) != 4))):
                all_onsets.append(narrow_temp[0]+150)
                
        final_onsets = []
        coinc = []

        # Filter onsets which are too close
        for i in range(len(all_onsets) - 1):
            if all_onsets[i+1] - all_onsets[i] > 200 or len(coinc) >= 6:
                coinc.append(all_onsets[i])
                final_onsets.append(np.array(coinc).mean())
                coinc = []
            else:
                coinc.append(all_onsets[i])
        final_onsets.append(all_onsets[-1])
            
        return final_onsets

