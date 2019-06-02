import numpy as np
import tensorflow as tf
import pandas as pd
import librosa
import pickle
import guitarpro

import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2

from Utilities_GUI import *
from Segmenter_GUI import *

class sheet_generator:
    def __init__(self, note_model, string_model, segmentation_model=Segmenter(), bpm=None):
        """
        Create a sheet_generator object
        
        Args:
            note_model (str): Name of the NN model used for note identification
            string_model (str): Name of the NN model sued for string identification
            segmentation_model (object): Segmentation model
            bpm (int): Beats per minute
            
        """
        # Load prediction models
        self.note_model = load_NN(note_model, verbose=0)
        self.string_model = load_NN(string_model, verbose=0)
        self.segmentation_model = segmentation_model
        
        # Set bpm
        self.bpm = bpm
        
        # Create string-notes correspondence
        self.note_table = pd.read_csv("../Data/Piano/Note_table.tsv", header=0, sep="\t")
        self.start_8 = self.note_table[self.note_table["Note"] == "E1"].index[0]
        self.start_7 = self.note_table[self.note_table["Note"] == "B1"].index[0]
        self.start_6 = self.note_table[self.note_table["Note"] == "E2"].index[0]
        self.start_5 = self.note_table[self.note_table["Note"] == "A2"].index[0]
        self.start_4 = self.note_table[self.note_table["Note"] == "D3"].index[0]
        self.start_3 = self.note_table[self.note_table["Note"] == "G3"].index[0]
        self.start_2 = self.note_table[self.note_table["Note"] == "B3"].index[0]
        self.start_1 = self.note_table[self.note_table["Note"] == "E4"].index[0]
        self.string_notes = {
            "8": self.note_table["Note"].iloc[self.start_8:self.start_8+25].values,
            "7": self.note_table["Note"].iloc[self.start_7:self.start_7+25].values,
            "6": self.note_table["Note"].iloc[self.start_6:self.start_6+25].values,
            "5": self.note_table["Note"].iloc[self.start_5:self.start_5+25].values,
            "4": self.note_table["Note"].iloc[self.start_4:self.start_4+25].values,
            "3": self.note_table["Note"].iloc[self.start_3:self.start_3+25].values,
            "2": self.note_table["Note"].iloc[self.start_2:self.start_2+25].values,
            "1": self.note_table["Note"].iloc[self.start_1:self.start_1+25].values
        }
    
    def note_extraction(self, wave):
        """
        Extract all notes from wave
        
        Args:
            wave (waveform): Input audio wave
            
        Returns:
            List: note predictions, with string, duration and fret info
            
        """
        onsets = list(map(int, self.segmentation_model.predict(wave)))
        
        # For now we add the end of file as an extra onset
        onsets.append(len(wave.y))
        results = []
        
        for i in range(len(onsets) - 1):
            sg.OneLineProgressMeter('Sheet generator', i+1, len(onsets) - 1, 'key', 'Calculating...', orientation="h")
            note = waveform(wave.y[onsets[i]:onsets[i+1]])
            
            if len(note.y) >= 500:
                # Predict note (chroma and octave)
                note_all_preds = self.note_model.predict(note.temp_data(500))
                note_confidence = np.max(note_all_preds)
                note_pred = librosa.midi_to_note(np.argmax(note_all_preds) + 28)

                # Predict string
                string_all_preds = self.string_model.predict(note.spectral_data(500, True))[0]
                string_confidence = np.max(string_all_preds)
                string_pred = np.argmax(string_all_preds) + 1
                
                # Calculate duration
                duration = len(note.y) / 8000
                
                while True:
                    try:
                        # Find corresponding fret
                        fret = self.find_fret(note_pred, string_pred)
                        break
                    except:
                        # Some string-note combinations are not possible
                        # This is due to an error in either note or string prediction (or both)
                        
                        # Note prediction error -> Pick next most likely note
                        if note_confidence < 0.1:
                            note_all_preds[note_all_preds == np.max(note_all_preds)] = 0
                            note_pred = librosa.midi_to_note(np.argmax(note_all_preds) + 28)
                            
                        # String prediction error -> Pick next most likely string
                        else:
                            string_all_preds[string_all_preds == np.max(string_all_preds)] = 0
                            string_pred = np.argmax(string_all_preds) + 1
                            
                results.append([note_pred, string_pred, duration, fret, string_confidence, string_all_preds])
        
        return results
    
    def find_fret(self, note, string):
        """
        Find corresponding fret given a note and a string
        
        Args:
            note (str): Note (chroma and octave)
            string (int): String number
            
        Returns:
            Int: Fret
        """
        return np.argwhere(self.string_notes[str(string)] == note)[0][0]
    
    def bpm_estimation(self, extracted_notes):
        """
        Estimate the bpm of an audio file given the extracted notes
        
        Args:
            extracted_notes (list): Note information extracted by the note_extraction method
            
        Returns:
            Int: bpm
            
        """
        extracted_notes = np.array(extracted_notes)
        return round_to_base(1/(2 * np.median(extracted_notes[:,2].astype(float))) * 60, 10)
    
    def raw_sheet(self, extracted_notes, apply_heuristics):
        """
        Generate a raw music sheet
        
        Args:
            extracted_notes (list): Note information extracted by the note_extraction method
            apply_heuristics (bool): Set to True to use heuristics to correct possible mistakes
            
        Returns:
            Array: Output sheet
            
        """
        extracted_notes = np.array(extracted_notes)
        
        if self.bpm is None:
            self.bpm = self.bpm_estimation(extracted_notes)
            
        # Round durations to sixteenth note duration
        extracted_notes[:,2] = round_to_base(extracted_notes[:,2].astype(float), 1/self.bpm * 60/4)
        extracted_notes = extracted_notes[extracted_notes[:,2].astype("float") > 0]
        
        if apply_heuristics:
            for i in range(extracted_notes.shape[0]):
                if i != 0:
                    # Heuristic 1 -> Change string if note is close to previous but fret is not
                    h1_cond1 = abs(librosa.note_to_midi(extracted_notes[i-1,0])-librosa.note_to_midi(extracted_notes[i,0])) < 4
                    h1_cond2 = abs(extracted_notes[i-1,3] - extracted_notes[i,3]) >= 5
                    if h1_cond1 and h1_cond2:
                        extracted_notes[i,1] = extracted_notes[i-1,1]
                        extracted_notes[i,3] = self.find_fret(extracted_notes[i,0], extracted_notes[i,1])
                
                # Heuristic 2 -> Change string if fret is far from previous and prediction confidence is not high
                h2_cond1 = abs(extracted_notes[i,3] - extracted_notes[i-1,3]) > 5
                h2_cond2 = extracted_notes[i,4] < 0.8
                if h2_cond1 and h2_cond2:
                    string_all_preds = extracted_notes[i,5]
                    string_all_preds[string_all_preds == np.max(string_all_preds)] = 0
                    extracted_notes[i,1] = np.argmax(string_all_preds) + 1
                    extracted_notes[i,3] = self.find_fret(extracted_notes[i,0], extracted_notes[i,1])
        
        return extracted_notes
    
    def display_tablature(self, sheet):
        """
        Generate a printable tab
        
        Args:
            sheet (array): Music sheet created by the raw_sheet method
            
        Returns:
            Str: Printable tab
            
        """
        tab = ""
        for string in range(1, 7):
            tab += self.string_notes[str(string)][0][0] + "  "
            for note in sheet:
                if int(note[1]) == string:
                    tab += str(note[3])
                else:
                    tab += "-" * len(str(note[3]))
                tab += "-"
            tab += "\n"
        return tab
        
    def create_guitarpro_tab(self, sheet, file_name):
        """
        Generate a guitarpro file from a music sheet
        
        Args:
            sheet (array): Music sheet created by the raw_sheet method
            file_name (str): Name of the output file
            
        Returns:
            None
            
        """
        # Create a blank gp5 structure from a template
        template = guitarpro.parse('../templates/blank.gp5')
        measure_list = template.tracks[0].measures
        del template.tracks[0].measures[0].voices[0].beats[0]
        bar_start = 0
        current_measure = 0
        
        for i in range(len(sheet)):
            # Check if previous notes and current fit in the same bar
            if (self.bpm/60) * (sheet[bar_start:i+1,2].astype("float").sum()) <= 4:
                duration = guitarpro.Duration(value= int(round(4/((self.bpm/60) * (float(sheet[i,2]))))))
                new_beat = guitarpro.Beat(template.tracks[0].measures[0].voices[0], duration=duration)
                new_note = guitarpro.Note(new_beat, value=int(sheet[i,3]), string=int(sheet[i,1]), 
                                          type=guitarpro.NoteType.normal)
                new_beat.notes.append(new_note)
                template.tracks[0].measures[current_measure].voices[0].beats.append(new_beat)
            else:
                # Check if previous notes fill up the bar entirely or if there's space left
                if (self.bpm/60) * (sheet[bar_start:i,2].astype("float").sum()) == 4:
                    current_measure += 1
                    new_measure = guitarpro.Measure(template.tracks[0], header=template.tracks[0].measures[0].header)
                    duration = guitarpro.Duration(value= int(round(4/((self.bpm/60) * (float(sheet[i,2]))))))
                    new_beat = guitarpro.Beat(template.tracks[0].measures[0].voices[0], duration=duration)
                    new_note = guitarpro.Note(new_beat, value=int(sheet[i,3]), string=int(sheet[i,1]), 
                                              type=guitarpro.NoteType.normal)
                    new_beat.notes.append(new_note)
                    new_measure.voices[0].beats.append(new_beat)
                    measure_list.append(new_measure)
                    bar_start = i
                else:
                    # Split the new note into beats that fit the bar and the remaining
                    fitting_beats = 4 - (self.bpm/60) * (sheet[bar_start:i,2].astype("float").sum())
                    duration = guitarpro.Duration(value= int(round(4/fitting_beats)))
                    new_beat = guitarpro.Beat(template.tracks[0].measures[0].voices[0], duration=duration)
                    new_note = guitarpro.Note(new_beat, value=int(sheet[i,3]), string=int(sheet[i,1]), 
                                              type=guitarpro.NoteType.normal)
                    new_beat.notes.append(new_note)
                    template.tracks[0].measures[current_measure].voices[0].beats.append(new_beat)
                    
                    remaining_beats = (self.bpm/60) * (float(sheet[i,2])) - fitting_beats
                    current_measure += 1
                    new_measure = guitarpro.Measure(template.tracks[0], header=template.tracks[0].measures[0].header)
                    duration = guitarpro.Duration(value= int(round(4/remaining_beats)))
                    new_beat = guitarpro.Beat(template.tracks[0].measures[0].voices[0], duration=duration)
                    new_note = guitarpro.Note(new_beat, value=int(sheet[i,3]), string=int(sheet[i,1]), 
                                              type=guitarpro.NoteType.tie)
                    new_beat.notes.append(new_note)
                    new_measure.voices[0].beats.append(new_beat)
                    measure_list.append(new_measure)
                    bar_start = i
        
        guitarpro.write(template, file_name)