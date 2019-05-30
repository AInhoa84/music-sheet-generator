import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import librosa
from tqdm import tqdm
import pickle
import IPython.display as ipd
from IPython.core.display import display, HTML, Javascript
import music21
import json, random
import guitarpro

import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from livelossplot import PlotLossesKeras

from Utilities import *
from Segmenter import *

class sheet_generator:
    def __init__(self, note_model, string_model, segmentation_model=Segmenter(), bpm=None):
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
    
    def note_extraction(self, wave, show_plots):
        onsets = list(map(int, self.segmentation_model.predict(wave, show_plots)))
        
        # For now we add the end of file as an extra onset
        onsets.append(len(wave.y))
        results = []
        
        for i in tqdm(range(len(onsets) - 1), leave=False):
            note = waveform(wave.y[onsets[i]:onsets[i+1]])
            
            if len(note.y) >= 500:
                note_all_preds = self.note_model.predict(note.temp_data(500))
                note_confidence = np.max(note_all_preds)
                note_pred = librosa.midi_to_note(np.argmax(note_all_preds) + 28)

                # Run the string prediction over chunks of the entire note to get a more accurate prediction
                #string_predictions = np.array(note.apply_window(500, 100, self.string_model.predict, True, False, True))[:,1]
                #string_pred = int(np.median(np.array([np.argmax(x) for x in string_predictions])) + 1)
                string_all_preds = self.string_model.predict(note.spectral_data(500, True))[0]
                string_confidence = np.max(string_all_preds)
                string_pred = np.argmax(string_all_preds) + 1
                duration = len(note.y) / 8000
                while True:
                    try:
                        print(note_pred, string_pred, note_confidence)
                        fret = self.find_fret(note_pred, string_pred)
                        break
                    except:
                        if note_confidence < 0.1:
                            note_all_preds[note_all_preds == np.max(note_all_preds)] = 0
                            note_pred = librosa.midi_to_note(np.argmax(note_all_preds) + 28)
                            print(note_pred)
                        else:
                            string_all_preds[string_all_preds == np.max(string_all_preds)] = 0
                            string_pred = np.argmax(string_all_preds) + 1
                results.append([note_pred, string_pred, duration, fret, string_confidence, string_all_preds])
        
        return results
    
    def find_fret(self, note, string):
        return np.argwhere(self.string_notes[str(string)] == note)[0][0]
    
    def bpm_estimation(self, extracted_notes):
        extracted_notes = np.array(extracted_notes)
        return round_to_base(1/(2 * np.median(extracted_notes[:,2].astype(float))) * 60, 10)
    
    def raw_sheet(self, extracted_notes, apply_heuristics):
        extracted_notes = np.array(extracted_notes)
        
        if self.bpm is None:
            self.bpm = self.bpm_estimation(extracted_notes)
            
        extracted_notes[:,2] = round_to_base(extracted_notes[:,2].astype(float), 1/self.bpm * 60/4)
        extracted_notes = extracted_notes[extracted_notes[:,2].astype("float") > 0]
        
        if apply_heuristics:
            for i in range(extracted_notes.shape[0]):
                if i != 0:
                    # Heuristic 1 -> Change string if note is close to previous but fret is not
                    h1_cond1 = abs(librosa.note_to_midi(extracted_notes[i-1,0])-librosa.note_to_midi(extracted_notes[i,0])) < 4
                    #h1_cond2 = extracted_notes[i-1,1] != extracted_notes[i,1]
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
        print(tab)
        
    def create_guitarpro_tab(self, sheet):
        template = guitarpro.parse('blank.gp5')
        measure_list = template.tracks[0].measures
        del template.tracks[0].measures[0].voices[0].beats[0]
        bar_start = 0
        current_measure = 0
        
        for i in range(len(sheet)):
            if (self.bpm/60) * (sheet[bar_start:i+1,2].astype("float").sum()) <= 4:
                duration = guitarpro.Duration(value= int(round(4/((self.bpm/60) * (float(sheet[i,2]))))))
                new_beat = guitarpro.Beat(template.tracks[0].measures[0].voices[0], duration=duration)
                new_note = guitarpro.Note(new_beat, value=int(sheet[i,3]), string=int(sheet[i,1]), 
                                          type=guitarpro.NoteType.normal)
                new_beat.notes.append(new_note)
                template.tracks[0].measures[current_measure].voices[0].beats.append(new_beat)
            else:
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
                    fitting_beats = 4 - (self.bpm/60) * (sheet[bar_start:i,2].astype("float").sum())
                    duration = guitarpro.Duration(value= int(round(4/fitting_beats)))
                    new_beat = guitarpro.Beat(template.tracks[0].measures[0].voices[0], duration=duration)
                    new_note = guitarpro.Note(new_beat, value=int(sheet[i,3]), string=int(sheet[i,1]), 
                                              type=guitarpro.NoteType.normal)
                    new_beat.notes.append(new_note)
                    template.tracks[0].measures[current_measure].voices[0].beats.append(new_beat)
                    
                    remaining_beats = (self.bpm/60) * (sheet[i,2].astype("float")) - fitting_beats
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
                    
            print(i,duration.value, (self.bpm/60) * (float(sheet[i,2])))
        
        guitarpro.write(template, 'final.gp5')
    
    def display_music_sheet(self, sheet, play_audio=False):
        stream1 = music21.stream.Stream()
        instrument = music21.instrument.ElectricGuitar()
        instrument.partName = "Guitar"
        stream1.append(instrument)
        
        for i in range(len(sheet)):
            current_note = music21.note.Note(sheet[i,0])
            current_note.quarterLength = (self.bpm/60) * (float(sheet[i,2]))
            stream1.append(current_note)
        
        if play_audio:
            stream1.show("midi")
        
        fp = stream1.write('midi', fp='test.mid')
            
        self.showScore(stream1)
        
    def showScore(self, score):
        xml = open(score.write('musicxml')).read()
        self.showMusicXML(xml)
        
    def showMusicXML(self, xml):
        DIV_ID = "OSMD-div-"+str(random.randint(0,1000000))
        print("DIV_ID", DIV_ID)
        display(HTML('<div id="'+DIV_ID+'">loading OpenSheetMusicDisplay</div>'))

        print('xml length:', len(xml))

        script = """
        console.log("loadOSMD()");
        function loadOSMD() { 
            return new Promise(function(resolve, reject){

                if (window.opensheetmusicdisplay) {
                    console.log("already loaded")
                    return resolve(window.opensheetmusicdisplay)
                }
                console.log("loading osmd for the first time")
                // OSMD script has a 'define' call which conflicts with requirejs
                var _define = window.define // save the define object 
                window.define = undefined // now the loaded script will ignore requirejs
                var s = document.createElement( 'script' );
                s.setAttribute( 'src', "https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@0.3.1/build/opensheetmusicdisplay.min.js" );
                //s.setAttribute( 'src', "/custom/opensheetmusicdisplay.js" );
                s.onload=function(){
                    window.define = _define
                    console.log("loaded OSMD for the first time",opensheetmusicdisplay)
                    resolve(opensheetmusicdisplay);
                };
                document.body.appendChild( s ); // browser will try to load the new script tag
            }) 
        }
        loadOSMD().then((OSMD)=>{
            console.log("loaded OSMD",OSMD)
            var div_id = "{{DIV_ID}}";
                console.log(div_id)
            window.openSheetMusicDisplay = new OSMD.OpenSheetMusicDisplay(div_id);
            openSheetMusicDisplay
                .load({{data}})
                .then(
                  function() {
                    console.log("rendering data")
                    openSheetMusicDisplay.render();
                  }
                );
        })
        """.replace('{{DIV_ID}}',DIV_ID).replace('{{data}}',json.dumps(xml))
        display(Javascript(script))
        return DIV_ID