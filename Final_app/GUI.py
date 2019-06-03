from Sheet_generator_GUI import *

import PySimpleGUI as sg

# All widgets
layout = [
    [sg.Text('Input file', size=(15, 1), auto_size_text=False, justification='right'),      
    sg.InputText(), sg.FileBrowse(file_types=(("All files", "*.*"),("Audio files", ".mp3")))],
    [sg.Checkbox('Manually input file bpm (will be estimated otherwise)', default=True, key="BPM")],
    [sg.Checkbox('Display tab', default=True, key="Display tab")],
    [sg.Checkbox("Apply heuristics", default=True, key="Apply heuristics")],
    [sg.Checkbox("Create gp5 file", default=True, key="Create gp5 file")],
    [sg.Submit(button_text="Run tab generator")],
    [sg.CButton("Close")]
]

# Main window
window = sg.Window('Guitar Tab Generator', layout)

# Main loop
while True:
    button, values = window.Read()
    
    file_name = values["Browse"]
    bpm = None
    
    # Ask for bpm if option is checked
    if values["BPM"] and file_name != "":
        while bpm == "":
            bpm = sg.PopupGetText("Enter the file's BPM", default_text="120")
    
    # Ask for save location and name of the gp5 file
    if values["Create gp5 file"] and file_name != "":
        gp5_name = sg.PopupGetFile("Save generated gp5 file to", save_as=True, 
                                   file_types=(("All files", "*.*"),("Guitarpro files", ".gp5")),
                                   default_extension="*.gp5")
        if gp5_name is None:
            gp5_name = "untitled"
        
    # Run sheet generator
    if bpm is not(None):
        SG = sheet_generator("../Single_note_models/Guitar/Guitar", 
                             "../Single_note_models/Guitar/Guitar_norm_string", 
                             "../Tempo/tempo_lgbm.txt",
                             bpm=int(bpm))
    else:
        SG = sheet_generator("../Single_note_models/Guitar/Guitar", 
                             "../Single_note_models/Guitar/Guitar_norm_string", 
                             "../Tempo/tempo_lgbm.txt")
        
    results = SG.note_extraction(waveform(file_name))
    sheet = SG.raw_sheet(results, values["Apply heuristics"])
    
    if values["Display tab"]:
        sg.Popup("Guitar tab", SG.display_tablature(sheet), font=("Consolas", 18))
        
    if values["Create gp5 file"]:
        if ".gp5" not in gp5_name:
            gp5_name += ".gp5"
            
        SG.create_guitarpro_tab(sheet, gp5_name)

window.Close()