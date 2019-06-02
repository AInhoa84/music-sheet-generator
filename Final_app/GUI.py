from Sheet_generator_GUI import *

import PySimpleGUI as sg

# All widgets
layout = [
    [sg.Text('Input file', size=(15, 1), auto_size_text=False, justification='right'),      
    sg.InputText(), sg.FileBrowse(file_types=(("All files", "*.*"),("Audio files", ".mp3")))],
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
    
    # Ask for save location and name of the gp5 file
    if values["Create gp5 file"] and file_name != "":
        gp5_name = sg.PopupGetFile("Save generated gp5 file to", save_as=True, 
                                   file_types=(("All files", "*.*"),("Guitarpro files", ".gp5")),
                                   default_extension="*.gp5")
        if gp5_name is None:
            gp5_name = "untitled"
        
    # Run sheet generator
    SG = sheet_generator("../Single_note_models/Guitar/Guitar", "../Single_note_models/Guitar/Guitar_norm_string", bpm=120)
    results = SG.note_extraction(waveform(file_name))
    sheet = SG.raw_sheet(results, values["Apply heuristics"])
    
    if values["Display tab"]:
        sg.Popup("Guitar tab", SG.display_tablature(sheet), font=("BatangChe", 18))
        
    if values["Create gp5 file"]:
        SG.create_guitarpro_tab(sheet, gp5_name + ".gp5")

window.Close()