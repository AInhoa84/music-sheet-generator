import tkinter
from tkinter import filedialog
from tkinter import messagebox
      
from Sheet_generator import *

def open_file():
    try:
        file_name = filedialog.askopenfilename(initialdir="/",
                                          filetypes =(("Audio File", "*.mp3"),("All Files","*.*")),
                                          title = "Choose a file")
        if file_name != "":
            test = waveform(file_name)
        
    except:
        messagebox.showerror("Loading error", "Invalid file")
        
def analyze():
    print(file_name)
    #SG = sheet_generator("../Single_note_models/Guitar/Guitar", "../Single_note_models/Guitar/Guitar_norm_string")
    #results = SG.note_extraction(waveform(str(file_name)), False)
    #sheet = SG.raw_sheet(results, True)
    #display_sheet = tkinter.Label(window, text=SG.display_tablature(sheet), font=("Adobe Myungjo Std M", 30))
    #display_sheet.grid(column=0, row=3)

global file_name
file_name = ""

window = tkinter.Tk()
window.title("Guitar Tab Generator")

lbl = tkinter.Label(window, text="Welcome to Guitar Tab Generator!", font=("Adobe Myungjo Std M", 30))
lbl.grid(column=0, row=0)

menu = tkinter.Menu(window)
window.config(menu=menu)
file = tkinter.Menu(menu)
file.add_command(label = 'Open', command = open_file)
file.add_command(label = 'Exit', command = window.destroy)
analyze_menu = tkinter.Menu(menu)
analyze_menu.add_command(label = 'Open', command = analyze)

menu.add_cascade(label = 'File', menu = file)
menu.add_cascade(label = 'Analyze', menu = analyze_menu)

window.geometry('1024x700')

window.mainloop()
