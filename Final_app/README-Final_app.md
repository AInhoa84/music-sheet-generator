# Building the final desktop application

In order to replicate the building process:

1. Install LightGBM following these steps: https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
2. Uninstall enum34
   `pip uninstall enum34`
3. Run "pyinstaller GUI.spec" on the "Final_app" dir. Specific imports have been added to the GUI.spec file (otherwise the app won't build or won't run). It might ask for permission to overwrite "dist", grant it.
4. Re-train the LightGBM model ("Tempo/Tempo_lightgbm.ipynb) and save the model to "Final_app/dist/Tempo"
5. Copy the "lightgbm" folder to "dist/GUI".
6. Copy the contents of the "add_to_dist" folder to "dist" (if they are not already there).
7. Make sure the "blank.gp5" file is inside the "Final_app" dir (if it is not, you can find it in other folders of the repo).

Now you should be able to run the app by executing "dist/GUI/GUI.exe".
