# Continuous Buhmbox New

Run with python 2.7.9 (you can use pyenv and pyenv virtualenv if you want).

Run 
```
pip install -r requirements.txt
```

Then you can run
```
python cont_bb.py
```

The first time the file is run will take time, two simulations will be run (could take 10ish hours).
It will then cache the data, so when you run the following times, the data will simply be loaded and displayed.
If you want to regenerate the data (rerun the simulations) simply move or delete the relevant pickle files.