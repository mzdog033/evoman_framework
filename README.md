### Installing dependant libraries:
```
python -m pip install -U pygame --user
pip install numpy tmx scipy matplotlib random pandas
pip uninstall tmx
```
We uninstall tmx since the local tmx in evoman folder is sufficient for the run.
### Running µ,λ survivor seletion. 
Open a terminal and run the following line
```
python muCommaLambda.py
```
### Running µ+λ survivor seletion. 
Open a terminal and run the following line
```
python muPlusLambda.py
```
### Plotting
Once the statistics from both runs are available, run the plotter
```
python plotter.py
```