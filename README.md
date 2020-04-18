# Improving Embodiment with Reinforcement Learning in Virtual Reality

Yawen Hou


The main structure of the folder is:
```
.   
├── Data                                    # Not available due to privacy issues
├── Python Scrips                           # Folder containing the UCB algorithms written in Python. Read the README inside this folder for details.
├── Powerpoints                             # Folder containing the final powerpoint presentation (in .pptx and .pdf)
├── Report                                  # Folder containing the latex project and the pdf version of the report.
├── Videos                                  # Folder containing the videos taken during the experiment.
├── Statistical Analysis - Final.ipynb      # The jupyter notebook containing the code for statistical analysis.
├── environment.yml                         # The yml file containing all the packages needed to run this project.       
└── README.md                               # This file
```

## Setup
1. Install Anaconda from https://www.anaconda.com/distribution/.
2. Naviguate to this current folder in the terminal.
3. Type ```conda env create -f environment.yml``` to create the Anaconda environment containing all the installed packages. 
4. Type ```source activate python3``` to activate the environment.
5. Type ```jupyter notebook``` in the terminal to run jupyter notebook. It will open in a browser. Then you can open the Statistical Analysis - Final.ipynb used for the statistical analysis. Input the subject ids of the subjects you want to include in the analysis in the variable ```SUBJECT_IDS``` that you can find at the beginning of the notebook.
6. When done, quit the environment by typing ```conda deactivate```. **Unfortunately I cannot publish the subjects data so the analysis is for viewing only** 
