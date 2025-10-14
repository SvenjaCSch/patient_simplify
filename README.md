# Patient Simplification
A Small Language Model (SLM) Application that is simplifying the explanation of the doctor.
It contains various different evaluations with PromptLayer focussing on the Input and Prompts as well as wandb for Metrics and Logging. 
The docker container is provided. 

## Installation guide
This repository was created under python 3.12.4. 
### Python installation
Ensure Python 3.12.4 is installed on your system. For Ubuntu:
```bash
sudo apt install python3.12.4
 ```
For MacOS:
```bash
brew install python@3.12.4
```
I myself used Windows and Visual Studio Code to install python. You can download VSC here:
```bash
https://code.visualstudio.com/download
```
I also installed the extensions for SQLite Viewer and SQLite by alexcuzz in VSC.
### Requirements
Make sure you clone the repository properly. After that create a virtual environment with
```bash
python -m venv venv
```
Get all the necessary requirements:
```bash
pip install -r requirements.txt
```

