# Introduction
The purpose of this project is to make up a prediction model where we will be able to predict whether a recommendation is positive or negative. In this project, we will not focus on the Score, but only the positive/negative sentiment of the recommendation.<br /><br />
To do so, we will work on Amazon's recommendation dataset, we will build a Term-doc incidence matrix using term frequency and inverse document frequency ponderation. When the data is ready, we will load it into predicitve algorithms, mainly naïve Bayesian and regression.

## Dataset
<br />

Dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1wxG9HndqfJXw4AdPTW3H69_kUXFA9U6S?usp=sharing).
After downloading the data copy it to the 'resources' folder.

## Installation
The application can be run in one of the two ways, either using Python Interpreter or using Jupyter Notebook.
<br />

**Python Environment** 
<br /><br />
This project require
[Python 3.6](https://www.python.org/ftp/python/3.6.3/python-3.6.3.exe) interpreter.<br /><br />
To use the Python interpreter to run the project, first install the python packages being used in this project.<br />
`pip3 install -r requirements.txt`<br />or
<br />
`pip install -r requirements.txt`
<br />
<br />
To run the application<br />
`$python P3`
<br />

**Conda Environment**
<br /><br />
The project requires [Anaconda 3](https://repo.continuum.io/archive/Anaconda3-5.0.1-Windows-x86_64.exe).
<br /><br />
To install Anaconda3 download the shell script from Anaconda website.<br />
Run the following command <br />
`bash Anaconda-latest-Linux-x86_64.sh`

Run the following to create environment and install packages<br />
`conda env create -f project_env.yml`

Run the following to run the Jupyter Notebook<br />
`jupyter notebook`

Select the '.ipynb' file.

