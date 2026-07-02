

![](utils/utils_README/banner.png)

## Installation

**1. Clone this git project on your computer**

**2. Install the virtual environment**

&nbsp;&nbsp;&nbsp;&nbsp; This virtual environment relies on Python3.12, install it.

&nbsp;&nbsp;&nbsp;&nbsp; Install this virtual environment from the requirements.txt files

## Folders


- "data" folder contains the data from various sources 

- "docs" folder contains some documentation, i.e. code examples and generated figures

- "emulator" folder contains the machine learning models for the emulation

- "optimization" folder contains classes for hyperparameter optimization

- "plot" folder contains functions to create diagnosis plots for emulators

- "projects" folder contains folders that generate results/plots  

- "results" folder will contain the generated plots

- "tests" folder contain all tests 

- "utils" folder contains transverse classes/functions that are used in all other folders.

## Code convention

- Variables and functions names must be explicit and with type hinting. Documentation should be used only for tricky functions
- Class names should have the same name as their file. By default, child classes should have the same prefix as their parents. Object name should be the same as its class (replacing CamelCaseStyle with an underscore_style)  
- Long python files must be avoided (using code decoupling or sub-folders, and by creating utils_*.py files to store decoupled functions) 