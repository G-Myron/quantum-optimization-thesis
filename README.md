# Quantum Optimization Algorithms - Diploma Thesis


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive Execution](#interactive-execution)
- [Project Structure](#project-structure)

## Introduction
This repository contains the code and resources for the project I developed for my Diploma Thesis.

## Installation
One can install the project by running the following code:

```bash
# Clone the repository
git clone https://github.com/G-Myron/quantum-optimization-thesis.git

# Navigate to the project directory
cd quantum-optimization-thesis

# Install dependencies
pip install -r requirements.txt
```

## Usage
To run the code for each of the three problems (knapsack, max-cut, TSP), open the terminal while inside the `quantum-optimization-thesis/` and run the lines of code bellow:
{The code provided is for the maxcut problem but in the same way can be excecuted any of the other problems according the [project structure](#project-structure)}

```sh
# Navigate to the specific problem's folder
cd maxcut

# Run the main script providing (optionally) a size n of the problem.
# Keep in mind that for values of n greater than a certain limit,
# the simulation of the quantum circuit will be problematic.
# It is suggested to keep n inside the following upper limits:
# n<10 for knapsack, n<16 for maxcut, n<5 for TSP.
python maxcut.py --n 6
```

## Interactive Execution
Alternatively, open the project in VScode and use the Python Interactive window. The main files are defined in Jupyter-like code cells beggining with a `# %%` comment which allows them to run interactively through the VScode environment.
To do that, open the problem's main file (`knapsack.py`, `maxcut.py` or `tsp.py`), place the cursor anywhere in the file and press `Ctrl+Enter`.
This way is suggested for exploring the code more interactively.

## Project Structure
The outline of the project's structure. There are three main directories, one for each problem.

    quantum-optimization-thesis/
    │   
    ├── knapsack/               # Knapsack code
    │   ├── helper_knapsack.py
    │   └── knapsack.py
    │       
    ├── maxcut/                 # MaxCut code
    │   ├── helper_maxcut.py
    │   └── maxcut.py
    │       
    ├── tsp/                    # TSP code
    │   ├── helper_tsp.py
    │   └── tsp.py
    │   
    ├── energy_landscape.py
    ├── problem_quantum.py
    ├── README.md               # Readme file
    └── requirements.txt        # List of dependencies
