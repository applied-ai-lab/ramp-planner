# RAMP Benchmark Planner 2023

A collaboration between the Manufacturing Technology Centre and Oxford Robotics Institute.
For more information please see the project website: https://sites.google.com/oxfordrobotics.institute/ramp

## Planning Module for the RAMP Assembly Benchmark

### 1. Installation

Tested on ubuntu 22.04.02 LTS

1. Install Dependencies

java runtime environment
   - install via apt
   ```bash
   sudo apt install default-jre
   ```

clingo
   - Add potassco stable ppa to apt and install clingo
   ```bash
   sudo add-apt-repository  ppa:potassco/stable
   sudo apt update
   sudo apt install clingo
   ```

 SPARC
   - Clone SPARC to ASP translator repo (https://github.com/iensen/sparc/)
   - Set environment variable
     ```bash
     echo 'export SPARC_PATH=path/to/sparc/folder' >> ~/.bashrc
     ```

Python distutils
   - if not included with your python build can be installed using apt, we use python3.10 but this should also work for other versions.
   ```bash
   sudo apt install python3.10-distutils
   ```

2. Download this Repo and configure
   - Clone this repo
     ```bash
     git clone git@github.com:applied-ai-lab/ramp-planner.git
     ```
   - Set PLANNER_PATH environment variable
     ```bash
     echo 'export PLANNER_PATH=path/to/repo/folder' >> ~/.bashrc
     ```
   - Change to repo base directory
     ```bash
     cd ramp-planner
     ```
   - Install python dependencies
     ```bash
     python3 -m pip install -r requirements.txt
     ```
   - Install python modules (may vary depening on your python install)
     ```bash
     python3 setup.py build
     sudo python3 setup.py install
     ```

### 2. Usage

1. Run example planner with
   ```bash
   cd ramp-planner
   python3 -m sparc_planning.src.main
   ```
