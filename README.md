# nimo

NIMO is a Python library to realize a closed loop of robotic experiments and artificial intelligence without human intervention for automated materials exploration. We started development as NIMS-OS (NIMS Orchestration System) (https://github.com/nimsos-dev/nimsos) and decided to adopt NIMO as a nickname to promote it as open source software. NIMO can perform automated materials exploration in various combinations by considering artificial intelligence and robotic experiments as modules. As artificial intelligence technique for materials science, Bayesian optimization method (PHYSBO), boundLess objective-free exploration method (BLOX), phase diagram construction method (PDC), Probability that properties within the Target Range(PTR), Bayesian optimization for materials and process parameters(BOMP), Bayesian optimization for combinatorial materials(COMBI), and random exploration (RE) can be used. Visualization tools for the results are also included, allowing users to check optimization results in real time. Newly created modules for artificial intelligence and robotic experiments can be added and used. More modules will be added in the future.


# Document

- [English](https://nims-da.github.io/nimo/en/)

# Required Packages

- Python >= 3.6
- matplotlib
- numpy
- physbo >= 3.0.0
- scikit-learn
- scipy


# About Cython
From NIMO v2.0.0, NIMO no longer uses Cython in order to simplify installation process particularly on Windows computer. This means that the performance of PHYSBO, PTR, BOMP, and COMBI is slightly degraded from older versions. If you need more performance, you can install physbo-core-cython additionally. This package offers Cythonized version of some functions of PHYSBO.


# Install

* From PyPI (recommended)

  ```bash
  pip install nimo
  ```

* From source

  1. Download or clone the github repository

  ```
  git clone https://github.com/NIMS-DA/nimo
  ```

  2. Install via pip in the nimo-main folder

  ```bash
  pip install .
  ```

* For physbo-core-cython (option, if C++ compiler is available)

  ```bash
  pip install physbo-core-cython
  ```


# Uninstall

```bash
pip uninstall nimo
```

# License

The program package and the complete source code of this software are distributed under the MIT License.
