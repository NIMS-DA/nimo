# nimo

NIMO is a Python library to realize a closed loop of robotic experiments and artificial intelligence without human intervention for automated materials exploration. We started development as NIMS-OS (NIMS Orchestration System) (https://github.com/nimsos-dev/nimsos) and decided to adopt NIMO as a nickname to promote it as open source software. NIMO can perform automated materials exploration in various combinations by considering artificial intelligence and robotic experiments as modules (see the figure below). As artificial intelligence technique for materials science, Bayesian optimization method (PHYSBO), boundLess objective-free exploration method (BLOX), phase diagram construction method (PDC), and random exploration (RE) can be used. NIMS Automated Robotic Electrochemical Experiments (NAREE) system is available as robotic experiments. Visualization tools for the results are also included, allowing users to check optimization results in real time. Newly created modules for artificial intelligence and robotic experiments can be added and used. More modules will be added in the future.

<img src="https://user-images.githubusercontent.com/125417779/234746422-245339fa-9902-41b6-8d0c-13c748ad839b.png" width="600px">


# Document

- [English](https://nimsos-dev.github.io/nimsos/docs/en/index.html)
- [日本語](https://nimsos-dev.github.io/nimsos/docs/jp/index.html)

# Required Packages

- Python >= 3.6
- Cython
- matplotlib
- numpy
- physbo >= 2.0
- scikit-learn
- scipy

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

# Uninstall

```bash
pip uninstall nimo
```

# License

The program package and the complete source code of this software are distributed under the MIT License.
