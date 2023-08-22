[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Tree partitioning
The repository contain implementations of /tree partitioning/ algorithms for power transmission networks, turning the network into clusters that are connected in a tree-like manner.

See [our paper](#paper) for more information about tree partitioning and the implemented algorithms.

![ALNS logo](IEEE-73.jpg)

## Installation
Make sure to have [Poetry](https://python-poetry.org/) installed with version 1.2 or higher. 
The following command will then install all necessary dependencies:

```bash
poetry install
```

If you don't have Poetry installed, make sure that you have Python 3.9 or higher and install the packages indicated in the `pyproject.toml` file. 

Running the code also requires a [Gurobi](https://www.gurobi.com/) license. 

## Paper

For more details about tree partitioning, see our paper [*Mixed-integer linear programming approaches for tree partitioning of power networks*](https://arxiv.org/abs/2110.07000). If this code is useful for your work, please consider citing our work:

``` bibtex
@misc{Lan2023a,
  title = {Mixed-integer linear programming approaches for tree partitioning of power networks},
  author = {Lan, Leon and Zocca, Alessandro},
  year = {2023},
  month = aug,
  number = {arXiv:2110.07000},
  eprint = {2110.07000},
  primaryclass = {cs, eess, math},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2110.07000},
}
```

