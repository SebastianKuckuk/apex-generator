# Accelerated Programming EXamples Generator (APEX-Generator)

Generated and additional manually implemented examples can be found in the [APEX main repository](https://github.com/SebastianKuckuk/apex).

## Prerequisites

Requires the following Python packages
* `sympy` for constructing pseudo-ASTs
* `pandas` for handling benchmark results
* `matplotlib` for plotting benchmark results
* `openpyxl` for exporting benchmark data as `xlsx` (MS Excel) files

Setup in a Conda environment can be done as follows
```bash
conda create -c conda-forge -n apex -y
conda activate apex
conda install sympy pandas matplotlib openpyxl -y
```

## Usage

```bash
cd src
# general pattern: python generate.py machine app backend
python generate.py nvidia.alex.a40 stream base
# general pattern: python compile.py machine app backend parallel
python compile.py nvidia.alex.a40 stream base true
# general pattern: python execute.py machine app backend
python execute.py nvidia.alex.a40 stream base
```

Alternatively, the `gce` script (**g**enerate, **c**ompile, **e**xecute) can be used

```bash
# general pattern: python gce.py machine app backend parallel
python gce.py nvidia.alex.a40 stream base true
```
