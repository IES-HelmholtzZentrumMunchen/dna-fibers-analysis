# dna-fibers-analysis

[![Build Status](https://travis-ci.org/IES-HelmholtzZentrumMunchen/dna-fibers-analysis.svg?branch=master)](https://travis-ci.org/IES-HelmholtzZentrumMunchen/dna-fibers-analysis)  [![Coverage Status](https://coveralls.io/repos/github/IES-HelmholtzZentrumMunchen/dna-fibers-analysis/badge.svg?branch=master)](https://coveralls.io/github/IES-HelmholtzZentrumMunchen/dna-fibers-analysis?branch=master)

This python package contains useful tools for analyzing the microscopy images obtained from a DNA fibers experiment.

## Installation

The package is compatible with `python 3.6` and above, so please check that you have it before using.

There is no automatic setup using PyPy yet, so the repository should be cloned on the local machine and the path must be added to the python path environnement variable, using the following commands.
```
git clone https://github.com/IES-HelmholtzZentrumMunchen/dna-fibers-analysis.git /path/on/local/machine
export PYTHONPATH=/path/on/local/machine:$PYTHONPATH
```

Before use, the dependencies must be installed. It is recommended to use a virtual environnement. To do so, use the following commands in the local path.
```
python3.6 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

To be sure that everything work fine, the tests can be ran by using the following command in the local path.
```
nosetests
```

When the work is done, you can leave the virtual environnement by typing `deactivate`. Do not forget to activate the environnement before using the package, using the command `source /path/to/local/machine/venv/bin/activate`.

## Documentation

The API documentation is available through Sphinx building in the doc folder, using for instance the following commands.
```
cd ./doc
make html
```
