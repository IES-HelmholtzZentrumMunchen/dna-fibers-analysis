# dna-fibers-analysis

[![Build Status](https://travis-ci.org/IES-HelmholtzZentrumMunchen/dna-fibers-analysis.svg?branch=master)](https://travis-ci.org/IES-HelmholtzZentrumMunchen/dna-fibers-analysis)  [![Coverage Status](https://coveralls.io/repos/github/IES-HelmholtzZentrumMunchen/dna-fibers-analysis/badge.svg?branch=master)](https://coveralls.io/github/IES-HelmholtzZentrumMunchen/dna-fibers-analysis?branch=master)

This python package contains useful tools for analyzing the microscopy images obtained from a DNA fibers experiment.

## Installation

There is no automatic setup using PyPy yet, so the repository should be cloned on the local machine and the path must be added to the python path environnement variable, using the following commands.
```
git clone https://github.com/IES-HelmholtzZentrumMunchen/dna-fibers-analysis.git /path/on/local/machine
export PYTHONPATH=/path/on/local/machine:$PYTHONPATH
```

The tests can be ran by using the following command in the local path.
```
nosetests .
```

## Documentation

The API documentation is available through Sphinx building in the doc folder.
