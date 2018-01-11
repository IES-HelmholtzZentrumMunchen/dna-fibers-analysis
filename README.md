# dna-fibers-analysis

[![Build Status](https://travis-ci.org/IES-HelmholtzZentrumMunchen/dna-fibers-analysis.svg?branch=master)](https://travis-ci.org/IES-HelmholtzZentrumMunchen/dna-fibers-analysis)  [![Coverage Status](https://coveralls.io/repos/github/IES-HelmholtzZentrumMunchen/dna-fibers-analysis/badge.svg?branch=master&service=github)](https://coveralls.io/github/IES-HelmholtzZentrumMunchen/dna-fibers-analysis?branch=master&service=github)

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

## Usage

The DFA module has several commands, that can be listed by typing the following in a terminal with an active environnement.

```
python -m dfa -h
```

Each command has it own help that can be displayed by using similar command as the following one, that is simulating an DNA fibers image.

```
python -m dfa simulate
```

The help for each command is accessible through
```
python -m dfa simulate -h
```

Although this usage section is organized as a small tutorial example with simulated data, the sub-section titles help to find information about a specific question. Therefore, despite it is meant to be red sequentially, parts can be picked up when needed.

### DNA fibers image simulation

Let's simulate a few images to be used in the following examples. For that purpose, we will need to get the PSF file (named psf_610nm-1.4NA-62.5nm-125nm-32.tif and located in doc directory) and run the following commands.

```
python -m dfa simulate ./psf_610nm-1.4NA-62.5nm-125nm-32.tif --number 10 --ij --output ./image-1.tif
python -m dfa simulate ./psf_610nm-1.4NA-62.5nm-125nm-32.tif --number 10 --ij --output ./image-2.tif
python -m dfa simulate ./psf_610nm-1.4NA-62.5nm-125nm-32.tif --number 10 --ij --output ./image-3.tif
```

Here we feed the commands with the path to the PSF file, we set the number of fibers to generate per image to 10, we activate the compatibility with ImageJ (for ROI showing the fibers) and we set the output path. All other parameters are set to default (see simulate command help).

For clarity, we put the images and the fibers paths into separate folders; for instance respectively into `images` and `fibers-truth` folders.

### Batch processing

The easiest way to process the images is to use the `pipeline` command, that is running the whole pipeline in batch processing on the input images. The usage is simply the following.

```
python -m dfa pipeline ./images .
```

The first and the second parameters are respectively the input (the images) and the output path (here we set the output as the current directory). Once the command starts, it shows progress bars that helps tracking the progress state.

With the default parameters, the output is only one file sumaryzing all the fibers, patterns and lengths in `./analysis/detailed_analysis.csv`. However, it is possible to save intermediate files, such as the detected fibers or the unfolded fibers, using `--save...` flags. It is also possible to save every intermediate file by using the flag `--save-all`.

The command also allows to control the algorithm, for instance by setting the sensitivity (`--intensity-sensitivity`), the reconstruction extent (`--reconstruction-extent`) or the pixel size (`--pixel-size`). Please refer to the command help (`-h` or `--help`) for a complete list of options.

### Quantifying

When images are processed and at least a detailed analysis output file is generated, quantification can be performed with the following command.

```
python -m dfa quantify ./analysis/detailed_analysis.csv --output ./analysis/example
```

Three files containing the quantifications are outputed: the fork rate, the fork speed and the fiber patterns. The `--output` option allows to specify the path and the prefix of the output files (here `./analysis` is the path and `example` is the prefix).

### Masking

When the interesting part of a huge image is located in a small area or some parts of the image are not usable (in case of overlapping of fibers or in the presence of dust for instance), using masks is a good practice.

The masks must be created with any image editor. They are simply images with non-zero pixels defining valid areas. To make them, it is possible to use ImageJ by drawing a region of interest, then using the tool `Create Mask` and finally save the them to a directory.

Assuming that we did so for all the images and we put the masks in the folder `./masks`, running the pipeline with the masks is simply done by typing the following command.

```
python -m dfa pipeline ./images . --masks ./masks
```

Only the regions inside the masks will be then processed. Also, you will notice that the processing is much faster.

### Interacting and correcting

Sometimes, the process does not work exactly as expected, for instance when a fiber is missed or not properly labelled. Therefore, it is a good practice to do the steps separately: fibers detection, fibers extraction and patterns analysis.

For that purpose, there are three dedicated commands. The first one is the fibers detection command, for which an exemple is written below.
```
mkdir ./fibers
python -m dfa detect ./images --mask ./masks --output ./fibers --ij
```
This command outputs two files per image: an overlay of the detected fibers on the images and a zip file containing the fibers files. When using the option `--ij`, it outputs ImageJ-compatible files that can be loaded into ImageJ as regions of interest.

After this step, it is possible to interact with the output to correct it. This is achieved by loading the ImageJ fibers into ImageJ, by removing, moving or adding fibers and by saving the modified set of fibers to the disk and overwriting the old set. To save the modified fibers with ImageJ, on the *ROI Manager*, click on `More >>` and `Save...`, then select the file you just loaded from the drive.

The second dedicated command is the fibers extraction command, which usage is the following.
```
mkdir ./profiles
python -m dfa extract ./images ./fibers --output ./profiles
```
Two files per fiber are outputed: the profiles (`csv` files) and the quickviews (`png` files). Note that using the `--group-fibers` option will group the profiles per image. This command accepts also the `--pixel-size` options that is used to set the image calibration.

The last command to use in the pipeline is the patterns analysis one, for which an example is shown below.
```
python -m dfa analyze ./profiles --output ./analysis.csv
```
When the progress bar displayed by the command reaches the end, the
output file of the analysis is written at the location indicated by the option `--output`. This command also accepts options for controlling the regularization of the patterns detection algorithm (see the help).

### Utilities

In addition with the `simulate` command, there are two other utility commands: one for comparing the results obtained by the algorithm to a ground-truth and one for creating and packing a dataset. Usage information can be found in the help of each command.
