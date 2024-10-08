
Introduction
------------
Implementation of the experiments as submitted to the Information Sciences Journal.

## Job Configurations

The job configurations for running different mutual information estimation techniques on synthetic datasets, used by `cluster_script.py`, are uploaded as:

1. `database\mutual_information.avail_jobs.csv`
2. `database\mutual_information.running_jobs.csv`
3. `database\automl.avail_jobs.csv`
4. `database\automl.running_jobs.csv`

For running the experiments to detect information leakage on real datasets generated using OpenSSL TLS server, used by `cluster_script_ild.py`, the configurations are uploaded as:

1. `database\leakage_detection_padding.avail_jobs.csv`
2. `database\leakage_detection_padding.running_jobs.csv`

These files contain all the necessary configurations to execute the respective tasks within your experiments. Ensure to update and review these files as needed for your simulations.

You can find the folder containing these files at the following link: [database folder](https://github.com/LeakDetectAI/automl-qild-experiments/tree/master/database).



Installation
------------
The package `autoqild` package used for running the experiments can be installed using the instructions below:

The latest release version of AutoMLQuantILDetect can be installed from GitHub as follows::
	
	pip install git+https://github.com/LeakDetectAI/AutoMLQuantILDetect.git
 
Another option is to clone the repository and install AutoMLQuantILDetect using::

	python setup.py install


### Dependencies

AutoMLQuantILDetect depends on the following libraries:
- AutoGLuon
- TabPFN
- Pytorch
- Tensorflow
- NumPy
- SciPy
- matplotlib
- Scikit-learn
- tqdm
- pandas (required for data processing and generation)

### Citing autoqild

If you use this toolkit in your research, please cite our paper available on arXiv:

```
@article{gupta2024information,
  title={Information Leakage Detection through Approximate Bayes-optimal Prediction},
  author={Pritha Gupta, Marcel Wever, and Eyke Hüllermeier},
  year={2024},
  eprint={2401.14283},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}
```
