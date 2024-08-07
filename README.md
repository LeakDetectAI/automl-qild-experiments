
Introduction
------------
Implementation of the experiments as submitted to the Information Sciences Journal.

The main scripts used to run the experiments to estimate the MI and detect leakage are cluster_script.py and cluster_script_ild.py. We used a Postgres database to configure all the jobs. The package autoqild can be installed using the instructions below:

Installation
------------
The latest release version of AutoMLQuantILDetect can be installed from GitHub as follows::
	
	pip install git+https://github.com/LeakDetectAI/AutoMLQuantILDetect.git
 
Another option is to clone the repository and install AutoMLQuantILDetect using::

	python setup.py install


Dependencies
------------
AutoMLQuantILDetect depends on AutoGLuon, TabPFN, Pytorch, Tensorflow, NumPy, SciPy, matplotlib, Scikit-learn, and tqdm. For data processing and generation, you will also need pandas.


Citing pyc-ilt
----------------

You can cite our `arXiv paper`_::


	@article{gupta2024information,
	      title={Information Leakage Detection through Approximate Bayes-optimal Prediction}, 
	      author={Pritha Gupta and Marcel Wever and Eyke Hüllermeier},
	      year={2024},
	      eprint={2401.14283},
	      archivePrefix={arXiv},
	      primaryClass={stat.ML}
	}


License
--------
[Apache License, Version 2.0](https://github.com/LeakDetectAI/automl_qild_experiments/blob/master/LICENSE)
