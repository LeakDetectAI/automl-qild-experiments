# AutoMLQuantILDetect: Using AutoML approaches to detect and quantify information leakage in systems

AutoMLQuantILDetect is an advanced toolkit that harnesses the power of Automated Machine Learning (AutoML) to accurately quantify information leakage. 
This package specializes in estimating mutual information (MI) within systems that release classification datasets. 
By leveraging state-of-the-art statistical tests, it not only precisely quantifies mutual information (MI) but also effectively detects instances of information leakage within classification datasets. 
With AutoMLQuantILDetect, you can confidently and comprehensively address the critical challenges of quantification and detection in the realm of information leakage analysis.


Installation
------------
The latest release version of AutoMLQuantILDetect can be installed from GitHub as follows::
	
	pip install git+https://github.com/LeakDetectAI/automl_quant_il_detect.git
 
Another option is to clone the repository and install AutoMLQuantILDetect using::

	python setup.py install


Dependencies
------------
AutoMLQuantILDetect depends on AutoGLuon, TabPFN, Pytorch, Tensorflow, NumPy, SciPy, matplotlib, Scikit-learn, and tqdm. For data processing and generation, you will also need pandas.

License
--------
[Apache License, Version 2.0](https://github.com/LeakDetectAI/automl_quant_il_detect/blob/master/LICENSE)
