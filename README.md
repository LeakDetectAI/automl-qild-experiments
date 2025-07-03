
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
### 💬 Cite Us
If you use this toolkit in your research, please cite our paper available on arXiv:
```
	@article{GUPTA2025122419,
		title = {Information leakage detection through approximate Bayes-optimal prediction},
		journal = {Information Sciences},
		volume = {719},
		pages = {122419},
		year = {2025},
		issn = {0020-0255},
		doi = {https://doi.org/10.1016/j.ins.2025.122419},
		url = {https://www.sciencedirect.com/science/article/pii/S0020025525005511},
		author = {Pritha Gupta and Marcel Wever and Eyke Hüllermeier}
		}
	@PhdThesis{Gupta2025,
		  author={Gupta, Pritha},
		  title={Advanced Machine Learning Methods for Information Leakage Detection in Cryptographic Systems},
		  series={Institut f{\"u}r Informatik},
		  year={2025},
		  month={2025},
		  day={05-29T18:28:39},
		  publisher={Ver{\"o}ffentlichungen der Universit{\"a}t},
		  address={Paderborn},
		  pages={1 Online-Ressource (3, xi, 272 Seiten) Diagramme},
		  note={Tag der Verteidigung: 09.05.2025},
		  note={Universit{\"a}t Paderborn, Dissertation, 2025},
		  url={https://nbn-resolving.org/urn:nbn:de:hbz:466:2-54956},
		  language={eng}
		}
```
