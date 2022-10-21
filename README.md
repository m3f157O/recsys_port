# RecSys algorithm porting project

This is the repository for the RecSys algorithm porting project
File _run_CIKM_18_ExampleAlgorithm_ contains an example of the full experiment pipeline which if you have implemented your DataReader and RecommenderWrapper correctly will work. 
File Test/run_unittest_recommenders.py contains the tests your RecommenderWrapper should pass.
Folder _Conferences_ contains an example of the material you have to deliver.

## Code organization
This repository is organized in several subfolders.

#### Deep Learning Algorithms
The Deep Learning algorithms are all contained in the _Conferences_ folder and further divided in the conferences they were published in.
For each DL algorithm the repository contains two subfolders:
* A folder named "_github" which contains the full original repository or "_original" which contains the source code provided by the authors upon request, with the minor fixes needed for the code to run.
* A folder named "_our_interface" which contains the python wrappers needed to allow its testing in our framework. The main class for that algorithm has the "Wrapper" suffix in its name. This folder also contains the functions needed to read and split the data in the appropriate way.

Note that in some cases the original repository contained also the data split used by the original authors, those are included as well.

#### Baseline algorithms
Folders like "KNN", "GraphBased", "MatrixFactorization", "SLIM_BPR", "SLIM_ElasticNet" and "EASE_R" contain all the baseline algorithms we used in our experiments.

#### Evaluation
The folder _Base.Evaluation_ contains the two evaluator objects (_EvaluatorHoldout_, _EvaluatorNegativeSample_) which compute all the metrics we report.

#### Data
The data to be used for each experiments is gathered from specific _DataReader_ objects withing each DL algoritm's folder. 

The folder _Data_manager_ contains a number of _DataReader_ objects each associated to a specific dataset, which are used to read datasets for which we did not have the original split. 

Whenever a new dataset is downloaded and parsed, the preprocessed data is saved in a new folder called _Data_manager_split_datasets_, which contains a subfolder for each dataset. The data split used for the experimental evaluation is saved within the result folder for the relevant algorithm, in a subfolder _data_ . 

#### Hyperparameter optimization
Folder _ParameterTuning_ contains all the code required to tune the hyperparameters of the baselines. The script _run_parameter_search_ contains the fixed hyperparameters search space used in all our experiments.
The object _SearchBayesianSkopt_ does the hyperparameter optimization for a given recommender instance and hyperparameter space, saving the explored configuration and corresponding recommendation quality. 




## Run the experiments

See see the following [Installation](#Installation) section for information on how to install this repository.
After the installation is complete you can run the experiments.


All experiments related to a DL algorithm reported in our paper can be executed by running the corresponding script, which is preceeded by _run__, the conference name and the year of publication.
The scripts have the following boolean optional parameters (all default values are False):
* '-b' or '--baseline_tune': Run baseline hyperparameter search
* '-a' or '--DL_article_default': Train the deep learning algorithm with the original hyperparameters
* '-p' or '--print_results': Generate the latex tables for this experiment


For example, if you want to run all the experiments for SpectralCF, you should run this command:
```console
python run_RecSys_18_SpectralCF.py -b True -a True -p True
```



The script will:
* Load and split the data.
* Run the bayesian hyperparameter optimization on all baselines, saving the best values found.
* Run the fit and test of the DL algorithm
* Create the latex code of the result tables, as well as plot the data splits, when required. 
* The results can be accessed in the _result_experiments_ folder.






## Installation

Note that this repository requires Python 3.8

First we suggest you create an environment for this project using virtualenv (or another tool like conda)

First checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment, if you are using conda:
```console
conda create -n RecSysFramework python=3.8 anaconda
conda activate RecSysFramework
```

In order to compile you must have installed: _gcc_ and _python3 dev_, which can be installed with the following commands:
```console
sudo apt install gcc 
sudo apt-get install python3-dev
```

Then install all the requirements and dependencies
```console
pip install -r requirements.txt
```


At this point you can compile all Cython algorithms by running the following command. The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some warnings. 
 
```console
python run_compile_all_cython.py
```

