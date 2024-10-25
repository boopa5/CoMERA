This repo contains an implementation of the Neurips 2024 paper CoMERA at https://arxiv.org/abs/2405.14377. Now, the repo replicates the experiment of CoMERA training a six-encoder Transformer on the MNLI dataset. 

To run the experiment on MNLI, follow the following steps. 
* Install required packages in requirements.txt.
* Create folders logs, datasets, models_MNLI.
* Run data_process.py to preprocess MNLI dataset.
* Run script run_MNLI.sh to start training.
