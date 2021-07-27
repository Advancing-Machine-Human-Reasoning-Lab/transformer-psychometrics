# transformer-psychometrics
Code to reproduce experiments in our *SEM 2021 Paper


This folder contains supplementary material to aid in reproducing our results.

# Psychometrics calculations

```processHumanResults.py``` contains the code for reproducing all experiments from our paper. We have also provided the diagnostic scores in ```Diagnostic Results.tsv```, there is no need to train and evaluate any neural models if one only wishes to experiment with the psychometric measures.

# Training neural LMs

```model.py, train.py, and util.py``` are the scripts for training the LSTM-based LMs. ```eval_finetune_transformers.py``` and ```eval_finetune_T5.py``` are the scripts for training transformer-based LMs. It is necessary to download the SNLI, MNLI, and ANLI train and dev set to use these scripts, please see our paper for further details on the precise configurations used. In the case of the LSTMs, each LSTM-based individual in the diagnostic tsv includes the precise hyperparameters used as part of the individual name.

The random population is generated in the psychometrics script. Note that, because a new random population is generated every time, your correlations for this population may not match what we have reported in the paper. However, we did not find that we could consistently achieve a strong correlation with the random baseline, as expected.

Please see each of these scripts for further details and necessary packages, as well as our paper.