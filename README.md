# Can Transformer Language Models Predict Psychometric Properties?


Code to reproduce experiments in our *SEM 2021 Paper

This folder contains supplementary material to aid in reproducing our results.

# Reproducing the human experiment results

```processHumanResults.py``` contains the code for reproducing all experiments from our paper. We have also provided the diagnostic scores in ```Diagnostic Results.tsv```, there is no need to train and evaluate any neural models if one only wishes to experiment with the psychometric measures.


```
# create a virtual environtment
python3 -m venv transformer-env
source transformer-env/bin/activate

# install the necessary libraries
pip install -r requirements.txt

# reproduce the human results from the paper (Table X). This script doesn't require SNLI, MNLI or ANLI datasets
python3 processHumanResults.py

```

If you are in a Windows machine, activate your environment using:

```
transformer-env\Scripts\activate.bat

```

# Reproducing the neural LM results

```model.py, train.py, and util.py``` are the scripts for training the LSTM-based LMs. ```eval_finetune_transformers.py``` and ```eval_finetune_T5.py``` are the scripts for training transformer-based LMs. It is necessary to download the SNLI, MNLI, and ANLI train and dev set to use these scripts (see below), please see our paper for further details on the precise configurations used. In the case of the LSTMs, each LSTM-based individual in the diagnostic tsv includes the precise hyperparameters used as part of the individual name.

## Dowloading datasets

To download the datasets, you can run the following commands:

SNLI
```
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
```

MNLI
```
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
```

ANLI
```
wget https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip
```

The random population is generated in the psychometrics script. Note that, because a new random population is generated every time, your correlations for this population may not match what we have reported in the paper. However, we did not find that we could consistently achieve a strong correlation with the random baseline, as expected.

Please see each of these scripts for further details and necessary packages, as well as our paper.

# Credits

If you want to use or reference this work, please cite the following:

```
@article{DBLP:journals/corr/abs-2106-06849,
  author    = {Antonio Laverghetta Jr. and
               Animesh Nighojkar and
               Jamshidbek Mirzakhalov and
               John Licato},
  title     = {Can Transformer Language Models Predict Psychometric Properties?},
  journal   = {CoRR},
  volume    = {abs/2106.06849},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.06849},
  archivePrefix = {arXiv},
  eprint    = {2106.06849},
  timestamp = {Tue, 15 Jun 2021 16:35:15 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-06849.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```