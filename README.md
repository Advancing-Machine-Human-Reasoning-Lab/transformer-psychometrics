# Can Transformer Language Models Predict Psychometric Properties?


Code to reproduce experiments in our *SEM 2021 Paper.

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
@inproceedings{laverghetta-jr-etal-2021-transformer,
    title = "Can Transformer Language Models Predict Psychometric Properties?",
    author = "Laverghetta Jr., Antonio  and
      Nighojkar, Animesh  and
      Mirzakhalov, Jamshidbek  and
      Licato, John",
    booktitle = "Proceedings of *SEM 2021: The Tenth Joint Conference on Lexical and Computational Semantics",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.starsem-1.2",
    doi = "10.18653/v1/2021.starsem-1.2",
    pages = "12--25",
    abstract = "Transformer-based language models (LMs) continue to advance state-of-the-art performance on NLP benchmark tasks, including tasks designed to mimic human-inspired {``}commonsense{''} competencies. To better understand the degree to which LMs can be said to have certain linguistic reasoning skills, researchers are beginning to adapt the tools and concepts of the field of psychometrics. But to what extent can the benefits flow in the other direction? I.e., can LMs be of use in predicting what the psychometric properties of test items will be when those items are given to human participants? We gather responses from numerous human participants and LMs (transformer- and non-transformer-based) on a broad diagnostic test of linguistic competencies. We then use the responses to calculate standard psychometric properties of the items in the diagnostic test, using the human responses and the LM responses separately. We then determine how well these two sets of predictions match. We find cases in which transformer-based LMs predict psychometric properties consistently well in certain categories but consistently poorly in others, thus providing new insights into fundamental similarities and differences between human and LM reasoning.",
}

```