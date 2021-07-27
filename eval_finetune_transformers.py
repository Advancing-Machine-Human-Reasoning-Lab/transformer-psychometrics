"""
    Gather results on the diagnostic using the specificed transformer
    See the argparser below for details on the command line arguments
"""
import pandas as pd
from simpletransformers.classification import ClassificationModel
# import sklearn
import torch
from sys import argv
from os import path, makedirs
import numpy as np
import wandb
import argparse

def DiagTrial(args):

    if not path.isdir('./checkpoint_results'):
        makedirs('./checkpoint_results/finetune')
        makedirs('./checkpoint_results/nofinetune')


    print(f"Starting {path.basename(args.model_type)}")
    diagnostic = pd.read_csv("./BERT_Diagnostic_Set.csv")
    dev_diag = pd.DataFrame({
        'index':diagnostic['ID'],
        'text_a':diagnostic['Premise'],
        'text_b':diagnostic['Hypothesis'],
        'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in diagnostic['Label']]})

    # load the GLUE diagnostic set
    if args.finetune != 'no':
        # get the specific finetuing set based on the passed arg
        # assert '+' in args.finetune
        train_sets = args.finetune.split("+")

        # read the input files
        train_snli = pd.read_json("./datasets/snli/snli_1.0_train_cleaned.jsonl", lines=True)
        train_snli = pd.DataFrame({
            'text_a':train_snli['sentence1'],
            'text_b':train_snli['sentence2'],
            'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in train_snli['gold_label']]})

        train_mnli = pd.read_json("./datasets/multinli_1.0/multinli_1.0_train.jsonl", lines=True)
        train_mnli = pd.DataFrame({
            'text_a':train_mnli['sentence1'],
            'text_b':train_mnli['sentence2'],
            'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in train_mnli['gold_label']]})


        train_anli_r1 = pd.read_json("./datasets/anli_v1.0/R1/train.jsonl", lines=True)
        train_anli_r2 = pd.read_json("./datasets/anli_v1.0/R2/train.jsonl", lines=True)
        train_anli_r3 = pd.read_json("./datasets/anli_v1.0/R3/train.jsonl", lines=True)
        train_anli_r1 = pd.DataFrame({
            'text_a':train_anli_r1['context'],
            'text_b':train_anli_r1['hypothesis'],
            'labels':[['c', 'n', 'e'].index(gs) for gs in train_anli_r1['label']]})
        train_anli_r2 = pd.DataFrame({
            'text_a':train_anli_r2['context'],
            'text_b':train_anli_r2['hypothesis'],
            'labels':[['c', 'n', 'e'].index(gs) for gs in train_anli_r2['label']]})
        train_anli_r3 = pd.DataFrame({
            'text_a':train_anli_r3['context'],
            'text_b':train_anli_r3['hypothesis'],
            'labels':[['c', 'n', 'e'].index(gs) for gs in train_anli_r3['label']]})


        dev_anli_r1 = pd.read_json("./datasets/anli_v1.0/R1/dev.jsonl", lines=True)
        dev_anli_r2 = pd.read_json("./datasets/anli_v1.0/R2/dev.jsonl", lines=True)
        dev_anli_r3 = pd.read_json("./datasets/anli_v1.0/R3/dev.jsonl", lines=True)
        dev_anli_r1 = pd.DataFrame({
            'text_a':dev_anli_r1['context'],
            'text_b':dev_anli_r1['hypothesis'],
            'labels':[['c', 'n', 'e'].index(gs) for gs in dev_anli_r1['label']]})
        dev_anli_r2 = pd.DataFrame({
            'text_a':dev_anli_r2['context'],
            'text_b':dev_anli_r2['hypothesis'],
            'labels':[['c', 'n', 'e'].index(gs) for gs in dev_anli_r2['label']]})
        dev_anli_r3 = pd.DataFrame({
            'text_a':dev_anli_r3['context'],
            'text_b':dev_anli_r3['hypothesis'],
            'labels':[['c', 'n', 'e'].index(gs) for gs in dev_anli_r3['label']]})

        dev_snli = pd.read_json("./datasets/snli/snli_1.0_dev_cleaned.jsonl", lines=True)
        dev_snli = pd.DataFrame({
            'text_a':dev_snli['sentence1'],
            'text_b':dev_snli['sentence2'],
            'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in dev_snli['gold_label']]})

        dev_mnli = pd.read_json("./datasets/multinli_1.0/multinli_1.0_dev_matched_cleaned.jsonl", lines=True)
        dev_mnli = pd.DataFrame({
            'text_a':dev_mnli['sentence1'],
            'text_b':dev_mnli['sentence2'],
            'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in dev_mnli['gold_label']]})

        dev = pd.DataFrame()
        train = pd.DataFrame()
        if 'snli' in train_sets:
            dev = dev.append(dev_snli)
            train = train.append(train_snli)
        if 'mnli' in train_sets:
            dev = dev.append(dev_mnli)
            train = train.append(train_mnli)
        if 'anli' in train_sets:
            dev = dev.append(dev_anli_r1.append(dev_anli_r2.append(dev_anli_r3)))
            train = train.append(train_anli_r1.append(train_anli_r2.append(train_anli_r3)))

        # you need to pass some combination of datasets
        assert len(train) != 0
        assert len(dev) != 0
        # shuffle the samples
        train = train.sample(frac=1).reset_index(drop=True)
        dev_snli = dev_snli.sample(frac=1).reset_index(drop=True)


    # Create a TransformerModel
    nopretrain = f'{"no" if args.pretrain == "False" else ""}'
    nofinetune = "no" if args.finetune == "no" else "" 
    trainsets = ('-' + args.finetune) if args.finetune != 'no' else ""

    model = ClassificationModel(
        args.model_name,
        args.model_type,
        num_labels=3,
        use_cuda=True,
        cuda_device=args.cuda_device,
        args=(
        {
        'output_dir':f'./checkpoint_results/{nofinetune}finetune/{path.basename(args.model_type)}-{nopretrain}pretrain-{nofinetune}finetune{trainsets}/',
        # 'output_dir':'./checkpoint_results/finetune/bert-base-cased-noanli_nopretrain',
        'overwrite_output_dir': True,
        'fp16': True, # uses apex
        'num_train_epochs': args.epochs,
        'reprocess_input_data': True,
        "learning_rate": 1e-5,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "max_seq_length": args.seq_len, #175
        "weight_decay": 0,
        "do_lower_case": False,
        "evaluate_during_training":True,
        "evaluate_during_training_verbose":True,
        "evaluate_during_training_steps":15000,
        "save_steps":15000,
        "n_gpu": args.num_gpus,
        "logging_steps":10,
        })
    )

    if args.finetune != 'no':
        model.train_model(train,eval_df=dev)
    
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(dev_diag)
    # save the results
    label_map = {'contradiction':0,'neutral':1,'entailment':2}
    for index, row in diagnostic.iterrows():
        prediction = np.argmax(model_outputs[index])
        if prediction == label_map[row['Label']]:
            diagnostic.at[index, 'NoFinetuneResults'] = 1
        else:
            diagnostic.at[index, 'NoFinetuneResults'] = 0



    diagnostic.to_csv(f"./checkpoint_results/{nofinetune}finetune/{path.basename(args.model_type)}-{nopretrain}pretrain-{nofinetune}finetune{trainsets}/diagnostic_results_{path.basename(args.model_type)}-{nopretrain}pretrain-{nofinetune}finetune{trainsets}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a GLUE diagnostic experiment using Transformers')
    parser.add_argument('--pretrain', action='store', type=str, required=True, help='If False, will not load a pretrained model. Else load a trained model.')
    parser.add_argument('--finetune', action='store', type=str, required=True, help='If no, won\'t finetune. Else must be each training set to use, separated by +')
    parser.add_argument('--model_name', action='store', type=str, required=True, help='The name of the class of model architectures (BERT, etc), must be supported by simpletransformers.')
    parser.add_argument('--model_type', action='store', type=str, required=True, help='The specific model to load, either huggingface id or path to local dir.')
    parser.add_argument('--epochs', action='store', type=int, default=5, help='Number of finetuning epochs, ignored if not finetuning.')
    parser.add_argument('--batch_size', action='store', type=int, default=16, help='Finetuning batch size')
    parser.add_argument('--seq_len', action='store', type=int, default=175, help='Max seq length for finetuning.')
    parser.add_argument('--cuda_device', action='store', type=int, required=True, help="Device to run experiment on.")
    parser.add_argument('--num_gpus', action='store', type=int, default=1, help='Number of gpus to use.')
    args = parser.parse_args()

    DiagTrial(args)
