"""
    For evaluating T5 only
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # must be set BEFORE importing torch
import argparse
import glob
import json
import time
import logging
import random

import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch import multiprocessing
import pytorch_lightning as pl


from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel, AutoModelForSeq2SeqLM
from numpy import argmax
from math import exp

logging.basicConfig(level=logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.getLogger("tensorflow").setLevel(logging.FATAL)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, anli):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.anli = anli

    def is_logger(self):
        # return self.trainer.global_rank <= 0
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"].detach().clone()
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"], attention_mask=batch["source_mask"], labels=labels, decoder_attention_mask=batch["target_mask"])
        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict
    
    # def get_dataset(tokenizer, type_path, args, train):
    # return NLIDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length,anli=args.anli,train=train)

    def train_dataloader(self):
        self.configure_optimizers()
        args = self.hparams
        train_dataset = NLIDataset(tokenizer=self.tokenizer, data_dir=args.data_dir, type_path="/train", max_len=args.max_seq_length, anli=args.anli, train=True)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        if type(self.hparams.n_gpu) == int:
            t_total = (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu))) // self.hparams.gradient_accumulation_steps * float(self.hparams.num_train_epochs)
        else:
            t_total = (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu)))) // self.hparams.gradient_accumulation_steps * float(self.hparams.num_train_epochs)
        scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        args = self.hparams
        val_dataset = NLIDataset(tokenizer=self.tokenizer, data_dir=args.data_dir, type_path="/val", max_len=args.max_seq_length, anli=args.anli, train=False)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))




class NLIDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=256, anli=False, train=True):
        self.path = os.path.join(data_dir, type_path + ".jsonl")

        self.premise_column = "text_a"
        self.hypothesis_column = "text_b"
        self.target_column = "labels"
        self.train = train
        self.anli = anli

        data_dir = data_dir+type_path
        
        if self.train:
            if self.anli:
                train_snli = pd.read_json(data_dir+"/snli_1.0_train_cleaned.jsonl", lines=True)
                train_snli = pd.DataFrame({
                    'text_a':train_snli['sentence1'],
                    'text_b':train_snli['sentence2'],
                    'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in train_snli['gold_label']]})
                train_mnli = pd.read_json(data_dir+"/multinli_1.0_train.jsonl", lines=True)
                train_mnli = pd.DataFrame({
                    'text_a':train_mnli['sentence1'],
                    'text_b':train_mnli['sentence2'],
                    'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in train_mnli['gold_label']]})
                
                train_anli_r1 = pd.read_json(data_dir+"/R1/train.jsonl", lines=True)
                train_anli_r2 = pd.read_json(data_dir+"/R2/train.jsonl", lines=True)
                train_anli_r3 = pd.read_json(data_dir+"/R3/train.jsonl", lines=True)
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
                
                self.data = train_snli.append(train_mnli.append(train_anli_r3.append(train_anli_r2.append(train_anli_r1))))

            else:
                train_snli = pd.read_json(data_dir+"/snli_1.0_train_cleaned.jsonl", lines=True)
                train_snli = pd.DataFrame({
                    'text_a':train_snli['sentence1'],
                    'text_b':train_snli['sentence2'],
                    'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in train_snli['gold_label']]})
                train_mnli = pd.read_json(data_dir+"/multinli_1.0_train.jsonl", lines=True)
                train_mnli = pd.DataFrame({
                    'text_a':train_mnli['sentence1'],
                    'text_b':train_mnli['sentence2'],
                    'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in train_mnli['gold_label']]})
                
                self.data = train_snli.append(train_mnli)
                self.data = train_snli
        else:
            if self.anli:
                dev_snli = pd.read_json(data_dir+"/snli_1.0_dev_cleaned.jsonl", lines=True)
                dev_snli = pd.DataFrame({
                    'text_a':dev_snli['sentence1'],
                    'text_b':dev_snli['sentence2'],
                    'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in dev_snli['gold_label']]})

                dev_mnli = pd.read_json(data_dir+"/multinli_1.0_dev_matched_cleaned.jsonl", lines=True)
                dev_mnli = pd.DataFrame({
                    'text_a':dev_mnli['sentence1'],
                    'text_b':dev_mnli['sentence2'],
                    'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in dev_mnli['gold_label']]})
                
                dev_anli_r1 = pd.read_json(data_dir+"/R1/dev.jsonl", lines=True)
                dev_anli_r2 = pd.read_json(data_dir+"/R2/dev.jsonl", lines=True)
                dev_anli_r3 = pd.read_json(data_dir+"/R3/dev.jsonl", lines=True)
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
                
                self.data = dev_snli.append(dev_mnli.append(dev_anli_r1.append(dev_anli_r2.append(dev_anli_r3))))
            
            else:
                dev_snli = pd.read_json(data_dir+"/snli_1.0_dev_cleaned.jsonl", lines=True)
                dev_snli = pd.DataFrame({
                    'text_a':dev_snli['sentence1'],
                    'text_b':dev_snli['sentence2'],
                    'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in dev_snli['gold_label']]})

                dev_mnli = pd.read_json(data_dir+"/multinli_1.0_dev_matched_cleaned.jsonl", lines=True)
                dev_mnli = pd.DataFrame({
                    'text_a':dev_mnli['sentence1'],
                    'text_b':dev_mnli['sentence2'],
                    'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in dev_mnli['gold_label']]})
                
                self.data = dev_snli.append(dev_mnli)
                self.data = dev_snli

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for index,row in self.data.iterrows():
            premise, hypothesis, target = row[self.premise_column], row[self.hypothesis_column], row[self.target_column]

            input_ = "mnli premise: " + str(premise) + " hypothesis: " + str(hypothesis)
            target = str(target)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus([input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt")
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus([target], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt")

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)






if __name__ == "__main__":

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    args_dict = dict(
        data_dir="./",  # path for data files FOR EVAL
        # data_dir = "./datasets", # path for data files FOR TRAIN
        output_dir="./checkpoint_results/finetune/t5-base-pretrain-finetune-snli/",  # path to save the checkpoints
        model_name_or_path="./checkpoint_results/finetune/t5-base-pretrain-finetune-snli/",
        # model_name_or_path="t5-base", # FOR TRAINING, NOT TESTING
        tokenizer_name_or_path="t5-base",
        max_seq_length=175,
        learning_rate=1e-5,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=1,
        eval_batch_size=1, # set to 1 for test set, 20 for dev set
        num_train_epochs=5,
        gradient_accumulation_steps=1,
        n_gpu=[0],
        # early_stopping_callback=False,
        fp_16=True,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level="O1",  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=0.5,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        anli=False,
        eval=True,
        fp16=True,

    )


    args = argparse.Namespace(**args_dict)
    print(args_dict)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5)

    train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            max_epochs=args.num_train_epochs,
            # early_stopping_callback=False,
            precision=16 if args.fp_16 else 32,
            amp_level=args.opt_level,
            gradient_clip_val=args.max_grad_norm,
            checkpoint_callback=checkpoint_callback,
            callbacks=[LoggingCallback()],
        )

    if args.eval == False:
        print("Beginning Training")

        train_path = "/train"
        val_path = "/val"

        if not os.path.exists("./checkpoint_results/finetune/t5-base-pretrain-finetune-snli"):
            os.makedirs("./checkpoint_results/finetune/t5-base-pretrain-finetune-snli")

        print("Initialize model")
        model = T5FineTuner(args,anli=False)

        trainer = pl.Trainer(**train_params,plugins='deepspeed', accelerator='dp') # accelerator=dp is for multi-gpus

        print(" Training model")
        trainer.fit(model)

        print("training finished")

        print("Saving model")
        model.model.save_pretrained("./checkpoint_results/finetune/t5-base-pretrain-finetune-snli")

        print("Saved model")

    else:
        print("Beginning evaluation")
        cuda = torch.device('cuda')

        diagnostic = pd.read_csv(args.data_dir+"/BERT_Diagnostic_Set.csv")
        diagnostic = pd.DataFrame({
            'index':diagnostic['ID'],
            'text_a':diagnostic['Premise'],
            'text_b':diagnostic['Hypothesis'],
            'labels':[['contradiction', 'neutral', 'entailment'].index(gs) for gs in diagnostic['Label']]})
        
        # t5_diag = pd.DataFrame(columns=["prefix","input_text","target_text"])
        # for index,row in diagnostic.iterrows():
            # t5_diag = t5_diag.append({"prefix":,"input_text":f"{row['text_a']}  {row['text_b']}","target_text":row['labels']},ignore_index=True)
        
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model.to(cuda)
        model.eval()

        # result = model.eval_model(t5_diag)
        # save the results
        with torch.no_grad():
            for index, row in diagnostic.iterrows():
                # example = tokenizer(, return_tensors='pt').input_ids
                tokenized_inputs = tokenizer.batch_encode_plus([f"mnli premise: {row['text_a']} hypothesis: {row['text_b']}"], max_length=175, pad_to_max_length=True, return_tensors="pt").to(cuda)
                # tokenized_targets =tokenizer.batch_encode_plus([str(row['labels'])], max_length=175, return_tensors="pt").to(cuda)
                prediction = model.generate(input_ids=tokenized_inputs.input_ids,top_k=50,max_length=3)
                dec = [tokenizer.decode(ids) for ids in prediction]
                # if there is padding get rid of it
                dec = dec[0].replace("<pad>","").replace("</s>","").replace("<s>","").strip()
                if int(dec) == row['labels']:
                    diagnostic.at[index, 'NoFinetuneResults'] = 1
                else:
                    diagnostic.at[index, 'NoFinetuneResults'] = 0
        



        # preds = open("preds.txt", "a")
        # for prediction in wrong_predictions:
        #     preds.write("ID: " + str(prediction.index) + "\n" + "label: " + str(prediction.label) + "\n\n\n")
        # diagnostic_new = diagnostic.assign(NoFinetuneResults = model_outputs)
        diagnostic.to_csv(f"./checkpoint_results/finetune/t5-base-pretrain-finetune-snli/diagnostic_results-t5-base-pretrain-finetune-snli.csv")