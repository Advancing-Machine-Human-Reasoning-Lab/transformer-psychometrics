"""
    For training and evaluating the LSTM-based LMs
    See util.py for details on the valid command line arguments.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # must be set BEFORE importing torch
import time
import glob
import shutil
import json as js

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import SNLIClassifier
from util import get_args, makedirs

args = get_args()

hyperparameter_defaults = dict(
    epochs = args.epochs,
    batch_size = args.batch_size,
    d_embed = args.d_embed,
    d_proj = args.d_proj,
    d_hidden = args.d_hidden,
    n_layers = args.n_layers,
    log_every = args.log_every,
    lr = args.lr,
    dev_every = 1000,
    save_every = 1000,
    dp_ratio = args.dp_ratio,
    birnn = args.birnn,
    lower = args.lower,
    projection = args.projection,
    fix_emb = args.fix_emb,
    gpu = args.gpu,
    save_path = args.save_path,
    vector_cache = args.vector_cache,
    word_vectors = args.word_vectors,
    resume_snapshot = args.resume_snapshot,
    dry_run = args.dry_run,
    experiment = args.experiment, # TODO: always change to whatever you are doing
    # anli = args.anli,
    no_train = args.no_train
)



# prevent errors later when updating config

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')

inputs = data.Field(lower=args.lower, tokenize='spacy',tokenizer_language='en_core_web_sm')
answers = data.Field(sequential=False,tokenizer_language='en_core_web_sm')

# use the SNLI dataset class to process the combined train set
# this is fine since the datasets have the same format
if args.experiment == "snli":
    print("Using snli")
    train, dev, test = datasets.SNLI.splits(inputs, answers,root='./snli/snli/snli_1.0/',train='snli_1.0_train_cleaned.jsonl',validation='snli_1.0_dev_cleaned.jsonl',test='GLUE_Diag_nliLike.jsonl')
    inputs.build_vocab(train, dev, test)
    vector_cache = ".vector_cache/input_vectors_snli.pt"
elif args.experiment == "mnli":
    print("Using mnli")
    train, dev, test = datasets.SNLI.splits(inputs, answers,root='./snli/snli/snli_1.0/',train='multinli_1.0_train.jsonl',validation='multinli_1.0_dev_cleaned.jsonl',test='GLUE_Diag_nliLike.jsonl')
    inputs.build_vocab(train, dev, test)
    vector_cache = ".vector_cache/input_vectors_mnli.pt"

"""
    For the snli+mnli and snli+mnli+anli runs, you should combine the train
    and dev sets into a single jsonl file before running training
    In addition, the GLUE diagnositic csv should be converted to jsonl
"""
elif args.experiment == "snli+mnli":
    print("Using snli+mnli")
    train, dev, test = datasets.SNLI.splits(inputs, answers,root='./snli/snli/snli_1.0/',train='snli_mnli_train_combined.jsonl',validation='snli_mnli_dev_combined.jsonl',test='GLUE_Diag_nliLike.jsonl')
    inputs.build_vocab(train, dev, test)
    vector_cache = ".vector_cache/input_vectors_snli+mnli.pt"
elif args.experiment == "snli+mnli+anli":
    print("Using snli+mnli+anli")
    train, dev, test = datasets.SNLI.splits(inputs, answers,root='./snli/snli/snli_1.0/',train='snli_mnli_anli_train_combined.jsonl',validation='snli_mnli_anli_dev_combined.jsonl',test='GLUE_Diag_nliLike.jsonl')
    inputs.build_vocab(train, dev, test)
    vector_cache = ".vector_cache/input_vectors_snli+mnli+anli.pt"
else:
    print("ERROR: dataset partition is invalid!")
    exit()

inputs.build_vocab(train, dev, test)

if args.word_vectors:
    if os.path.isfile(vector_cache):
        inputs.vocab.vectors = torch.load(vector_cache)
    else:
        inputs.vocab.load_vectors(args.word_vectors)
        makedirs(os.path.dirname(vector_cache))
        torch.save(inputs.vocab.vectors, vector_cache)
    
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=args.batch_size, device=device)

# need a separate iterator for the test set
test_iter = data.Iterator(test,batch_size=1,device=device,train=False)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
# config.n_cells = config.n_layers

# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells = config.n_layers * 2
else:
    config.n_cells = config.n_layers

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=device)
else:
    model = SNLIClassifier(config)
    if args.word_vectors:
        model.embed.weight.data.copy_(inputs.vocab.vectors)
        model.to(device)

criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
train_log = open(f'./results/{args.experiment}-train-log-{args.epochs}-{args.lr}-{args.dp_ratio}-{args.n_layers}-{args.d_hidden}-{args.experiment}.txt','w+')
dev_log = open(f'./results/{args.experiment}-dev-log-{args.epochs}-{args.lr}-{args.dp_ratio}-{args.n_layers}-{args.d_hidden}-{args.experiment}.txt','w+')
makedirs(args.save_path)
print(header)

# for zero-shot evaluation
if args.no_train:
    for epoch in range(args.epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_iter):

            # switch model to training mode, clear gradient accumulators
            model.train(); opt.zero_grad()

            iterations += 1

            # forward pass
            answer = model(batch)

            # calculate accuracy of predictions in the current batch
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
            n_total += batch.batch_size
            train_acc = 100. * n_correct/n_total

            # calculate loss of the network output with respect to training labels
            loss = criterion(answer, batch.label)

            # backpropagate and update optimizer learning rate
            loss.backward(); opt.step()


            # evaluate performance on validation set periodically
            if iterations % args.dev_every == 0:

                # switch model to evaluation mode
                model.eval(); dev_iter.init_epoch()

                # calculate accuracy on validation set
                n_dev_correct, dev_loss = 0, 0
                with torch.no_grad():
                    for dev_batch_idx, dev_batch in enumerate(dev_iter):
                        answer = model(dev_batch)
                        n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()
                        dev_loss = criterion(answer, dev_batch.label)
                dev_acc = 100. * n_dev_correct / len(dev)

                print(dev_log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))
                
                dev_log.writelines(dev_log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))

                # update best valiation set accuracy
                if dev_acc > best_dev_acc:

                    # found a model with better validation set accuracy

                    best_dev_acc = dev_acc
                    snapshot_prefix = os.path.join(args.save_path, f'{args.experiment}-best_snapshot-{args.epochs}-{args.lr}-{args.dp_ratio}-{args.n_layers}-{args.d_hidden}')
                    snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

            elif iterations % args.log_every == 0:

                # print progress message
                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))
                train_log.writelines((log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12)))
            if args.dry_run:
                break

with torch.no_grad():
    diag = open('./snli/snli/snli_1.0/GLUE_Diag_nliLike.jsonl','r')
    test_f = open(f'./GLUE_Diag_nliLike-{args.epochs}-{args.lr}-{args.dp_ratio}-{args.n_layers}-{args.d_hidden}-{args.experiment}.jsonl','w+')
    for test_batch_idx, test_batch in enumerate(test_iter):
        answer = model(test_batch)
        is_test_correct = (torch.max(answer, 1)[1].view(test_batch.label.size()) == test_batch.label).sum().item()
        example = js.loads(diag.readline())
        example['correct'] = is_test_correct
        test_f.writelines(js.dumps(example))
        test_f.writelines("\n")

    diag.close()
    test_f.close()


train_log.close()
dev_log.close()
