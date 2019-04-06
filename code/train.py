import torch
import torch.optim as optim
import torch.nn as nn
import time
import os, sys
import glob
import numpy as np
import logging
from args import get_args
from models import RNNSubjectRecognition
from DataLoader import SubjectRecognitionLoader, load_pretrained_vectors
from predict import evaluation


def set_logger(name):
    '''
    Write logs to checkpoint and console
    '''

    log_file = './log/%s.log' % name

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


args = get_args()
set_logger("subject_recognition")
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
    device = torch.device("cpu")
else:
    device = torch.device("cuda:%d" % args.gpu)

if torch.cuda.is_available() and args.cuda:
    logging.info("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    logging.info("Warning: You have Cuda but do not use it. You are using CPU for training")

train_loader = SubjectRecognitionLoader(args.train_file, args.gpu)
logging.info('load train data, batch_num: %d\tbatch_size: %d'
      %(train_loader.batch_num, train_loader.batch_size))
valid_loader = SubjectRecognitionLoader(args.valid_file, args.gpu)
logging.info('load valid data, batch_num: %d\tbatch_size: %d'
      %(valid_loader.batch_num, valid_loader.batch_size))

# load word vocab for questions
word_vocab = torch.load(args.vocab_file)
logging.info('load word vocab, size: %s' % len(word_vocab))

os.makedirs(args.save_path, exist_ok=True)

args.n_out = 2
args.n_cells = args.n_layers
if args.birnn:
    args.n_cells *= 2

model = RNNSubjectRecognition(len(word_vocab), args)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        pretrained = torch.load(args.vector_cache)
    else:
        pretrained = load_pretrained_vectors(args.word_vectors, binary=False,
                                             normalize=args.word_normalize)
        torch.save(pretrained, args.vector_cache)

    model.embed.weight.data.copy_(pretrained)
    logging.info('load pretrained word vectors from %s, pretrained size: %s' % (args.word_vectors,
                                                                         pretrained.size()))

model.to(device)
for name, param in model.named_parameters():
    print(name, param.size())
criterion = nn.NLLLoss() # negative log likelyhood loss function
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# train the model
iterations = 0
best_dev_acc = 0
best_dev_F = 0
num_iters_in_epoch = train_loader.batch_num
patience = args.patience * num_iters_in_epoch
iters_not_improved = 0
early_stop = False
train_loss = 0
snapshot_path = "subject_recognition.pth"
best_snapshot_path = "best_subject_recognition.pth"

for epoch in range(1, args.epochs+1):
    if early_stop:
        logging.info("Early stopping. Epoch: {}, Best Dev. Acc: {}".format(epoch, best_dev_acc))
        break

    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_loader.next_batch()):
        # seq: (batch_size, len)
        # seq_len: (batch_size)
        # label: (batch_size, len)
        iterations += 1
        seq, seq_len, label = batch
        model.train()
        optimizer.zero_grad()
        scores = model.forward(seq, seq_len)   # scores :(batch_size, length, 2)->(batch_size * length, 2)

        n_correct += ((torch.max(scores, dim=1)[1].view(label.size()).data == label.data).sum(dim=1) \
                      == label.size()[1]).sum()
        n_total += train_loader.batch_size
        train_acc = 100. * n_correct / n_total

        loss = criterion(scores, label.view(-1, 1)[:, 0])
        train_loss += loss.data.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        if iterations % args.save_every == 0:
            torch.save(model, snapshot_path)

        if iterations % args.dev_every == 0:
            valid_loss = 0
            model.eval()
            n_dev_correct = 0
            gold_list = []
            pred_list = []
            for valid_batch_idx, valid_batch in enumerate(valid_loader.next_batch()):
                seq, seq_len, valid_label = batch
                answer = model.forward(seq, seq_len)
                n_dev_correct += ((torch.max(answer, 1)[1].view(valid_label.size()).data == \
                                   valid_label.data).sum(dim=1) == valid_label.size()[1]).sum()
                valid_loss += criterion(answer, valid_label.view(-1, 1)[:, 0]).data.item()
                index_tag = np.transpose(torch.max(answer, 1)[1].view(valid_label.size()).cpu().data.numpy())
                gold_list.append(np.transpose(valid_label.cpu().data.numpy()))
                pred_list.append(index_tag)
            P, R, F = evaluation(gold_list, pred_list)
            dev_acc = 100. * n_dev_correct / (valid_loader.batch_num * valid_loader.batch_size)
            logging.info("-----------------------------")
            logging.info("iterations %d" % (iterations))
            logging.info("train loss is %f" % train_loss)
            logging.info("valid loss is %f" % valid_loss)
            logging.info("valid acc is %f" % dev_acc)
            logging.info("train acc is %f" % train_acc)
            logging.info("P is %f" % P)
            logging.info("R is %f" % R)
            logging.info("F is %f" % F)
            train_loss = 0
            if F > best_dev_F:
                best_dev_F = F
                iters_not_improved = 0
                torch.save(model, best_snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved > patience:
                    early_stop = True
                    break