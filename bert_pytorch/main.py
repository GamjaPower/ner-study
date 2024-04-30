import argparse
from unittest.mock import MagicMock

from torch.utils.data import DataLoader
import sys
sys.path.append('./')

from bert_pytorch.dataset.dataset import BERTDataset
from bert_pytorch.dataset.vocab import WordVocab
from bert_pytorch.model.bert import BERT
from bert_pytorch.trainer.pretrain import BERTTrainer




def train():

    args = MagicMock()
    args.vocab_path = 'data/vocab.small'
    args.train_dataset = 'data/corpus.small'
    args.test_dataset = 'data/corpus.small'
    args.output_path = 'output/bert.model'
    args.hidden = 256
    args.layers = 8
    args.attn_heads = 8
    args.seq_len = 20
    args.batch_size = 64
    args.epochs = 10
    args.num_workers = 5
    args.with_cuda = True
    args.log_freq = 10
    args.corpus_lines = None
    args.cuda_devices = None
    args.on_memory = True
    args.lr = 1e-3
    args.adam_weight_decay = 0.01
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == "__main__":
    train()