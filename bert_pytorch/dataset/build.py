import os
import sys
from unittest.mock import MagicMock
sys.path.append('./')
from bert_pytorch.dataset.vocab import WordVocab



def build():

    args = MagicMock()
    args.corpus_path = 'data/corpus.small'
    args.output_path = 'data/vocab.small'
    args.vocab_size = 1000
    args.encoding = 'utf-8'
    args.min_freq = 1
    

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)


if __name__ == '__main__':
    build()