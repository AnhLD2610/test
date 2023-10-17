"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import MyDataLoader
from util.tokenizer import Tokenizer
import torchtext.transforms as T

tokenizer = Tokenizer()
loader = MyDataLoader(
                    tokenize=tokenizer,
                    init_token='<sos>',
                    eos_token='<eos>')
path = '/media/data/thanhnb/test/data/C#'
train, valid, test = loader.make_dataset(path)
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size)
source_vocab, vocab_size = loader.build_vocab(train_iter=train_iter, min_freq=10)

# gọi get_transform 
text_tranform = loader.getTransform(source_vocab)
# print(source_vocab)
# print(type(source_vocab))
src_pad_idx = source_vocab.get_stoi()['<pad>']
trg_pad_idx = source_vocab.get_stoi()['<pad>']
trg_sos_idx = source_vocab.get_stoi()['<sos>']

enc_voc_size = vocab_size
dec_voc_size = vocab_size 
