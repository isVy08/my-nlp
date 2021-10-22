import torch, os, random
from tqdm import tqdm
from torchtext.data import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from utils import load, load_pickle, write_pickle
from torchtext.vocab import build_vocab_from_iterator
from tokenizers import CharBPETokenizer, Tokenizer


class SpaTorchTokenizer(object):
    
    def __init__(self, spacy_tokenizer, torch_vocab):
        self.vocab = torch_vocab
        self.tokenizer = spacy_tokenizer
    
    def encode(self, sequence):
        tokens_list = self.tokenizer(sequence)
        return self.vocab(tokens_list)
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def __str__(self):
        return self.__class__.__name__

def train_tokenizer(text_file, vocab_size, min_frequency, path_to_tokenizer, by='char'):
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] # to be compatible with BertTokenizer
    if by == 'char':
        tokenizer = CharBPETokenizer()                            
        tokenizer.train(text_file, special_tokens=special_tokens, 
                        vocab_size=vocab_size, min_frequency=min_frequency)
        tokenizer.enable_padding()
        tokenizer.save(path_to_tokenizer)
    
    elif by == 'token':
        spacy_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        if isinstance(text_file, str):
            text_file = [text_file]
        data = []
        for file in text_file:
            data.extend(load(file))
        data_iter = map(spacy_tokenizer, data)
        vocab = build_vocab_from_iterator(data_iter, min_freq=min_frequency, 
                                          specials=special_tokens, special_first=True)
        vocab.set_default_index(1)
        tokenizer = SpaTorchTokenizer(spacy_tokenizer, vocab)
        
        write_pickle(tokenizer, path_to_tokenizer)
        
    return tokenizer

def to_tokenizer(tokenizer_type, path_to_tokenizer):
    
    if tokenizer_type == 'bert':
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained("bert-base-cased")
    
    elif tokenizer_type == 'char':
        tokenizer = Tokenizer.from_file(path_to_tokenizer)
        tokenizer.enable_padding()
        return tokenizer
    
    elif tokenizer_type == 'token':
        return load_pickle(path_to_tokenizer)

class DataGenerator(IterableDataset):
    

    def __init__(self, path_to_input, source_filename, target_filename, split, 
                 path_to_tokenizer=None, tokenizer_type='char', 
                 vocab_size=None, min_frequency=None, data_fraction=1.0):
        super(DataGenerator).__init__()

        self.split = split
        self.tokenizer_type = tokenizer_type


        
        # Tokenizer:

        if os.path.isfile(path_to_tokenizer):
            self.tokenizer = to_tokenizer(tokenizer_type, path_to_tokenizer)
                
        else: 
            vocab_size = 20000 if vocab_size is None else vocab_size
            self.min_frequency = 5 if min_frequency is None else min_frequency
        
            self.tokenizer = train_tokenizer([source_filename, target_filename], 
                                             vocab_size, min_frequency, 
                                             path_to_tokenizer, tokenizer_type)
        
        if self.tokenizer_type == 'bert':
            self.vocab_size = self.tokenizer.vocab_size
            self.bos_idx, self.eos_idx, self.pad_idx, self.unk_idx = 101, 102, 100, 0
        else:
            self.vocab_size = self.tokenizer.get_vocab_size()
            self.bos_idx, self.eos_idx, self.pad_idx, self.unk_idx = 2, 3, 0, 1


        if os.path.isfile(path_to_input):
            self.source_input, self.target_input = load_pickle(path_to_input)


        else: 
            source_data, target_data = self._load(source_filename), self._load(target_filename)
            assert len(source_data) == len(target_data)
            self.source_input = self._transform(source_data)
            self.target_input = self._transform(target_data)
            write_pickle((self.source_input, self.target_input), path_to_input)
        
        if data_fraction < 1.0:
            n = self.source_input.shape[0]
            k = int(data_fraction * n)
            indices = random.sample(range(n), k) 
            self.source_input = self.source_input[indices, :]
            self.target_input = self.target_input[indices, :] 
    
    def _load(self, filename):
        def _preprocess(text):
            return text.strip().lower()

        data = load(filename)
        return [_preprocess(text) for text in data]
    
    def _transform(self, data):
        if self.tokenizer_type == 'bert':
            return self.tokenizer(data, padding=True, return_tensors='pt').input_ids
        
        else:
            print('Tokenizing data ...')
            _input = []
            for text in tqdm(data):
                if self.tokenizer_type == 'char':
                    item = [2] + self.tokenizer.encode(text).ids + [3] 
                else:
                    item = [2] + self.tokenizer.encode(text) + [3]
                _input.append(torch.tensor(item))
            _input = pad_sequence(_input, batch_first=True, padding_value=self.pad_idx) 
            print('Finish tokenization!')
            return _input

    def __iter__(self):
        
        return zip(self.source_input, self.target_input)
    
    def __len__(self):
        return self.source_input.shape[0]