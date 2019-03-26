import collections
import codecs
import re
import time
import token
import os
import pickle
import random
import json
import timeit
import numpy as np
import subprocess
import sys
from flair.data import TaggedCorpus
from flair.data_fetcher import  NLPTaskDataFetcher
from typing import List



def get_corpus_vocab(config):
    count_token = {} 
    count_label = {} 
    count_character = {}
    

    datasets = [('train',config.path_train), ('eval', config.path_eval), ('test', config.path_test)]
    for dataset in datasets:
        count_token[dataset[0]], count_label[dataset[0]], count_character[dataset[0]] = get_vocabs(dataset[1], separator = config.separator, lowercase = config.lowercase)

    vocab_token_corpus = count_token['train'] + count_token['eval'] + count_token['test']
    vocab_label = count_label['train'] + count_label['eval'] + count_label['test']
    vocab_char = count_character['train'] + count_character['eval'] + count_character['test']

    # sorted the vocabu by frequency 
    vocab_token_corpus = [x[0] for x in vocab_token_corpus.most_common()]
    vocab_label = [x[0] for x in vocab_label.most_common()]
    vocab_char = [x[0] for x in vocab_char.most_common()]

    # future features: limit the vocabulary by threshold
    ###############################################
    # if config.vocabulary_threshold > 1:
    #     vocab_token_corpus = 
    ###############################################

    return vocab_token_corpus, vocab_label, vocab_char




def get_inputs(dataset, token2idx, char2idx, label2idx, config):

    dataset_filepath = None
    if dataset == 'train':
        dataset_filepath = config.path_train
    elif dataset == 'eval':
        dataset_filepath = config.path_eval
    elif dataset == 'test':
        dataset_filepath = config.path_test
    else:
        print("unknown dataset: ", dataset)


    separator = config.separator
    lowercase = config.lowercase
    
    # collection per sentence
    # format [[[char_idxs], word_idx], ...]
    sentence_token = []
    # format [[label], ...]
    sentence_label = []
    
    # format [[sentence1_token], [sentence2_token], ...]
    tokens = []
    # format [[sentence1_label], [sentence2_label], ...]
    labels = []

    # go throught whole CoNLL file
    f = codecs.open(dataset_filepath, 'r', 'UTF-8')
    for line in f:
        line = line.strip().split(separator)
        # encouter a new sentence
        if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
            if len(sentence_token) > 0:
                labels.append(sentence_label)
                tokens.append(sentence_token)
                sentence_label = []
                sentence_token = []
            continue
                
        token = str(line[0])
        label = str(line[-1])    
        # 1. preprocess word
        if lowercase:
            word = token.lower()
        else:
            word = token
        # don't use NUM
#         if word.isdigit():
#             word = NUM

        # char idxs
        char_idxs = []
        for char in word:
            if char in char2idx:
                char_idxs += [char2idx[char]]  
            else:
                print("encounter UNK char:", char)
        
        # word idx
        if word in token2idx:
            word_idx = token2idx[word]
        else:
            word_idx = token2idx['$UNK$']

        
        # label idx
        if label in label2idx:
            label_idx = label2idx[label]
        else:
            print("encounter UNK label:", label)
            
        sentence_token.append((char_idxs, word_idx))
        sentence_label.append(label_idx)

    if len(sentence_token) > 0:
        tokens.append(sentence_token)
        labels.append(sentence_label)

    f.close()
    
    return tokens, labels





def get_idx(config):

    start = timeit.default_timer()
    print("Building vocab...")

    # collect the vocab in corpus
    vocab_token_corpus, vocab_label, vocab_char = get_corpus_vocab(config)


    # collect the vocab in pre-trained embedding file
    if config.embedding_type == 'glove':
        # filename_glove = '../data/glove/glove.6B.100d.txt'
        vocab_glove = get_glove_vocab(config.glove_file_path)
        # selected only common vocabs in corpus and pre-trained embedding(like glove)
        vocab_token_final = [token for token in vocab_token_corpus if token.strip() in vocab_glove]
        vocab_token_final = ['$UNK$'] + vocab_token_final
    elif config.embedding_type == 'fasttext':
        result = get_fasttext_embedding(vocab_token_corpus, config.command, config.option,
                               config.bin_file, config.vocab_file, config.fasttext_embedding_file)
        if not result :
            print('fail to intial lookup_table, exit')
            sys.exit()        
    elif config.embedding_type == 'w2v':
        vocab_w2v = config.w2v.vocab
        # selected only common vocabs in corpus and pre-trained embedding(like glove)
        vocab_token_final = [token for token in vocab_token_corpus if token.strip() in vocab_w2v]
        vocab_token_final = ['$UNK$'] + vocab_token_final
    else: # contextual embedding case, do nothing 
        pass
        
        
               
    # generate token2index mapping dict for token, char, label
    if config.embedding_type not in ['w2v', 'glove']:
        vocab_token_final = vocab_token_corpus     
        
    token2idx = get_2idx(vocab_token_final, config.save_idx, config.file_token_idx)
    char2idx = get_2idx(vocab_char, config.save_idx, config.file_char_idx)
    label2idx = get_2idx(vocab_label, config.save_idx, config.file_label_idx)
    # path = '../data/idx/'
    # save_idx = True
    # paths = ['../data/idx/token2idx.json', '../data/idx/label2idx.json', '../data/idx/tag2idx.json']


    # get embedding lookup table
    if config.embedding_type == 'glove':
        lookup_table = get_embedding_lookup_table(token2idx, config.glove_file_path, config.dim_word, config.save_table, config.lookup_table_file_path)
    elif config.embedding_type == 'fasttext':
        lookup_table = get_embedding_lookup_table(token2idx, config.fasttext_embedding_file, config.dim_word, config.save_table, config.lookup_table_file_path)
    elif config.embedding_type == 'w2v':
        lookup_table = get_embedding_lookup_table_word2vec(token2idx, config.w2v, config.dim_word, config.save_table, config.lookup_table_file_path)


    stop = timeit.default_timer()
    print("vocabulary for this corpus: {} tokens, {} chars, {} labels"
          .format(len(vocab_token_final), len(vocab_char),len(vocab_label)))
    print('vocabulary construction time: ', stop - start) 


    # update config
    config.set_n_label(len(vocab_label))
    config.set_n_word(len(vocab_token_final))
    config.set_n_char(len(vocab_char))
    if config.embedding_type in ['w2v', 'glove', 'fasttext']:
        config.set_lookup_table(lookup_table)
    config.set_idx2label(label2idx)
    config.set_idx2token(token2idx)

    # save index version
    if config.save_idx :
        with open(config.indx_config, 'wb') as f:
            pickle.dump(len(vocab_label), f)
            pickle.dump(len(vocab_char), f)
            pickle.dump(len(vocab_token_final), f)
            pickle.dump(token2idx, f)
            pickle.dump(char2idx, f)
            pickle.dump(label2idx, f)
            pickle.dump(config.idx2label, f)
            pickle.dump(config.idx2token, f)

    if config.embedding_type in ['w2v', 'glove', 'fasttext']:
        return token2idx, char2idx, label2idx, lookup_table
    else:
        return token2idx, char2idx, label2idx





def get_fasttext_embedding(vocab_token_corpus, command, option,
                           path_bin_file, path_vocab_file, path_fasttext_embedding_file):
    with open(path_vocab_file, 'w+') as vocab_file:
        for i, token in enumerate(vocab_token_corpus):
            vocab_file.write(token)
            if i < len(vocab_token_corpus) - 1:
                vocab_file.write('\n')
    with open(path_vocab_file, 'r') as stdin_file,open(path_fasttext_embedding_file, 'w+') as stdout_file: 
        result = subprocess.call([command, option, path_bin_file], stdin = stdin_file, stdout = stdout_file)
    
    if result == 0:
        print('fasttext_lookup_table built')
        return True
    else:
        print('fail to build fasttext_lookup_table')
        return False


def get_vocabs(filepath, separator = ' ', lowercase = True):
    
    count_token = collections.Counter()
    count_label = collections.Counter()
    count_character = collections.Counter()
    
    if filepath:
        f = codecs.open(filepath, 'r', 'UTF-8')
        for line in f:
            line = line.strip().split(separator)

            #skip sentence separator
            if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
                continue
            
            token = str(line[0])
            for character in token:
                count_character.update({character: 1})
                
            # lowercase & digit
            if lowercase:
                token = str(line[0]).lower()
            else:
                token = str(line[0])
                
            # use the digit in pretrained embedding
#             if token.isdigit():
#                 token = '$NUM$'
                
            label = str(line[-1])
            count_token.update({token: 1})
            count_label.update({label: 1})              
        
        f.close()    
            
    return count_token, count_label, count_character


def get_glove_vocab(filename):
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    return vocab

def get_embedding_lookup_table_word2vec(vocab, w2v, dim = 300, save_table = False, file_path = None):

    lookup_table = np.zeros([len(vocab), dim])
    for w in list(vocab.keys()):
        if w in w2v:
            word_idx = vocab[w]
            embedding = [float(x) for x in w2v[w]]
            lookup_table[word_idx] = np.asarray(embedding)
                
    # save lookup table
    if save_table:
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise                    
        with open(file_path, 'w+') as fp:
            np.savez_compressed(file_path, lookup_table=lookup_table)
            
    return lookup_table


def get_embedding_lookup_table(vocab, glove_filename, dim = 100, save_table = False, file_path = None):

    lookup_table = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = []
            if word in vocab:
                embedding = [float(x) for x in line[1:]]
                word_idx = vocab[word]
                lookup_table[word_idx] = np.asarray(embedding)
                
    # save lookup table
    if save_table:
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise                    
        with open(file_path, 'w+') as fp:
            np.savez_compressed(file_path, lookup_table=lookup_table)
            
    return lookup_table



def get_2idx(vocabu, save_idx = False, file_path = None):
    dictionary = dict()
    for idx, word in enumerate(vocabu):
        word = word.strip()
        dictionary[word] = idx
        
    # save index    
    if save_idx:
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise                    
        with open(file_path, 'w+') as fp:
            json.dump(dictionary, fp, indent=4)
            
    return dictionary


def get_inputs(dataset, token2idx, char2idx, label2idx, config):

    dataset_filepath = None
    if dataset == 'train':
        dataset_filepath = config.path_train
    elif dataset == 'eval':
        dataset_filepath = config.path_eval
    elif dataset == 'test':
        dataset_filepath = config.path_test
    else:
        print("unknown dataset: ", dataset)


    separator = config.separator
    lowercase = config.lowercase
    
    # collection per sentence
    # format [[[char_idxs], word_idx], ...]
    sentence_token = []
    # format [[label], ...]
    sentence_label = []
    
    # format [[sentence1_token], [sentence2_token], ...]
    tokens = []
    # format [[sentence1_label], [sentence2_label], ...]
    labels = []

    # go throught whole CoNLL file
    f = codecs.open(dataset_filepath, 'r', 'UTF-8')
    for line in f:
        line = line.strip().split(separator)
        # encouter a new sentence
        if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
            if len(sentence_token) > 0:
                labels.append(sentence_label)
                tokens.append(sentence_token)
                sentence_label = []
                sentence_token = []
            continue
                
        token = str(line[0])
        label = str(line[-1])    
        # 1. preprocess word
        if lowercase:
            word = token.lower()
        else:
            word = token
        # don't use NUM
#         if word.isdigit():
#             word = NUM

        # char idxs
        char_idxs = []
        for char in word:
            if char in char2idx:
                char_idxs += [char2idx[char]]  
            else:
                print("encounter UNK char:", char)
        
        # word idx
        if word in token2idx:
            word_idx = token2idx[word]
        else:
            word_idx = token2idx['$UNK$']

        
        # label idx
        if label in label2idx:
            label_idx = label2idx[label]
        else:
            print("encounter UNK label:", label)
            
        sentence_token.append((char_idxs, word_idx))
        sentence_label.append(label_idx)

    if len(sentence_token) > 0:
        tokens.append(sentence_token)
        labels.append(sentence_label)

    f.close()
    
    return tokens, labels

def get_chunk_type(tok, idx_to_tag):

    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, idx_to_tag):

    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if idx_to_tag[tok] == 'O' and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif idx_to_tag[tok] != 'O':
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def next_batch(tokens, labels, batch_size = 1, shuffle = True):
    # shuffle the data at the beginning of each epoch
    if shuffle :
        data = np.array([[tokens[i], labels[i]] for i in range(len(tokens))])
        np.random.shuffle(data)
        tokens, labels = zip(*data)


    
    # generate mini batches
    for i in np.arange(0, len(tokens), batch_size):
        offset = min(i+batch_size, len(tokens))
        yield (tokens[i:offset], labels[i:offset])

def pad_sentence(batch_setence):
    
    # find the max_length
    max_length = max(map(lambda x : len(x), batch_setence))
    
    # padding
    sequence_padded = []
    sequence_length = []
    for seq in batch_setence:
        seq = list(seq)
        seq_ = seq[:max_length] + [0]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_word(batch_setence_word):
    '''
    https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
    '''
    max_length_word = max([max(map(lambda x: len(x), seq))
                           for seq in batch_setence_word])
    sequence_padded, sequence_length = [], []
    for seq in batch_setence_word:
        # all words are same length now
        sp, sl = _pad_sequences(seq, 0, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]

    max_length_sentence = max(map(lambda x : len(x), batch_setence_word))
    sequence_padded, _ = _pad_sequences(sequence_padded,
            [0]*max_length_word, max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0,
            max_length_sentence)

    return sequence_padded, sequence_length

def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length



def get_cropus_len(dataset_filepaths):
    totall_len = 0
    for file in dataset_filepaths: 
        totall_len += get_inputs_len(file)
    return totall_len

def get_inputs_len(dataset_filepath):
    
    sentence_token = []
    # format [[label], ...]
    sentence_label = []
    
    # format [[sentence1_token], [sentence2_token], ...]
    tokens = []
    # format [[sentence1_label], [sentence2_label], ...]
    labels = []    
    
    number_of_token = 0

    # go throught whole CoNLL file
    f = codecs.open(dataset_filepath, 'r', 'UTF-8')
    for line in f:
        line = line.strip().split(' ')
        # encouter a new sentence
        if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
            if len(sentence_token) > 0:
                labels.append(sentence_label)
                tokens.append(sentence_token)
                sentence_label = []
                sentence_token = []
            continue
                
        token = str(line[0])
        label = str(line[-1])
        number_of_token += 1


    f.close()
    
    return number_of_token



def load_cropus(config):
    '''
    this function load the cropus to flair library : https://github.com/zalandoresearch/flair
    the orgnization of data files required can be find in the above link
    ''' 
    # the 3rd column should avoid named as 'ner', otherwise it will be convert into BIOES format by flair library
    columns = {0: 'text', 1: 'pos', 2: 'np', 3: 'ner11'}
    data_folder = config.path_data_root
    # retrieve corpus using column format, data folder and the names of the train, dev and test files
    corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                                #train_file='eng.train',
                                                                  train_file='eng.testb',
                                                                  test_file='eng.testb',
                                                                  dev_file='eng.testa')
    
    # skip the document separator in the CONLL cropus
    filtered_train = list(filter(lambda x: x.to_tokenized_string() != '-DOCSTART-', corpus.train))
    filtered_dev = list(filter(lambda x: x.to_tokenized_string() != '-DOCSTART-', corpus.dev))
    filtered_test = list(filter(lambda x: x.to_tokenized_string() != '-DOCSTART-', corpus.test))
    
    return filtered_train, filtered_dev, filtered_test


def get_inputs_contextual(corpus,stacked_embeddings, offset, lookup_table,token2idx, char2idx, label2idx):
    
    # collection per sentence
    # format [[[char_idxs], word_idx], ...]
    sentence_token = []
    # format [[label], ...]
    sentence_label = []
    
    # format [[sentence1_token], [sentence2_token], ...]
    tokens = []
    # format [[sentence1_label], [sentence2_label], ...]
    labels = []
    
    
    # the offset between training, dev, test
    word_idx_count = offset
    # token2idx dict

    # go throught whole CoNLL file
    for sentence in corpus:
        stacked_embeddings.embed(sentence)
        for token_obj in sentence:
            token = token_obj.text
            label = token_obj.get_tag('ner11').value
            
            # old algorithms
            # char idxs
            char_idxs = []
            for char in token:
                if char in char2idx:
                    char_idxs += [char2idx[char]]  
                else:
                    print("encounter UNK char:", char)

            # word idx
            word_idx = word_idx_count
            word_idx_count += 1   
            
            # label idx
            if label in label2idx:
                label_idx = label2idx[label]
            else:
                print("encounter UNK label:", label)

            # append token inside one sentence
            sentence_token.append((char_idxs, word_idx))
            sentence_label.append(label_idx)
            
            # new part: token_dict, lookup table
            token2idx.append((token, word_idx_count - 1))
            lookup_table[word_idx] = np.asarray(token_obj.embedding)
            
            
            
        # append each sentence
        tokens.append(sentence_token)
        labels.append(sentence_label)
        sentence_label = []
        sentence_token = []
            
            
    
    return tokens, labels,  word_idx_count