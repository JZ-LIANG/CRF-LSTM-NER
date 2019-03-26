import logging
import os
import numpy as np
import datetime
import sys

class Config():
    """
    Initialize hyperparameters
    modify the parameters in following:
    """

    # types: glove/fasttext/w2v/contextual
    # contextual means: flair/elmo or combinations of them
    embedding_type = 'glove'
    


    ############################################
    # setting hyperparamters here              #
    #                                          #
    ############################################
    # embeddings_size
    dim_word = 100
    dim_char = 50

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 128 # lstm on word embeddings

    # training
    train_embeddings = False
    loss_regularization = False
    n_epochs          = 5
    nepoch_no_imprv = 3
    # if apply regularization of W and b in loss function
    loss_regularization = False
    # regularization rate
    l2_reg       = 0


    # The probability that each element is kept.
    dropout_embed          = 0.5
    dropout_fc            = 0.8
    batch_size       = 50
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping


    # contextual embedding lookup table size
    # these parameters only need to be set when use contextual 
    # (lookup table is too big to be fit into one single tf.variable)
    # e.g. : total tokens in Conll2003: 301418, len(flair + w2v): 4396
    lookup_table_d0 = 301418 
    lookup_table_d1 = 4396


    ############################################
    # configurate the load/save dir here       #
    #                                          #
    ############################################
    # load dataset files
    path_data_root = 'data/CoNLL2003/'
    path_train = path_data_root +'eng.testa'
    path_eval = path_data_root +'eng.testa'
    path_test = path_data_root +'eng.testb'


    save_path = 'output/result_' + embedding_type + '/'
    # save model and output
    if_save_model = True
    path_output_train = 'train.txt'
    path_output_test = save_path + 'test.txt'
    path_output_eval = save_path + 'eval.txt'
    path_result_train = save_path + 'train_result.txt'
    path_result_eval = save_path + 'eval_result.txt'
    path_result_test = save_path + 'test_result.txt'
    path_model =  save_path + 'model/'
    path_log = save_path + 'log.log'


    # file format
    separator = ' '
    lowercase = True

    # index files
    save_idx = True
    save_table = True
    lookup_table_file_path = save_path + 'lookup_table.npz'
    path_idx = save_path + 'idx/'
    file_token_idx = path_idx + 'token2idx.json'
    file_char_idx = path_idx + 'char2idx.json'
    file_label_idx = path_idx + 'tag2idx.json'
    # file to save the indx_config
    indx_config = path_idx + 'indx_config.pkl'


    ##########################################################
    # modify following  when use glove embedding             #
    # or set it use config.init_glove()                      #
    ##########################################################
    glove_file_path = 'data/glove/glove.6B.100d.txt'


    ##########################################################
    # modify following when use fasttext embedding           #
    # or set it use config.init_fasttext()                   #                                                 
    ##########################################################
    command = '../../../word_embedding/fastText/fasttext'
    option = 'print-word-vectors'
    bin_file = '../../../word_embedding/fastText/data/cc.en.300.bin'
    fasttext_embedding_file = path_idx + 'fasttext_CoNLL_lookup_table.txt'
    vocab_file = path_idx + 'fasttext_CoNLL_vocabs.txt'



    def __init__(self, type = None):

        if type != None:
            self.embedding_type = type
            self.log_name = 'log_' + self.embedding_type
            save_path = 'output/result_' + self.embedding_type + '/'
            self._set_save_path(save_path)
        else:
            print('specific the embedding')
            
        # to be update
        self.n_label = -1
        self.n_char = -1
        self.n_word = -1
        self.lookup_table = None


    def init_glove(self, glove_file_path, save_path = None):
        if glove_file_path != None:
            self.glove_file_path = glove_file_path
        if save_path != None:
            self._set_save_path(save_path)
        self.logger = self._myLogger(self.log_name, self.path_log)
        self.logger.info('config object Initialized')
        

    def init_fasttext(self, command, bin_file, save_path = None, fasttext_embedding_file = None, vocab_file = None, option = 'print-word-vectors'):
        self.command = command
        self. bin_file = bin_file

        if fasttext_embedding_file == None :
            self.fasttext_embedding_file = self.path_idx + 'fasttext_CoNLL_lookup_table.txt'
        else: 
            self.fasttext_embedding_file = fasttext_embedding_file

        if vocab_file == None :
            self.vocab_file = self.path_idx + 'fasttext_CoNLL_vocabs.txt'
        else: 
            self.vocab_file = vocab_file

        self.option = option
        
        if save_path != None:
            self._set_save_path(save_path)
        self.logger = self._myLogger(self.log_name, self.path_log)
        self.logger.info('config object Initialized')

    def init_w2v(self, w2v, save_path = None):
        self.w2v = w2v
        if save_path != None:
            self._set_save_path(save_path)
        self.logger = self._myLogger(self.log_name, self.path_log)
        self.logger.info('config object Initialized')  
        
    def init_contextual(self,lookup_table, token2idx, save_path = None):
        self.lookup_table = lookup_table
        self.idx2token = {tuple1[1]: tuple1[0] for tuple1 in token2idx}
        if save_path != None:
            self._set_save_path(save_path)
        self.logger = self._myLogger(self.log_name, self.path_log)
        self.logger.info('config object Initialized')

    def set_n_label(self, n_):
        self.n_label = n_
    def set_n_char(self, n_):
        self.n_char = n_
    def set_n_word(self, n_):
        self.n_word = n_
    def set_lookup_table(self, lookup_table):
        self.lookup_table = lookup_table
    # reverse index for label
    def set_idx2label(self, label2idx):
        self.idx2label = {v: k for k, v in label2idx.items()}
    def set_idx2token(self, token2idx):
        self.idx2token = {v: k for k, v in token2idx.items()}
        

    def load_lookup_table(self, file =None):
        if file == None:
            file = self.lookup_table_file_path

        try:
            with np.load(file) as data:
                self.lookup_table = data["lookup_table"]

        except IOError:
            raise MyIOError(file)

    def load_indx(self, file =None):
        if file == None:
            file = self.indx_config

        try:
            with open(file, 'rb') as f:
                self.n_label = pickle.load(f)
                self.n_char = pickle.load(f)
                self.n_word = pickle.load(f)
                self.token2idx = pickle.load(f)
                self.char2idx = pickle.load(f)
                self.label2idx = pickle.load(f)
                self.idx2label = pickle.load(f)
                self.idx2token = pickle.load(f)

        except IOError:
            raise MyIOError(file)
  
    def _set_save_path(self, save_path):
        self.save_path = save_path
        self.path_output_train = 'train.txt'
        self.path_output_test = save_path + 'test.txt'
        self.path_output_eval = save_path + 'eval.txt'
        self.path_result_train = save_path + 'train_result.txt'
        self.path_result_eval = save_path + 'eval_result.txt'
        self.path_result_test = save_path + 'test_result.txt'
        self.path_model =  save_path + 'model/'
        self.path_log = save_path + 'log.log'
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)
        if not os.path.exists(self.path_idx):
            os.makedirs(self.path_idx )
            
 
    def _myLogger(self, name, path):
        name = name + 'test'
        logger = logging.getLogger(name)
        if len(logger.handlers):
            return logger
        else:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            now = datetime.datetime.now()
            handler = logging.FileHandler(path + now.strftime("%Y-%m-%d") +'.log')
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            # console
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            return logger