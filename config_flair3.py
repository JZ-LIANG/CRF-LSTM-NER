import logging
import os
import numpy as np
import sys

class Config():
    """
    Initialize hyperparameters

    modify the parameters in following:

    """
    # file path
    # pretrained embedding 
    # dataset
    # vocab 
    config_NO ='flair, dropout_rate = (0.5,0.8), main_lstm = 100'

    # embeddings_size
    dim_word = 100
    dim_char = 100

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 100 # lstm on word embeddings

    # training
    train_embeddings = False
    loss_regularization = False
    n_epochs          = 50
    nepoch_no_imprv = 6
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


    # dataset files
    dataset_root = '/home/semantic/Liang_NER/data/corpus/conll_03/'
    path_train = dataset_root + 'eng.train'
    path_eval = dataset_root + 'eng.testa'
    path_test = dataset_root + 'eng.testb'



    # save model and output
    if_save_model = True

    path_root = '/home/semantic/Liang_NER/output/result_flair3/'
    path_output_train = 'train.txt'
    path_output_test = path_root + 'test.txt'
    path_output_eval = path_root + 'eval.txt'
    path_result_train = path_root + 'train_result.txt'
    path_result_eval = path_root + 'eval_result.txt'
    path_result_test = path_root + 'test_result.txt'
    path_model =  path_root + 'model/'
    path_log = path_root + 'log.log'
    log_name = 'flair'


    # file format
    separator = ' '
    lowercase = True

    # index files
    save_idx = True
    save_table = True
    lookup_table_file_path = path_root + 'lookup_table.npz'
    path_idx = path_root + 'idx/'
    file_token_idx = path_idx + 'token2idx.json'
    file_char_idx = path_idx + 'char2idx.json'
    file_label_idx = path_idx + 'tag2idx.json'
    # file to save the indx_config
    indx_config = path_idx + 'indx_config.pkl'








    def __init__(self):
            # to be update
        self.n_label = -1
        self.n_char = -1
        self.n_word = -1
        self.lookup_table = None

        if not os.path.exists(self.path_root):
            os.makedirs(self.path_root)

        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)
        if not os.path.exists(self.path_idx):
            os.makedirs(self.path_idx )


        # loggging
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(self.path_log)
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)
        # self.logger.addHandler(ch)

        self.logger.info('the config of this run are :' + self.config_NO )


        # logger.info('the config of this run are :' + self.config_NO)

        # handlers = [logging.FileHandler(filename = self.path_log, mode='w+'), logging.StreamHandler()]
        # logging.basicConfig(handlers = handlers, level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
        # self.logger = logging.getLogger(self.log_name)
        # self.logger.info('the config of this run are :' + self.config_NO )




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
    def set_idx2token_flair(self, token2idx):
        self.idx2token = {tuple1[1]: tuple1[0] for tuple1 in token2idx}



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
