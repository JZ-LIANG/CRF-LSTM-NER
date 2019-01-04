import logging
import os

class Config():
    """
    Initialize hyperparameters

    modify the parameters in following:

    """
    # file path
    # pretrained embedding 
    # dataset
    # vocab 

    config_NO ='Fasttext_wiki, dropout_rate = (0.5,0.5) main_lstm = 500'
    # embeddings_size
    dim_word = 300
    dim_char = 50

    # model hyperparameters
    hidden_size_char = 150 # lstm on chars
    hidden_size_lstm = 500 # lstm on word embeddings

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
    dropout_fc            = 0.5
    batch_size       = 30
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.85
    clip             = -1 # if negative, no clipping


    # dataset files
    path_train = 'data/CoNLL2003/eng.train'
    path_eval = 'data/CoNLL2003/eng.testa'
    path_test = 'data/CoNLL2003/eng.testb'

    # fasttext 
    command = '../../word_embedding/fastText/fasttext'
    option = 'print-word-vectors'
    bin_file = '../../word_embedding/fastText/data/wiki.simple.bin'
    fasttext_embedding_file = '../../word_embedding/fastText/data/fasttext_CoNLL_lookup_table.txt'
    vocab_file = '../../word_embedding/fastText/data/fasttext_CoNLL_vocabs.txt'


    # save model and output
    if_save_model = True

    path_root = 'output/result_fastext2/'
    path_output_train = 'train.txt'
    path_output_test = path_root + 'test.txt'
    path_output_eval = path_root + 'eval.txt'
    path_result_train = path_root + 'train_result.txt'
    path_result_eval = path_root + 'eval_result.txt'
    path_result_test = path_root + 'test_result.txt'
    path_model =  path_root + 'model/'
    path_log = path_root + 'log.log'
    log_name = 'fasttext1'


    # file format
    separator = ' '
    lowercase = False

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

        logging.basicConfig(filename= self.path_log, filemode='w+', level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
        self.logger = logging.getLogger(self.log_name)
        self.logger.info('the config of this run are :' + self.config_NO )



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
