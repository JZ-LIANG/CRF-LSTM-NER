import logging

class Config():
    """
    Initialize hyperparameters

    modify the parameters in following:

    """
    # file path
    # pretrained embedding 
    # dataset
    # vocab 

    config_NO ='Fasttext_wiki, dropout_rate = 0.5, main_lstm = 300'
    # embeddings_size
    dim_word = 300
    dim_char = 50

    # model hyperparameters
    hidden_size_char = 150 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

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
    dropout          = 0.5
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

    path_output_train = 'output/result_fastext/train.txt'
    path_output_test = 'output/result_fastext/test.txt'
    path_output_eval = 'output/result_fastext/eval.txt'
    path_result_train = 'output/result_fastext/train_result.txt'
    path_result_eval = 'output/result_fastext/eval_result.txt'
    path_result_test = 'output/result_fastext/test_result.txt'
    path_model =  'output/result_fastext/model/'
    path_log = 'output/result_fastext/model/log.log'
    log_name = 'fasttext1'


    # file format
    separator = ' '
    lowercase = False

    # index files
    save_idx = True
    save_table = False
    lookup_table_file_path = None
    file_token_idx = 'output/idx/token2idx.json'
    file_char_idx = 'output/idx/char2idx.json'
    file_label_idx = 'output/idx/tag2idx.json'








    def __init__(self):
            # to be update
        self.n_label = -1
        self.n_char = -1
        self.n_word = -1
        self.lookup_table = None

        logging.basicConfig(filename= path_log,level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
        self.logger = logging.getLogger(log_name)
        self.logger.info('the config of this run are :' + config_NO )



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



