
class Config():
    """
    Initialize hyperparameters

    modify the parameters in following:

    """
    # file path
    # pretrained embedding 
    # dataset
    # vocab 


    # embeddings_size
    dim_word = 100
    dim_char = 100

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # training
    train_embeddings = False
    loss_regularization = False
    l2_reg       = 0
    n_epochs          = 15
    nepoch_no_imprv = 3


    # The probability that each element is kept.
    dropout          = 0.9
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping


    # dataset files
    path_train = 'data/CoNLL2003/eng.train'
    path_eval = 'data/CoNLL2003/eng.testa'
    path_test = 'data/CoNLL2003/eng.testb'
    filename_glove = 'data/glove/glove.6B.100d.txt'

    # file format
    separator = ' '
    lowercase = True

    # index files
    save_idx = True
    save_table = False
    lookup_table_file_path = None
    file_token_idx = 'data/idx/token2idx.json'
    file_char_idx = 'data/idx/char2idx.json'
    file_label_idx = 'data/idx/tag2idx.json'


    # to be update
    n_label = -1
    n_char = -1
    n_word = -1
    lookup_table = None
    def set_n_label(self, n_):
        n_label = n_
    def set_n_char(self, n_):
        n_char = n_
    def set_n_word(self, n_):
        n_word = n_
    def set_lookup_table(self, lookup_table):
        lookup_table = lookup_table





    def __init__(self):
        pass

