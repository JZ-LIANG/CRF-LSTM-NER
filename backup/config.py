
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
    n_epochs          = 5
    nepoch_no_imprv = 3
    # if apply regularization of W and b in loss function
    loss_regularization = False
    # regularization rate
    l2_reg       = 0


    # The probability that each element is kept.
    dropout          = 0.9
    batch_size       = 50
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping


    # dataset files
    path_train = 'data/CoNLL2003/eng.train'
    path_eval = 'data/CoNLL2003/eng.testa'
    path_test = 'data/CoNLL2003/eng.testb'
    filename_glove = 'data/glove/glove.6B.100d.txt'


    # save model and output
    if_save_model = True

    path_output_train = 'output/result/train.txt'
    path_output_test = 'output/result/test.txt'
    path_output_eval = 'output/result/eval.txt'
    path_result_train = 'output/result/train_result.txt'
    path_result_eval = 'output/result/eval_result.txt'
    path_result_test = 'output/result/test_result.txt'
    path_model =  'output/result/model/'


    # file format
    separator = ' '
    lowercase = True

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



