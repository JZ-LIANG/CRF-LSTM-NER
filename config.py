
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
    dim_word = 300
    dim_char = 100

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # training
    train_embeddings = False
    loss_regularization = False
    l2_reg       = 0
    n_epochs          = 15


    # The probability that each element is kept.
    dropout          = 0.9
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    n_epoch_no_imprv  = 3






    def __init__(self):

        self.load()


    
    def load(self):
        
        # load pre-trained embeddings
        self.pretrained_embeddings = get_trimmed_glove_vectors(self.filename_trimmed)


        # compute from corpus 
        self.nwords = 10000000    
        self.nchars = 48     
        self.ntags  = 5     


    