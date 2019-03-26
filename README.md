# CRF-LSTM-NER
A CRF-BiLSTM model to benchmark the performances of different word embeddings on your own corpus. <br>

the objectives of this model are:
*  Build a CRF-BiLSTM  Network in Tensorflow with methods for easisly switching among different word embeddings (word2vec, glove, fasttext, elmo, flair and any combinations of them) while keep the same CRF-LSTM Network  unchanged.
* Methods for easily gridsearch on the suitable parameters.



## Requirements

Python 3, TensorFlow 1.0+, gensim, and Flair(optinal):

- if need to use "Contextual" embedding, Flair library(https://github.com/zalandoresearch/flair) is required. 


## How To Use
1. Modify the data directory and Configure the **Hyper-parameter** accordingly in **config.py**

```
    # embeddings_size
    dim_word = 300
    dim_char = 50
    #
    hidden_size_char = 64 # lstm on chars
    hidden_size_lstm = 128 # lstm on word embeddings

    # dataset
    path_data_root = 'data/CoNLL2003/'
    path_train = path_data_root +'eng.testa'
    path_eval = path_data_root +'eng.testa'
    path_test = path_data_root +'eng.testb'

```

2. Designate the embedding you want. Since different embeddings come with different file formats, this part maybe vary slightly accordding to the embedding you choose. there is a example for them in "[How To Use.ipynb](https://github.com/JZ-LIANG/CRF-LSTM-NER/blob/master/Example.ipynb)"

```
    # glove
	config = Config('glove')
	glove_file_path = 'data/glove/glove.6B.100d.txt'
	config.init_glove(glove_file_path)

    # fasttext
	config = Config('fasttext')
	command ='../fastText/fasttext'
	bin_file ='../fastText/data/cc.en.300.bin'
	config.init_fasttext(command, bin_file)

```

3. Parse the corpus and generate the "index" and "input". the following code will base on the vocabularies of embedding and corpus to generate the index for token/character/label and the mapping the each sentence into a sequence of index. this part also handle the sepcific configuration of model base on the corpus, like the number of kind of label, the number of unique character in corpus.

```
# parse the corpus and generate the input data
token2idx, char2idx, label2idx, lookup_table = get_idx(config)
train_x, train_y = get_inputs('train', token2idx, char2idx, label2idx, config)
eval_x, eval_y = get_inputs('eval', token2idx, char2idx, label2idx, config)
test_x, test_y = get_inputs('test', token2idx, char2idx, label2idx, config)

```

4. initial the model's graph and train/eval/test.
```
# initial the same NER model 
ner_model = Model(config)
ner_model.build_graph()
ner_model.initialize_session()

```

5. resutl: the F1 score based on the label will be print and detail of training processing can be find in "./output/log.log". 

you could find more detail in "[How To Use.ipynb](https://github.com/JZ-LIANG/CRF-LSTM-NER/blob/master/Example.ipynb)"




## Reference
This model is based on the following papers:
* Lample, Guillaume, et al. "[Neural architectures for named entity recognition.](https://arxiv.org/abs/1603.01360)" arXiv preprint arXiv:1603.01360 (2016).
* Zhiheng Huang, et al. "[Bidirectional LSTM-CRF Models for Sequence Tagging.](https://arxiv.org/abs/1508.01991)" arXiv preprint arXiv:1508.01991 (2015).
	
