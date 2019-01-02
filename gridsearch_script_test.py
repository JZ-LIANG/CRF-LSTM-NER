from config_fasttext1 import Config as Config1
from config_fasttext2 import Config as Config2

from model import Model
import tensorflow as tf
import numpy as np
from pre_processing import initial_2idxs_fasttext, get_inputs
import time
import warnings
warnings.filterwarnings('ignore')


def main():
	config = Config1()
	token2idx, char2idx, label2idx, lookup_table = initial_2idxs_fasttext(config)
	train_x, train_y = get_inputs('train', token2idx, char2idx, label2idx, config)
	eval_x, eval_y = get_inputs('eval', token2idx, char2idx, label2idx, config)
	test_x, test_y = get_inputs('test', token2idx, char2idx, label2idx, config)
	ner_model = Model(config)
	ner_model.build_graph()
	ner_model.initialize_session()
	print(ner_model.config.config_NO)

	config = Config2()
	token2idx, char2idx, label2idx, lookup_table = initial_2idxs_fasttext(config)
	train_x, train_y = get_inputs('train', token2idx, char2idx, label2idx, config)
	eval_x, eval_y = get_inputs('eval', token2idx, char2idx, label2idx, config)
	test_x, test_y = get_inputs('test', token2idx, char2idx, label2idx, config)
	ner_model = Model(config)
	ner_model.build_graph()
	ner_model.initialize_session()
	print(ner_model.config.config_NO)


if __name__ == "__main__":
	main()
