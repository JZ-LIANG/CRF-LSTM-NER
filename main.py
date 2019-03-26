from pre_processing import initial_2idxs, get_inputs
from config import Config
from model import Model
import tensorflow as tf
import numpy as np


def main():
	"""
	a test main for illustration.
	"""
	# prepare input data
	config = Config()
	token2idx, char2idx, label2idx, lookup_table = initial_2idxs(config)
	train_x, train_y = get_inputs('train', token2idx, char2idx, label2idx, config)
	eval_x, eval_y = get_inputs('eval', token2idx, char2idx, label2idx, config)
	test_x, test_y = get_inputs('test', token2idx, char2idx, label2idx, config)

	# build and train
	ner_model = Model(config)
	tf.reset_default_graph()
	ner_model.build_graph()
	ner_model.initialize_session()
	ner_model.train(train_x,train_y,eval_x,eval_y)

	# test
	ner_model.test(train_x,train_y,config.path_output_train, config.path_result_train)
	ner_model.test(eval_x,eval_y,config.path_output_eval, config.path_result_eval)
	ner_model.test(test_x,test_y,config.path_output_test, config.path_result_test)

	# close
	ner_model.close()


if __name__ == "__main__":
    main()
