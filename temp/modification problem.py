modification problem

1. config
	not load and processing word
	no n_char, n_word parameter

	delete
		use_crf
		use_chars

	define self.config.nchars,
	minibatches
	pad_sequences
	get_chunks


2.delete
	all the logger.info
	summary
	self.config.lr *= self.config.lr_decay # decay learning rate

3. graph
	self.dropout = self.config.dropout
	self.lr = self.config.lr


4. 分开提供：
	char_ids, word_ids = zip(*words)









################################################
learn

easy way to have a config file and use config


#####################################################
modified
#####################################################