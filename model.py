import os
import tensorflow as tf

class Model(object):
    """CRF-BiLSTM NER model"""

    def __init__(self, config):
        """
		Load the hyperparams in config

        """
        self.config = config



    def Bi_LSTM_Layer (self, hidden_state_size, inputs, sequence_length):
        with tf.variable_scope('forward'): 
            fw = tf.contrib.rnn.LSTMCell(hidden_state_size, state_is_tuple=True)

        with tf.variable_scope('backward'): 
            bw = tf.contrib.rnn.LSTMCell(hidden_state_size, state_is_tuple=True)

        output = tf.nn.bidirectional_dynamic_rnn(fw, bw, inputs, sequence_length=word_lengths, dtype=tf.float32)

        return output


    def FCNN_layer(self, output):
        W = tf.get_variable("W", dtype = tf.float32,
               shape = [2*self.config.hidden_size_lstm, self.config.n_label])

        b = tf.get_variable("b", shape = [self.config.n_label], 
                            dtype = tf.float32, initializer = tf.zeros_initializer())

        shape = tp.shape(output)
        output = tf.reshape(output, shape = [-1, 2*self.config.hidden_size_lstm])
        scores = tf.matmul(output, W) +b

        return tf.reshape(scores, shape = [-1, shape[1], self.config.n_label]), W, b


    def CRF_LOSS_layer(self, scores, labels, sequence_lengths, W, b):
        log_likelihood, trans_matrix = tf.contrib.crf.crf_log_likelihood(scores, 
        labels, sequence_lengths)
        loss = tf.reduce_mean(-log_likelihood)
        # regularization the W, b
        if loss_regularization :
            reg = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            cost += reg * self.config.l2_reg
        return trans_matrix, loss



    def build_graph(self):

        # inputs
        with tf.name_scope('inputs'):
            # shape = [batch_size, max_length of sentence in this batch]
            self.word_ids = tf.placeholder(tf.int32, shape = [None, None], name = "word_ids")

            # shape = [batch_size]
            self.sentence_lengths = tf.placeholder(tf.int32, shape=[None], name = "sentence_lengths")

            # shape =[batch_size, max_length of sentence, max_length of word]
            self.char_ids = tf.placeholder(tf.int32, shape = [None, None, None], name = "char_ids")

            # shape = [batch_size, max_length of sentence]
            self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name = "word_lengths")

            # shape = [batch_size, max_length of sentence]
            self.labels = tf.placeholder(tf.int32, shape = [None, None], name = "labels")



        with tf.variable_scope("pretrained_embedding"): 
            _word_embeddings_lookup_table = tf.Variable(self.config.lookup_table, 
                name = "_word_embeddings_lookup_table", dtype = tf.float32, trainable = False)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings_lookup_table, self.word_ids, name="word_embeddings")


        with tf.variable_scope("character_embedding"):
            # random initiate character embeddings table
            _char_embedding_table = tf.get_variable(name="_char_embedding_table", dtype=tf.float32,
                shape=[self.config.n_char, self.config.dim_char])

            # shape =[batch_size, max_length of sentence, max_length of word, char_embedd_size]
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                    self.char_ids, name="char_embeddings")

            # reshape sequence to the requirement of rnn, put the time dimension on axis=1
            shape = tf.shape(char_embeddings)

            # shape = [batch_size * max_length of sentence, max_length of sentence, char_embedd_size]
            char_embeddings = tf.reshape(char_embeddings,
                    shape=[-1, shape[-2], self.config.dim_char])

            # shape = [batch_size * max_length of sentence]
            word_lengths = tf.reshape(self.word_lengths, shape=[-1])

            # sub-BiLSTM to generate character embeddings
            with tf.name_scope("sub-BiLSTM"):
                # # for main LSTM we extract only final hidden state
                _, ((_, state_fw), (_, state_bw)) = Bi_LSTM_Layer(self.config.hidden_size_char, 
                    char_embeddings, word_lengths)

                # shape = [batch_size * max_length of sentence, 2*char hidden size]
                trained_embeddings = tf.concat([state_fw, state_bw], axis=-1)

                # shape = (batch size, max_length of sentence, 2*char hidden size)
                trained_embeddings = tf.reshape(output,shape=[shape[0], shape[1], 2*self.config.hidden_size_char])


        # concat to get thefinal embedding
        word_embeddings = tf.concat([word_embeddings, trained_embeddings], axis = -1)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.config.dropout)

        # main Bi-LSTM layer
        with tf.variable_scope("Bi-LSTM"):
            # for main LSTM we extract outpur on each time steps
            (output_fw, output_bw), _ = Bi_LSTM_Layer(self.config.hidden_size_lstm, 
                self.word_embeddings, self.sentence_lengths)

            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.config.dropout)


        # FC layer to project word+context representation to a score vector
        with tf.variable_scope("FCNN"):
            self.scores, W, b = FCNN_layer(output)

        # CRF layer + loss
        with tf.variable_scope("CRF_LOSS"):
            self.trans_params, self.loss =CRF_LOSS_layer(self.scores, self.labels, self.sentence_lengths, W, b)


        with tf.variable_scope("train"):

            optimizer = tf.train.AdamOptimizer(self.config.lr)
            # gradient clipping if clip is positive
            if self.config.clip > 0: 
                grads, vs     = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)


    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        print("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def train(self, train_x, train_y, dev_x, dev_y):
        """
        train model

        """
        best_F1 = 0
        # early stopping metric
        nepoch_no_imprv = 0 

        for epoch in range(self.config.nepochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

            for i, (words, labels) in enumerate(next_batch(train_x, train_y, self.config.batch_size, shuffle = True)):
                fd, _ = self.padding (words, labels)
                _, train_loss  = self.sess.run([self.train_op, self.loss], feed_dict=fd)

            metrics = self.evaluate(dev)
            print("Epoch {:} 's F1 ={:}".format(epoch + 1, metrics[F1]))

            if metrics[F1] >= best_F1:
                nepoch_no_imprv = 0
                best_F1 = metrics[F1]
                print("- new best score!")

            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    print("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    break





    def evaluate(self, dev_x, dev_y):
        """
        Evaluates performance on dev set

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for i, (words, labels) in enumerate(next_batch(dev_x, dev_y, self.config.batch_size, shuffle = True)):
            fd, sentence_lengths = self.padding(words, None)

            scores, trans_params = self.sess.run(
                [self.scores, self.trans_params], feed_dict=fd)

            viterbi_sequences = self.viterbi_decode(scores, sentence_lengths)

            for lab, lab_pred, length in zip(labels, viterbi_sequences,
                                             sentence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}




    def viterbi_decode(self, scores, sequence_lengths):
        # iterate over the sentences because no batching in vitervi_decode
        for score, sequence_length in zip(scores, sequence_lengths):
            logit = logit[:sequence_length] # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences



    # def padding (self, words, labels):


    #     return  feed, sentence_lengths





       
     