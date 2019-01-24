import os
import tensorflow as tf
from pre_processing import next_batch, pad_sentence, pad_word, get_chunks
import codecs
import numpy as np
import time

class Model(object):
    """CRF-BiLSTM NER model"""

    def __init__(self, config):
        """
        Load the hyperparams in config

        """
        self.config = config
        self.sess = None



    def Bi_LSTM_Layer (self, hidden_state_size, inputs, sequence_length):
        with tf.variable_scope('forward'): 
            fw = tf.contrib.rnn.LSTMCell(hidden_state_size, state_is_tuple=True)

        with tf.variable_scope('backward'): 
            bw = tf.contrib.rnn.LSTMCell(hidden_state_size, state_is_tuple=True)

        output = tf.nn.bidirectional_dynamic_rnn(fw, bw, inputs, sequence_length=sequence_length, dtype=tf.float32)

        return output


    def FCNN_layer(self, output):
        W = tf.get_variable("W", dtype = tf.float32,
               shape = [2*self.config.hidden_size_lstm, self.config.n_label])

        b = tf.get_variable("b", shape = [self.config.n_label], 
                            dtype = tf.float32, initializer = tf.zeros_initializer())

        shape = tf.shape(output)
        output = tf.reshape(output, shape = [-1, 2*self.config.hidden_size_lstm])
        scores = tf.matmul(output, W) +b

        return tf.reshape(scores, shape = [-1, shape[1], self.config.n_label]), W, b


    def CRF_LOSS_layer(self, scores, labels, sequence_lengths, W, b):
        log_likelihood, trans_matrix = tf.contrib.crf.crf_log_likelihood(scores, 
        labels, sequence_lengths)
        loss = tf.reduce_mean(-log_likelihood)
        # regularization the W, b
        if self.config.loss_regularization :
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

            # learning rate & dropout_rate
            self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
            self.dropout_embed = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_embed")
            self.dropout_fc = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_fc")

            # embedding lookup table
            self._word_embeddings_lookup_table = tf.placeholder(dtype=tf.float32, shape=[], name="_word_embeddings_lookup_table")




        # with tf.variable_scope("pretrained_embedding"): 
        #     _word_embeddings_lookup_table = tf.Variable(self.config.lookup_table, 
        #         name = "_word_embeddings_lookup_table", dtype = tf.float32, trainable = False)

            word_embeddings = tf.nn.embedding_lookup(self._word_embeddings_lookup_table, self.word_ids, name="word_embeddings")


        with tf.variable_scope("character_embedding"):
            # random initiate character embeddings table
            _char_embedding_table = tf.get_variable(name="_char_embedding_table", dtype=tf.float32,
                shape=[self.config.n_char, self.config.dim_char])

            # shape =[batch_size, max_length of sentence, max_length of word, char_embedd_size]
            char_embeddings = tf.nn.embedding_lookup(_char_embedding_table,
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
                # # for sub LSTM we extract only final hidden state
                _, ((_, state_fw), (_, state_bw)) = self.Bi_LSTM_Layer(self.config.hidden_size_char, 
                    char_embeddings, word_lengths)

                # shape = [batch_size * max_length of sentence, 2*char hidden size]
                trained_embeddings = tf.concat([state_fw, state_bw], axis=-1)

                # shape = (batch size, max_length of sentence, 2*char hidden size)
                trained_embeddings = tf.reshape(trained_embeddings,shape=[shape[0], shape[1], 2*self.config.hidden_size_char])


        # concat to get the final embedding
        word_embeddings = tf.concat([word_embeddings, trained_embeddings], axis = -1)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_embed)


        # main Bi-LSTM layer
        with tf.variable_scope("Bi-LSTM"):
            # for main LSTM we extract outpur on each time steps
            (output_fw, output_bw), _ = self.Bi_LSTM_Layer(self.config.hidden_size_lstm, 
                self.word_embeddings, self.sentence_lengths)

            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout_fc)


        # FC layer to project word+context representation to a score vector
        with tf.variable_scope("FCNN"):
            self.scores, W, b = self.FCNN_layer(output)

        # CRF layer + loss
        with tf.variable_scope("CRF_LOSS"):
            self.trans_params, self.loss = self.CRF_LOSS_layer(self.scores, self.labels, self.sentence_lengths, W, b)


        with tf.variable_scope("train"):

            optimizer = tf.train.AdamOptimizer(self.lr)
            # gradient clipping if clip is positive
            if self.config.clip > 0: 
                grads, vs     = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)


    def initialize_session(self):
        """Defines self.sess and initialize the variables"""

        self.config.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.saver  = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())



    def train(self, train_x, train_y, dev_x, dev_y, path_model = None):
        """
        train model

        """
        # re-load the model
        if path_model != None:
            self.restore_session(path_model)
            self.config.logger.info("Model restored.")
        elif self.sess == None and path_model == None:
            self.config.logger.info('can not find model, exit')
            exit()


        best_F1 = 0
        # early stopping metric
        nepoch_no_imprv = 0 
        lr = self.config.lr

        for epoch in range(self.config.n_epochs):
            self.config.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            epoch_start_time = time.time()

            # self.config.batch_size
            for i, (x_batch, y_batch) in enumerate(next_batch(train_x, train_y, self.config.batch_size, shuffle = True)):
                # print("batch:{}".format(i))
                fd, sentence_lengths,_,_ = self.get_fd(x_batch, y_batch, lr, dropout_embed = self.config.dropout_embed, dropout_fc = self.config.dropout_fc)
                
                _, train_loss  = self.sess.run([self.train_op, self.loss], feed_dict=fd)

            metrics = self.evaluate(dev_x, dev_y)
            self.config.logger.info("Epoch {:} 's F1 ={:}, epoch_runing_time ={:} .".format(epoch + 1, metrics["f1"], (time.time() - epoch_start_time)))
            # if there is more then 1 epoch without improvement, try a small lr
            if (self.config.lr_decay < 1) and (nepoch_no_imprv > 1):
                lr *= self.config.lr_decay



            if metrics["f1"] >= best_F1:
                nepoch_no_imprv = 0
                best_F1 = metrics["f1"]
                self.config.logger.info("- new best F1, save new model.")
                if self.config.if_save_model:
                    self.save_session()

            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.config.logger.info("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    break





    def evaluate(self, dev_x, dev_y):
        """
        Evaluates performance on dev set

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for i, (x_batch, y_batch) in enumerate(next_batch(dev_x, dev_y, self.config.batch_size, shuffle = True)):
            
            fd, sentence_lengths,label_padded,_ = self.get_fd(x_batch, y_batch)
            
            scores, trans_params = self.sess.run(
                [self.scores, self.trans_params], feed_dict=fd)

            viterbi_sequences = self.viterbi_decode(scores, sentence_lengths, trans_params)

            for lab, lab_pred, length in zip(label_padded, viterbi_sequences,
                                             sentence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.idx2label))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.idx2label))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}




    def viterbi_decode(self, scores, sequence_lengths, trans_params):
        viterbi_sequences = []
        # iterate over the sentences because no batching in vitervi_decode
        for score, sequence_length in zip(scores, sequence_lengths):
            logit = score[:sequence_length] # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences
    
    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.path_model):
            os.makedirs(self.config.path_model)
        model = self.config.path_model + 'model.ckpt'
        self.saver.save(self.sess, model)
        
    def restore_session(self, path_model):
        model = path_model + 'model.ckpt'
        self.saver.restore(self.sess, model)
        
    def close(self):
        self.sess.close()
        
    def get_fd(self, x_batch, y_batch, lr = None, dropout_embed = 1, dropout_fc = 1):
        sentences = [list(zip(*x))[1] for x in x_batch]
        char_sentences = [list(zip(*x))[0] for x in x_batch]

        sentences_padded, sentence_lengths = pad_sentence(sentences)
        label_padded, _ = pad_sentence(y_batch)
        chars_padded, chars_lengths = pad_word(char_sentences)

        fd = {
            self.word_ids: sentences_padded,
            self.sentence_lengths: sentence_lengths,
            self.char_ids: chars_padded,
            self.word_lengths: chars_lengths,
            self.labels: label_padded,
            self.lr: lr,
            self.dropout_embed: dropout_embed,
            self.dropout_fc: dropout_fc,
            self._word_embeddings_lookup_table: self.config.lookup_table
        }
        return fd,sentence_lengths,label_padded,sentences


    def test(self, test_x, test_y, path_output_file, path_result, path_model = None):
        # check the sess exist
        if path_model != None:
            self.restore_session(path_model)
            self.config.logger.info("Model restored.")
        elif self.sess == None and path_model == None:
            self.config.logger.info('can not find model, exit')
            exit()
            
        row = ''
        output_file = codecs.open(path_output_file, 'w', 'UTF-8')
        first_row = 'token' + '\t' + 'true' + '\t' + 'prediction'+ '\n'
        output_file.write(first_row)
        for i, (x_batch, y_batch) in enumerate(next_batch(test_x, test_y, self.config.batch_size, shuffle = False)):
        # predict  
            fd, sentence_lengths, _, sentences = self.get_fd(x_batch, y_batch)

            scores, trans_params = self.sess.run(
                [self.scores, self.trans_params], feed_dict=fd)
            viterbi_sequences = self.viterbi_decode(scores, sentence_lengths, trans_params)

            # write to output file

            for sentence, golden_label, prediction in zip(sentences,y_batch, viterbi_sequences):
                for i in range(len(golden_label)):
                    row = self.config.idx2token[sentence[i]] + '\t' + self.config.idx2label[golden_label[i]] + '\t' + self.config.idx2label[prediction[i]] 
                    output_file.write(row + '\n')
                output_file.write('\n')
        output_file.close()

        # write F1 result
        script = 'conlleval'
        shell_command = 'perl {0} < {1} > {2}'.format(script, path_output_file, path_result)
        os.system(shell_command)
        with open(path_result, 'r') as f:
            classification_report = f.read()
            self.config.logger.info(classification_report) 
            

