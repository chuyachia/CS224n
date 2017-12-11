



import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def get_minibatch_data(data,minibatch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for start in np.arange(0,data_size,minibatch_size):
        minibatch_indices = indices[start:start+minibatch_size]
        yield [minibatch(d,minibatch_indices) for d in data] if list_data else minibatch(data,minibatch_indices)
        
def minibatch(data,indices):
    return data[indices] if type(data) is np.ndarray else [data[i] for i in indices]
    
    

class Match_LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,state_size,question_outputs):
        self._state_size = state_size
        self.question_outputs= question_outputs
    @property
    def state_size(self):		
        return tf.nn.rnn_cell.LSTMStateTuple(self._state_size, self._state_size) 
    @property
    def output_size(self):
        return self._state_size
		
    def __call__(self, inputs, state, scope=None): #inputs = context output at time t
        scope = scope or type(self).__name__ # scope can be passed here through calling tf.dynamic_rnn
        with tf.variable_scope(scope):
            cell, hidden_state = state
            # alignment
            _,max_time,_ = self.question_outputs.get_shape().as_list()
            # reshape question_outputs

            W_q = tf.get_variable(name="W_q", shape = [self._state_size,self._state_size],dtype=tf.float32)
            W_p = tf.get_variable(name="W_p", shape = [self._state_size,self._state_size],dtype=tf.float32)
            W_r = tf.get_variable(name="W_r", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_p = tf.get_variable(name="b_p", shape = [1,self._state_size],initializer=tf.constant_initializer(0.),dtype=tf.float32)
            e_Q =  tf.constant(1.0, shape=[max_time,1])
            question_output= tf.reshape(self.question_outputs,(-1,self._state_size))
            question_rep= tf.matmul(question_output,W_q)
            question_rep = tf.reshape(question_rep,(-1,max_time,self._state_size)) #batch_size* max_len *state
            #print(question_rep)
            context_rep = tf.matmul(inputs,W_p) #b*state
            hidden_rep = tf.matmul(hidden_state,W_r) #b*state
            #print(context_rep)
            #print(hidden_rep)
            context_hidden_rep = tf.reshape(context_rep+hidden_rep+b_p,(-1,1))
            context_hidden_rep = tf.reshape(tf.matmul(context_hidden_rep,tf.transpose(e_Q)),(-1,self._state_size,max_time))
            context_hidden_rep = tf.transpose(context_hidden_rep,perm=[0,2,1])
            #print(context_hidden_rep)
            G_t = tf.tanh(question_rep+context_hidden_rep)
            G_t = tf.reshape(G_t,(-1,self._state_size))
            w = tf.get_variable(name="w", shape = [self._state_size,1],dtype=tf.float32)
            b = tf.get_variable(name="b", shape = [],dtype=tf.float32)
            attention_rep = tf.matmul(G_t,w) #b max_len,1
            #print(attention_rep)
            attention_rep = tf.reshape(attention_rep,(-1,max_time,1))
            alignment_t = tf.nn.softmax(attention_rep+tf.multiply(e_Q,b)) #batch_size*max_len*1
            weighted_question = tf.matmul(tf.transpose(self.question_outputs, perm=[0, 2, 1]),alignment_t) #batch_size*state*1
            matched_input= tf.concat(1,[hidden_state,tf.squeeze(weighted_question)])#batch_size*2input_size
            # input gate
            W_i = tf.get_variable(name="W_i", shape = [self._state_size*2,self._state_size],dtype=tf.float32)
            U_i = tf.get_variable(name="U_i", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_i = tf.get_variable(name="b_i", shape = [1,self._state_size],dtype=tf.float32)
            # forget gate
            W_f = tf.get_variable(name="W_f", shape = [self._state_size*2,self._state_size],dtype=tf.float32)
            U_f = tf.get_variable(name="U_f", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_f = tf.get_variable(name="b_f", shape = [1,self._state_size],dtype=tf.float32)	
            # output gate
            W_o = tf.get_variable(name="W_o", shape = [self._state_size*2,self._state_size],dtype=tf.float32)
            U_o = tf.get_variable(name="U_o", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_o = tf.get_variable(name="b_o", shape = [1,self._state_size],dtype=tf.float32)
            # new memory cell
            W_c = tf.get_variable(name="W_c", shape = [self._state_size*2,self._state_size],dtype=tf.float32)
            U_c = tf.get_variable(name="U_c", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_c = tf.get_variable(name="b_c", shape = [1,self._state_size],dtype=tf.float32)
            i_t =  tf.sigmoid(tf.matmul(matched_input,W_i)+tf.matmul(hidden_state,U_i)+b_i)
            f_t =  tf.sigmoid(tf.matmul(matched_input,W_f)+tf.matmul(hidden_state,U_f)+b_f)
            o_t =  tf.sigmoid(tf.matmul(matched_input,W_o)+tf.matmul(hidden_state,U_o)+b_o)
            c_hat_t = tf.tanh(tf.matmul(matched_input,W_c)+tf.matmul(hidden_state,U_c)+b_c)
            c_t = tf.multiply(f_t,cell)+tf.multiply(i_t,c_hat_t)
            h_t = tf.multiply(o_t,tf.tanh(c_t))
            new_state = tf.nn.rnn_cell.LSTMStateTuple(c_t, h_t)
        return h_t, new_state
class decode_LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,state_size,knowledge_outputs):# batch_size*max_time*2state_size
        self._state_size = state_size
        self.knowledge_outputs= knowledge_outputs
    @property
    def state_size(self):		
        return tf.nn.rnn_cell.LSTMStateTuple(self._state_size, self._state_size) #because state is tuple
    @property
    def output_size(self):
        return self._state_size		
    def __call__(self, state, scope=None): 
        scope = scope or type(self).__name__ # scope can be passed here through calling tf.dynamic_rnn
        with tf.variable_scope(scope):
            cell,hidden_state = state
            # alignment
            _,max_time,_ = self.knowledge_outputs.get_shape().as_list()
            V = tf.get_variable(name="V", shape = [self._state_size*2,self._state_size],dtype=tf.float32)
            W_a = tf.get_variable(name="W_a", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_a = tf.get_variable(name="b_a", shape = [1,self._state_size],dtype=tf.float32)
            e_P =  tf.constant(1.0, shape=[max_time,1])
            knowledge_outputs = tf.reshape(self.knowledge_outputs,(-1,self._state_size*2))
            knowledge_rep= tf.matmul(knowledge_outputs,V)
            knowledge_rep = tf.reshape(knowledge_rep,(-1,max_time,self._state_size))#batch*max_len*state
            hidden_rep = tf.matmul(hidden_state,W_a)+b_a #batch*state
            hidden_rep = tf.reshape(tf.matmul(tf.reshape(hidden_rep,(-1,1)),tf.transpose(e_P)),(-1,self._state_size,max_time))
            hidden_rep = tf.transpose(hidden_rep,perm=[0,2,1])
            F_t = tf.tanh(knowledge_rep+hidden_rep)
            F_t = tf.reshape(F_t,(-1,self._state_size))
            v = tf.get_variable(name="v", shape = [self._state_size,1],dtype=tf.float32)
            c = tf.get_variable(name="c", shape = [],dtype=tf.float32)
            attention_rep = tf.matmul(F_t,v) #b max_len,1
            attention_rep = tf.reshape(attention_rep,(-1,max_time,1))
            prob_t = attention_rep+tf.multiply(e_P,c)
            logit_t = tf.nn.softmax(prob_t)#batch_size*max_len*1
            #knowledge_outpts = 
            weighted_input = tf.squeeze(tf.matmul(tf.transpose(self.knowledge_outputs, perm=[0, 2, 1]),logit_t)) #batch_size*2state*1 ->batch_size*2state
            # input gate
            W_i = tf.get_variable(name="W_i", shape = [self._state_size*2,self._state_size],dtype=tf.float32)
            U_i = tf.get_variable(name="U_i", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_i = tf.get_variable(name="b_i", shape = [1,self._state_size],dtype=tf.float32)
            # forget gate
            W_f = tf.get_variable(name="W_f", shape = [self._state_size*2,self._state_size],dtype=tf.float32)
            U_f = tf.get_variable(name="U_f", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_f = tf.get_variable(name="b_f", shape = [1,self._state_size],dtype=tf.float32)	
            # output gate
            W_o = tf.get_variable(name="W_o", shape = [self._state_size*2,self._state_size],dtype=tf.float32)
            U_o = tf.get_variable(name="U_o", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_o = tf.get_variable(name="b_o", shape = [1,self._state_size],dtype=tf.float32)
            # new memory cell
            W_c = tf.get_variable(name="W_c", shape = [self._state_size*2,self._state_size],dtype=tf.float32)
            U_c = tf.get_variable(name="U_c", shape = [self._state_size,self._state_size],dtype=tf.float32)
            b_c = tf.get_variable(name="b_c", shape = [1,self._state_size],dtype=tf.float32)
            i_t =  tf.sigmoid(tf.matmul(weighted_input,W_i)+tf.matmul(hidden_state,U_i)+b_i)
            f_t =  tf.sigmoid(tf.matmul(weighted_input,W_f)+tf.matmul(hidden_state,U_f)+b_f)
            o_t =  tf.sigmoid(tf.matmul(weighted_input,W_o)+tf.matmul(hidden_state,U_o)+b_o)
            c_hat_t = tf.tanh(tf.matmul(weighted_input,W_c)+tf.matmul(hidden_state,U_c)+b_c)
            c_t = tf.multiply(f_t,cell)+tf.multiply(i_t,c_hat_t)
            h_t = tf.multiply(o_t,tf.tanh(c_t))
            new_state = tf.nn.rnn_cell.LSTMStateTuple(c_t, h_t)
        return h_t, new_state, tf.squeeze(prob_t)		
## here too
class Encoder(object):
    def __init__(self, state_size, vocab_dim):
        self.size = state_size #state_size
        self.vocab_dim = vocab_dim #embedding size
                           
    def encode(self, inputs, masks,scope,match_lstm=False,question_outputs=None):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        if match_lstm:
            assert question_outputs is not None, "Question outputs should be provided"
            cell_fw=Match_LSTMCell(self.size,question_outputs)
            cell_bw=Match_LSTMCell(self.size,question_outputs)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,sequence_length= masks,dtype=tf.float32,scope=scope)            
        else:
            cell=tf.nn.rnn_cell.BasicLSTMCell(self.size, state_is_tuple=True)    
            outputs, state = tf.nn.dynamic_rnn(cell,inputs,sequence_length= masks,dtype=tf.float32,scope=scope)
        return outputs, state

class Decoder(object):
    def __init__(self, state_size):
        self.size = state_size

    def decode(self, knowledge_rep): #batch_size*max_contex_length*2*state_size
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        #Run a final LSTM that does a 2-class classification of these vectors as O or ANSWER
        h_t = tf.zeros(shape=[tf.shape(knowledge_rep)[0],self.size])
        c_t = tf.zeros(shape=[tf.shape(knowledge_rep)[0],self.size])
        state = tf.nn.rnn_cell.LSTMStateTuple(c_t, h_t)
        cell = decode_LSTMCell(self.size,knowledge_rep)
        prob = []
        with tf.variable_scope("decode"):
            for time_step in range(2):
                if time_step >0:
                    tf.get_variable_scope().reuse_variables()
                h_t, state, p_t = cell(state) #p_t = batch_size*max_len
                prob.append(p_t)
        logits = tf.stack((prob[0],prob[1]))#2*batch*max_len(classes)
        return logits

class QASystem(object): ##maybe we don't need to pass question_max_len and context_max_len and can just leave them as none
    def __init__(self, encoder, decoder,question_max_len,context_max_len,embed_path,starter_rate,batch_size,dropout_rate,optimizer_opt,max_grad_norm): #, **kwargs 
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========

        self.global_step = tf.Variable(0,name='global_step',trainable=False)

        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.encoder = encoder
        self.decoder = decoder
        self.question_max_len= question_max_len
        self.context_max_len = context_max_len


        self.q_placeholder = tf.placeholder(tf.int32, shape=(None,self.question_max_len))
        self.c_placeholder = tf.placeholder(tf.int32, shape=(None,self.context_max_len))
        self.span_id_placeholder = tf.placeholder(tf.int32, shape=(None,2))
        self.q_seq_len_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.c_seq_len_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.c_mask_placeholder = tf.placeholder(tf.bool, shape=(None,self.context_max_len))
        self.keep_rate_placeholder = tf.placeholder(tf.float32,())

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):#
            self.embed_matrix = self.setup_embeddings(embed_path)
            self.logits = self.setup_system()
            self.loss = self.setup_loss()
        ### Seems OK up to here
            
        # ==== set up training/updating procedure ====
        learning_rate = tf.train.exponential_decay(starter_rate, self.global_step,100000, 0.96, staircase=True)
        optimize= get_optimizer(optimizer_opt)
        optimizer = optimize(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        [grad, var] = zip(*gradients)
        grad,_ = tf.clip_by_global_norm(grad,max_grad_norm)
        gradients = zip(grad, var)
        self.train_op = optimizer.apply_gradients(gradients,global_step=self.global_step)
        self.saver =  tf.train.Saver() # create saver after the graph is built

        #pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return: 
        """
		
        ## Featurize input_placeholder
        q_featurized = tf.nn.embedding_lookup(self.embed_matrix,self.q_placeholder)
        c_featurized = tf.nn.embedding_lookup(self.embed_matrix,self.c_placeholder)
        ## Encode question
        q_outputs, q_state= self.encoder.encode(q_featurized,self.q_seq_len_placeholder,scope='question')
        # Add dropout to output
        q_outputs = tf.nn.dropout(q_outputs,keep_prob=self.keep_rate_placeholder) 
        ## Encode context
        c_outputs, c_state=self.encoder.encode(c_featurized,self.c_seq_len_placeholder,scope='context')
        # Add dropout to output
        c_outputs = tf.nn.dropout(c_outputs,keep_prob=self.keep_rate_placeholder)
        ## Attention over context
        k_outputs, k_state = self.encoder.encode(c_outputs,self.c_seq_len_placeholder,scope='attention',match_lstm=True,question_outputs=q_outputs)
        k_outputs = tf.concat(2,k_outputs) #batch_size*max_contex_length*2*state_size
        ## Decode
        logits = self.decoder.decode(k_outputs)
        return logits
        #raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            loss_temp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.transpose(self.span_id_placeholder))
            loss = tf.reduce_mean(loss_temp)
        return loss

    def setup_embeddings(self,path):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with np.load(path) as data:     
            with vs.variable_scope("embeddings"):
                embed_matrix = tf.Variable(data['glove'],dtype=tf.float32,trainable=False)
        return embed_matrix

    def optimize(self, session, train): #train_x, train_y
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """

        input_feed = {
            self.q_placeholder:train[0],
            self.q_seq_len_placeholder:train[1],
            self.c_placeholder:train[2],
            self.c_seq_len_placeholder:train[3],
            self.span_id_placeholder:train[4],
            self.keep_rate_placeholder:1.0-self.dropout_rate,
        }


        output_feed = [self.train_op,self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session,valid):#valid_x, valid_y # called by self.validate
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {
            self.q_placeholder:valid[0],
            self.q_seq_len_placeholder:valid[1],
            self.c_placeholder:valid[2],
            self.c_seq_len_placeholder:valid[3],
            self.span_id_placeholder:valid[4],
            self.keep_rate_placeholder:1.0
        }



        output_feed = self.loss

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {
            self.q_placeholder:test[0],
            self.q_seq_len_placeholder:test[1],
            self.c_placeholder:test[2],
            self.c_seq_len_placeholder:test[3],
            self.keep_rate_placeholder:1.0
        }

        output_feed = self.logits

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):
        logits = self.decode(session, test_x) #2*batch*max_len(classes)
        print(logits)
        span_id = np.argmax(logits,-1)# 2*batch
        return span_id

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100,use_sample=True, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        if use_sample:
            sample_indices = np.random.choice(dataset[4].shape[0], sample)
            sample_data = [data[sample_indices]  for data in dataset]
        else:
            sample_data = dataset
        s_e_preds = np.transpose(self.answer(session,sample_data))
        print(s_e_preds)
        s_e_labels = sample_data[4]
        print(s_e_labels)
        preds=[]
        labels=[]
        for i in range(sample_data[0].shape[0]):
            preds_temp= [0 if indx <s_e_preds[i][0] or indx > s_e_preds[i][1] else 1 for indx in range(sample_data[3][i])]
            labels_temp= [0 if indx <s_e_labels[i][0] or indx > s_e_labels[i][1] else 1 for indx in range(sample_data[3][i])]
            preds.extend(preds_temp)
            labels.extend(labels_temp)		
        preds= np.array(preds)
        labels = np.array(labels)
        f1 = 0.
        em = 0.
        TP = np.count_nonzero(preds*labels)
        TN = np.count_nonzero((preds - 1)*(labels - 1))
        FP = np.count_nonzero(preds * (labels - 1))
        FN = np.count_nonzero((preds - 1) * labels)
        try:
            precision = TP/(TP + FP)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1= 0
        em = recall
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em


    def train(self, session, dataset, train_dir, small_data_test=False): #
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """
        
        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        if small_data_test:
            for i in range(1000):
                _, loss = self.optimize(session, dataset)
                print("Training cross-entroy loss: {}".format(loss))
                if i>0 and i%10==0:
                    f1, em = self.evaluate_answer(session, dataset,use_sample=False)
                    print("F1 score: {} EM: {}".format(f1,em))
                
        else:
            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum([np.prod(tf.shape(t.value()).eval()) for t in params])
            toc = time.time()
            logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
            mean_loss=[]
            for i, data in enumerate(get_minibatch_data(dataset,self.batch_size)):
                _, loss = self.optimize(session, data)
                mean_loss.append(loss)
                #print("Training cross-entroy loss: {}".format(loss))
                if i%20==0:
                    print("Training cross-entroy loss: {}".format(sum(mean_loss)/len(mean_loss)))
                    mean_loss=[]
                if i>0 and i% 100 ==0:
                    f1, em = self.evaluate_answer(session, dataset)
                    print("F1 score: {} EM: {}".format(f1,em))

                    self.saver.save(session,train_dir+'/qa',global_step=self.global_step)
            
        



if __name__ == "__main__":
    question_max_len = 40
    context_max_len = 600
    state_size=200
    encoder = Encoder(state_size=state_size, vocab_dim=100)
    decoder = Decoder(state_size=750)
    qa = QASystem(encoder, decoder,question_max_len,context_max_len,"C:/Users/Client/Desktop/NLPwithDeepLearning/assignment4/data/squad/glove.trimmed.100.npz",0.01,10,0.15,"adam",10)

