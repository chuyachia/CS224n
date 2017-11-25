



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

def compute_alignment(input_hidden_state, attention_hidden_states):# attention_hidden_states: max_time* batch_size*_state_size, input_hidden_state:batch_size*_state_size
    print(attention_hidden_states.get_shape().as_list())
    _max_time,_,_state_size = attention_hidden_states.get_shape().as_list()
    attention_hidden_states = tf.reshape(attention_hidden_states,[_max_time,-1])
    input_hidden_state= tf.reshape(input_hidden_state,[-1])
    logits = tf.reshape(tf.multiply(input_hidden_state,attention_hidden_states),[_max_time,-1,_state_size])
    logits = tf.reduce_sum(logits,2)
    alignment = tf.nn.softmax(logits) 
    alignment = tf.reshape(alignment,[_max_time,-1,1])
    return alignment

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
    
    

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,input_size,state_size,condition_state=None):# input_size:dimensions the embedding of each input
        self.input_size = input_size
        self._state_size = state_size
        self.condition_state= condition_state
    @property
    def state_size(self):		
        return tf.nn.rnn_cell.LSTMStateTuple(self._state_size, self._state_size) #because state is tuple
    @property
    def output_size(self):
        return self._state_size
		
    def __call__(self, inputs, state, scope=None): #input : batch_size*input_size, state : batch_size*_state_size, condition_state:max_time* batch_size*_state_size -> is the output of question encoding (tf.nn.dynamic_rnn)
        # tf.nn.dynamic_rnn return outputs [ max_time, batch_size, cell.output_size], state[batch_size, cell.state_size]
        scope = scope or type(self).__name__ # scope can be passed here through calling tf.dynamic_rnn
        with tf.variable_scope(scope):
            cell, hidden_state = state
            # input gate
            W_i = tf.get_variable(name="W_i", shape = [self.input_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            U_i = tf.get_variable(name="U_i", shape = [self._state_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            b_i = tf.get_variable(name="b_i", shape = [1,self._state_size],initializer=tf.constant_initializer(0.),dtype=tf.float32)
            # forget gate
            W_f = tf.get_variable(name="W_f", shape = [self.input_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            U_f = tf.get_variable(name="U_f", shape = [self._state_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            b_f = tf.get_variable(name="b_f", shape = [1,self._state_size],initializer=tf.constant_initializer(0.),dtype=tf.float32)	
            # output gate
            W_o = tf.get_variable(name="W_o", shape = [self.input_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            U_o = tf.get_variable(name="U_o", shape = [self._state_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            b_o = tf.get_variable(name="b_o", shape = [1,self._state_size],initializer=tf.constant_initializer(0.),dtype=tf.float32)
            # new memory cell
            W_c = tf.get_variable(name="W_c", shape = [self.input_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            U_c = tf.get_variable(name="U_c", shape = [self._state_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            b_c = tf.get_variable(name="b_c", shape = [1,self._state_size],initializer=tf.constant_initializer(0.),dtype=tf.float32)
            if self.condition_state is not None:
                V_i = tf.get_variable(name="V_i", shape = [self._state_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
                V_f = tf.get_variable(name="V_f", shape = [self._state_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
                V_o = tf.get_variable(name="V_o", shape = [self._state_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
                V_c = tf.get_variable(name="V_c", shape = [self._state_size,self._state_size],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
                print(self.condition_state)
                alignment = compute_alignment(hidden_state,self.condition_state)
                attention = tf.reduce_sum(tf.multiply(alignment,self.condition_state),0) # [batch_size,_state_size]
                i_t =  tf.sigmoid(tf.matmul(inputs,W_i)+tf.matmul(hidden_state,U_i)+tf.matmul(attention,V_i)+b_i)
                f_t =  tf.sigmoid(tf.matmul(inputs,W_f)+tf.matmul(hidden_state,U_f)+tf.matmul(attention,V_f)+b_f)
                o_t =  tf.sigmoid(tf.matmul(inputs,W_o)+tf.matmul(hidden_state,U_o)+tf.matmul(attention,V_o)+b_o)
                c_hat_t = tf.tanh(tf.matmul(inputs,W_c)+tf.matmul(hidden_state,U_c)+tf.matmul(attention,V_c)+b_c)
            else:
                i_t =  tf.sigmoid(tf.matmul(inputs,W_i)+tf.matmul(hidden_state,U_i)+b_i)
                f_t =  tf.sigmoid(tf.matmul(inputs,W_f)+tf.matmul(hidden_state,U_f)+b_f)
                o_t =  tf.sigmoid(tf.matmul(inputs,W_o)+tf.matmul(hidden_state,U_o)+b_o)
                c_hat_t = tf.tanh(tf.matmul(inputs,W_c)+tf.matmul(hidden_state,U_c)+b_c)
            c_t = tf.multiply(f_t,cell)+tf.multiply(i_t,c_hat_t)
            h_t = tf.multiply(o_t,tf.tanh(c_t))
            new_state = tf.nn.rnn_cell.LSTMStateTuple(c_t, h_t)
        return h_t, new_state		

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size #state_size
        self.vocab_dim = vocab_dim #embedding size
                           
    def encode(self, inputs, masks, encoder_state_input=None):
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
        if encoder_state_input is not None:
            cell_fw = LSTMCell(self.vocab_dim, self.size*2,encoder_state_input)
            cell_bw = LSTMCell(self.vocab_dim, self.size*2,encoder_state_input)
            
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,sequence_length= masks,dtype=tf.float32,time_major=True,scope='context')
        else:
            cell_fw = LSTMCell(self.vocab_dim, self.size)
            cell_bw = LSTMCell(self.vocab_dim, self.size)
            
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,sequence_length= masks,dtype=tf.float32,time_major=True,scope='question')
        return outputs, state

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep): #knowledge_rep :context_max_time* batch_size*context_state_size
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
        _max_time,_batch_size,_input_size = knowledge_rep.get_shape().as_list()
        #_max_time,_batch_size,_input_size = [s[i].value for i in range(0, len(s))]
        cell = LSTMCell(_input_size,self.output_size)
        outputs, state = tf.nn.dynamic_rnn(cell,knowledge_rep,dtype=tf.float32,time_major=True,scope='knowledge') #outputs: context_max_time* batch_size*output_size
        W_d = tf.get_variable(name="W_d", shape = [self.output_size,2],initializer= tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        b_d = tf.get_variable(name="b_d", shape = [1,2],initializer=tf.constant_initializer(0.),dtype=tf.float32)
        logits = tf.map_fn(lambda x: tf.matmul(x, W_d) + b_d , outputs) #context_max_time*batch_size*2
        return logits

class QASystem(object): ##maybe we don't need to pass question_max_len and context_max_len and can just leave them as none
    def __init__(self, encoder, decoder,question_max_len,context_max_len,embed_path,learning_rate,batch_size,dropout_rate,optimizer): #, **kwargs 
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


        self.q_placeholder = tf.placeholder(tf.int32, shape=(self.question_max_len,None))#self.question_max_len
        self.c_placeholder = tf.placeholder(tf.int32, shape=(self.context_max_len,None))#self.context_max_len
        self.label_placeholder = tf.placeholder(tf.float32, shape=(self.context_max_len,None,2)) #
        self.q_seq_len_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.c_seq_len_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.c_mask_placeholder = tf.placeholder(tf.bool, shape=(self.context_max_len,None))#self.context_max_len
        self.dropout_placeholder = tf.placeholder(tf.float32,())

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.embed_matrix = self.setup_embeddings(embed_path)
            self.logits = self.setup_system()
            self.loss = self.setup_loss()

        # ==== set up training/updating procedure ====
        optimizer= get_optimizer(optimizer)
        self.train_op = optimizer(learning_rate).minimize(self.loss,global_step=self.global_step)
        self.saver =  tf.train.Saver() # create saver after the graph is built

        #pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return: 
        """
		
        # Featurize input_placeholder
        q_featurized = tf.nn.embedding_lookup(self.embed_matrix,self.q_placeholder)
        c_featurized = tf.nn.embedding_lookup(self.embed_matrix,self.c_placeholder)
        # Encode question
        q_outputs, q_state= self.encoder.encode(q_featurized,self.q_seq_len_placeholder)
        q_outputs = tf.concat(2,q_outputs) # concat first before applying dropout
        ## Add dropout to output
        q_outputs = tf.nn.dropout(q_outputs,keep_prob=self.dropout_placeholder)
        q_rep = tf.concat(1,[q_state[0].h, q_state[1].h])
        # Encode context
        c_outputs, c_state=self.encoder.encode(c_featurized,self.c_seq_len_placeholder,q_outputs)
        c_outputs = tf.concat(2,c_outputs)
        ## Add dropout to output
        c_outputs = tf.nn.dropout(c_outputs,keep_prob=self.dropout_placeholder)
        # Attention over context reppresentation
        context_align = compute_alignment(tf.concat(1,[q_rep,q_rep]), c_outputs) ### problem here [10(batch_size)*400(state_size*2)] [600(context_max_len)*10(batch_size)*800(state_size*2*2)] dimension does not match
        context_attention = tf.reduce_sum(tf.multiply(context_align,c_outputs),0) #[batch_size,state_size*2*2]
        knowledge = tf.multiply(context_attention,c_outputs)
        # Decode context
        logits = self.decoder.decode(knowledge)
        return logits
        #raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            loss_temp = tf.nn.sigmoid_cross_entropy_with_logits(logits= self.logits,targets=self.label_placeholder)
            loss_temp = tf.boolean_mask(loss_temp,self.c_mask_placeholder)
            loss = tf.reduce_mean(loss_temp)
        return loss

    def setup_embeddings(self,path):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with np.load(path) as data:     
            with vs.variable_scope("embeddings"):
                embed_matrix = tf.Variable(data['glove'],dtype=tf.float32)
        return embed_matrix

    def optimize(self, session, train): #train_x, train_y
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {
            self.q_placeholder:np.transpose(train[0]),
            self.q_seq_len_placeholder:train[1],
            self.c_placeholder:np.transpose(train[2]),
            self.c_seq_len_placeholder:train[3],
            self.c_mask_placeholder:np.transpose(train[4]),
            self.label_placeholder:np.transpose(train[5],(2,0,1)),
			self.dropout_placeholder:self.dropout_rate
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
            self.q_placeholder:np.transpose(valid[0]),
            self.q_seq_len_placeholder:valid[1],
            self.c_placeholder:np.transpose(valid[2]),
            self.c_seq_len_placeholder:valid[3],
            self.c_mask_placeholder:np.transpose(valid[4]),
            self.label_placeholder:np.transpose(valid[5],(2,0,1)),
			self.dropout_placeholder:1.0
        }



        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {
            self.q_placeholder:np.transpose(test[0]),
            self.q_seq_len_placeholder:test[1],
            self.c_placeholder:np.transpose(test[2]),
            self.c_seq_len_placeholder:test[3],
            self.c_mask_placeholder:np.transpose(test[4]),
			self.dropout_placeholder:1.0
        }

        output_feed = [self.logits]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):
        logits = self.decode(session, test_x)
        yp, yp2 = np.split(np.array(logits[0]),2,2)
        a_s = np.argmax(yp, axis=0)
        a_e = np.argmax(yp2, axis=0)
        a_s= np.reshape(a_s,(-1))
        a_e= np.reshape(a_e,(-1))
        return (a_s, a_e)

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

    def evaluate_answer(self, session, dataset, sample=100, log=False):
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
        ### NEED to be rewritten since label are now 2 clases
        #print(dataset[6].shape[0])
        #print(sample)
        sample_indices = np.random.choice(dataset[6].shape[0], sample)
        #print(sample_indices)
        sample_data = [data[sample_indices]  for data in dataset]
        a_s, a_e = self.answer(session,sample_data)
        s_e_labels = sample_data[6]
        preds=[]
        labels=[]
        for i in range(sample):
            preds_temp= [0 if indx <a_s[i] or indx > a_e[i] else 1 for indx in range(sample_data[3][i])]
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


    def train(self, session, dataset, train_dir): #
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
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum([np.prod(tf.shape(t.value()).eval()) for t in params])
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        for i, data in enumerate(get_minibatch_data(dataset,self.batch_size)):
            _, loss = self.optimize(session, data)
            print("Training cross-entroy loss: {}".format(loss))
            if i% 10 ==0:
                f1, em = self.evaluate_answer(session, dataset)
                print("F1 score: {} EM: {}".format(f1,em))
                print(train_dir)
                self.saver.save(session,train_dir+'/qa',global_step=i)
            
        



if __name__ == "__main__":
    ## Test
    # Toy data
    batch_size = 2
    question_max_len=5
    context_max_len = 10
    input_size = 3
    state_size = 4
    output_size = 7 #size of state vector in decoder
    question = np.arange(batch_size*question_max_len*input_size,dtype=np.float32)
    question = question.reshape((question_max_len,batch_size,input_size))
    context = np.arange(batch_size*context_max_len*input_size,dtype=np.float32)
    context = context.reshape((context_max_len,batch_size,input_size))  
    question_mask = np.array([question_max_len]*batch_size,dtype=np.int32)
    context_mask = np.array([context_max_len]*batch_size,dtype=np.int32)
    # Placeholder
    q_placeholder = tf.placeholder(tf.float32, shape=(question_max_len,batch_size,input_size))
    c_placeholder = tf.placeholder(tf.float32, shape=(context_max_len,batch_size,input_size))
    q_seq_len_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    c_seq_len_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    # Initialize encoder
    encoder = Encoder(state_size,input_size)
    # Encode question
    q_outputs, q_state= encoder.encode(q_placeholder,q_mask_placeholder)
    q_outputs = tf.concat(2,q_outputs)
    q_rep = tf.concat(1,[q_state[0].h, q_state[1].h])
    # Encode context
    c_outputs, c_state=encoder.encode(c_placeholder,c_seq_len_placeholder,q_outputs)
    c_rep = tf.concat(2,c_outputs)
    # Attention over knowledge
    print(q_rep)
    print(c_rep)
    context_align = compute_alignment(tf.concat(0,[q_rep,q_rep]), c_rep)
    context_attention = tf.reduce_sum(tf.multiply(context_align,c_rep),0) #[batch_size,state_size*2*2]###problem
    knowledge = tf.multiply(context_attention,c_rep)
    # Decode context
    decoder = Decoder(output_size)
    preds, logits = decoder.decode(knowledge)


