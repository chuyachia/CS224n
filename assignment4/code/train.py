



import os
import json

import tensorflow as tf
import numpy as np

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/Users/Client/AppData/Local/Temp/cs224n-squad-train' #/tmp/cs224n-squad-train
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def pad_sentences(data_path,max_len):
    if tf.gfile.Exists(data_path):
        seq_len = []
        data = []
        mask = []
        with tf.gfile.GFile(data_path,mode="rb") as f:
            for line in f:
                mask_line = [True]*max_len
                sent = line.strip().split(' ')
                sent_len = len(sent)
                seq_len.append(sent_len)                
                if sent_len < max_len:
                    mask_line[sent_len:] = [False]*(max_len-sent_len)
                    sent.extend([0]*(max_len-sent_len))
                elif sent_len > max_len:
                    sent = sent[:max_len]
                data.append(sent)
                mask.append(mask_line)
        return  np.array(data),  np.array(seq_len),  np.array(mask)
    else:
        raise ValueError("Data file %s not found.", data_path)

def one_hot_encode_answer_span(data_path,context_max_len):
    if tf.gfile.Exists(data_path):
        data_one_hot= []
        data_id=[]
        with tf.gfile.GFile(data_path,mode="rb") as f:
            for line in f:
                start,end = [int(s) for s in line.strip().split(' ')]
                data_id.append([start,end])
                start_one_hat = [0.0 if i!= start else 1.0 for i in range(context_max_len)]
                end_one_hat = [0.0 if i!= end else 1.0 for i in range(context_max_len)]
                data_one_hot.append([start_one_hat,end_one_hat])
        return np.array(data_one_hot), np.array(data_id)                
    else:
        raise ValueError("Data file %s not found.", data_path)
    


def main(_):
    # Do what you need to load datasets from FLAGS.data_dir
    dataset = {}
    question_max_len = 40
    context_max_len = 600
    # Preprocess and collect train data
    train_q_path = pjoin(FLAGS.data_dir, "train.ids.question")
    train_q_data, train_q_seq_len,_ = pad_sentences(train_q_path,question_max_len)
    train_c_path = pjoin(FLAGS.data_dir, "train.ids.context")
    train_c_data, train_c_seq_len,train_c_mask = pad_sentences(train_c_path,context_max_len)
    train_s_path = pjoin(FLAGS.data_dir, "train.span")
    train_label,train_s_e_id = one_hot_encode_answer_span(train_s_path,context_max_len)
    dataset['train'] =[train_q_data,train_q_seq_len,train_c_data,train_c_seq_len,train_c_mask,train_label,train_s_e_id]
    # Preprocess and collect validation data
    val_q_path = pjoin(FLAGS.data_dir, "val.ids.question")
    val_q_data, val_q_seq_len,_ = pad_sentences(val_q_path,question_max_len)
    val_c_path = pjoin(FLAGS.data_dir, "val.ids.context")
    val_c_data, val_c_seq_len,val_c_mask = pad_sentences(val_c_path,context_max_len)
    val_s_path = pjoin(FLAGS.data_dir, "val.span")
    val_label,val_s_e_id = one_hot_encode_answer_span(val_s_path,context_max_len)
    dataset['val'] = [val_q_data,val_q_seq_len,val_c_data,val_c_seq_len,val_c_mask,val_label,val_s_e_id]
    
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder,question_max_len,context_max_len, embed_path,FLAGS.learning_rate,FLAGS.batch_size,FLAGS.dropout,FLAGS.optimizer)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        print(save_train_dir)
        for i in range(FLAGS.epochs):
            qa.train(sess, dataset['train'], save_train_dir)#
            print('Finish training epoch {}'.format(i))
            qa.evaluate_answer(sess, dataset['val'])# vocab, FLAGS.evaluate

if __name__ == "__main__":
    tf.app.run()
