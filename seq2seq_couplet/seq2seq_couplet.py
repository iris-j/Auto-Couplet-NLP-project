import tensorflow as tf
from tensorflow.python.layers.core import Dense
import pickle
import numpy as np
import os

# 超参数
# Number of Epochs
epochs = 500
# Batch Size
batch_size = 200
# num_units in LSTM
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.01
# 每50步显示一次loss
display_step = 50

"""
source = open("test/in_clear.txt", 'w', encoding='utf-8')
target = open("test/out_clear.txt", 'w', encoding='utf-8')

with open('test/in.txt', 'r', encoding='utf-8') as f:
    source_data = f.readlines()
    for i in range(len(source_data)):
        source_data[i] = source_data[i].rstrip('\n')
        source_data[i] = source_data[i].replace(' ', '')
        source.write(source_data[i] + '\n')
with open('test/out.txt', 'r', encoding='utf-8') as f:
    target_data = f.readlines()
    for i in range(len(target_data)):
        target_data[i] = target_data[i].rstrip('\n')
        target_data[i] = target_data[i].replace(' ', '')
        target.write(target_data[i] + '\n')
source.close()
target.close()
"""

with open('data/in_clear.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()

with open('data/out_clear.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()

with open('test/in_clear.txt', 'r', encoding='utf-8') as f:
    source_val_data = f.read()

with open('test/out_clear.txt', 'r', encoding='utf-8') as f:
    target_val_data = f.read()

print(source_data.split('\n')[:10])
print(target_data.split('\n')[:10])
print(source_val_data.split('\n')[:10])
print(target_val_data.split('\n')[:10])


def extract_character_vocab(data):  # 构造汉字词典
    """
    data是原始语料
    返回字符映射表
    """
    if not os.path.exists('int_to_vocab.pkl'):  # 存储词汇表
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        set_words = list(set([character for line in data.split('\n') for character in line]))
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
        with open('int_to_vocab.pkl', 'wb') as of1:
            pickle.dump(int_to_vocab, of1)
        with open('vocab_to_int.pkl', 'wb') as of2:
            pickle.dump(vocab_to_int, of2)
    else:
        with open('int_to_vocab.pkl', 'rb') as if1:
            int_to_vocab = pickle.load(if1)
        with open('vocab_to_int.pkl', 'rb') as if2:
            vocab_to_int = pickle.load(if2)

    return int_to_vocab, vocab_to_int


# 得到输入和输出的字符映射表(同一张字符映射表)
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data + target_data + source_val_data + target_val_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(source_data + target_data + source_val_data + target_val_data)

# 将每一行转换成字符id的list
source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data.split('\n')]
source_int = source_int[:100000]
target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]
target_int = target_int[:100000]
source_val_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_val_data.split('\n')]
target_val_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] + [target_letter_to_int['<EOS>']] for line in target_val_data.split('\n')]


# 输入层
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


# Encoder
def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    """
    构造Encoder层
    - input_data: 输入数据
    - rnn_size: lstm的num_units
    - num_layers: lstm层数
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    """
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, sequence_length=source_sequence_length,
                                                      dtype=tf.float32)

    return encoder_output, encoder_state


def process_decoder_input(data, vocab_to_int, batch_size):  # 处理decoder的输入数据，删除EOS标记，增加GO标记
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return decoder_input


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
    """"
    构造Decoder层
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: lstm层数
    - rnn_size: lstm的num_units
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的Context vector
    - decoder_input: decoder端输入
    """

    # Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    # Output全连接层
    # target_vocab_size定义了输出层的大小
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1))

    # Training decoder
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length)

    # Predicting decoder
    # 与training共享参数

    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size],
                               name='start_token')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens,
                                                                     target_letter_to_int['<EOS>'])

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
                                                                            maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


# 构建seq2seq模型
def seq2seq_model(input_data, targets, lr, target_sequence_length, max_target_sequence_length,
                  source_sequence_length, source_vocab_size, target_vocab_size, encoder_embedding_size,
                  decoder_embedding_size, rnn_size, num_layers):
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoding_embedding_size)

    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)

    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input)

    return training_decoder_output, predicting_decoder_output


# 构造graph
train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       lr,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(source_letter_to_int),
                                                                       len(target_letter_to_int),
                                                                       encoding_embedding_size,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers)

    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name="masks")

    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks
        )

        optimizer = tf.train.AdamOptimizer(lr)

        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


def pad_sentence_batch(sentence_batch, pad_int):
    """
    对batch中的序列进行padding，补全到最大句子长度，保证batch中的每行都有相同的sequence_length
    - sentence batch
    - pad_int: <PAD>的index
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i: start_i + batch_size]
        targets_batch = targets[start_i: start_i + batch_size]

        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

"""
# Train
train_source = source_int[:]
train_target = target_int[:]

# 利用另一个test集的前200句做验证
valid_source = source_val_int[:batch_size]
valid_target = target_val_int[:batch_size]

(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
    get_batches(valid_target, valid_source, batch_size,
                source_letter_to_int['<PAD>'],
                target_letter_to_int['<PAD>']))


checkpoint = "data/trained_model.ckpt"

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    print()
    ckpt = tf.train.get_checkpoint_state('data/')
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
    for epoch_i in range(1, epochs + 1):
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(get_batches(
                train_target, train_source, batch_size, source_letter_to_int['<PAD>'],
                target_letter_to_int['<PAD>']
        )):
            _, loss = sess.run([train_op, cost], feed_dict={
                input_data: sources_batch,
                targets: targets_batch,
                lr: learning_rate,
                target_sequence_length: targets_lengths,
                source_sequence_length: sources_lengths
            })

            if batch_i % display_step == 0:
                # 计算validation loss
                validation_loss = sess.run(
                    [cost],
                    {input_data: valid_sources_batch,
                     targets: valid_targets_batch,
                     lr: learning_rate,
                     target_sequence_length: valid_targets_lengths,
                     source_sequence_length: valid_sources_lengths})

                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              loss,
                              validation_loss[0]))
        if epoch_i % 10 == 0:
            print('After %d epochs, save model.' % epoch_i)
            saver = tf.train.Saver()
            saver.save(sess, checkpoint, global_step=30) # 每10步保存一次最新模型


    # saver.save(sess, checkpoint)
    print('Model Trained and Saved')

"""

# 预测
def source_to_seq(text):
    sequence_length = 10
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [
        source_letter_to_int['<PAD>']] * (sequence_length - len(text))


shanglian = '红梅开岁首'
text = source_to_seq(shanglian)

checkpoint = "data/trained_model.ckpt-30"
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, tf.train.latest_checkpoint('data/'))

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(shanglian)] * batch_size,
                                      source_sequence_length: [len(shanglian)] * batch_size})[0]

    pad = source_letter_to_int["<PAD>"]

    print('原始输入:', shanglian)

    print('\n上联')
    print('  词典Index:    {}'.format([i for i in text if i != pad]))
    print('  上联: {}'.format(" ".join([source_int_to_letter[i] for i in text if i != pad])))

    print('\n下联')
    print('  词典Index:       {}'.format([i for i in answer_logits if i != pad]))
    print('  预测下联: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))