import numpy as np
import data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data import get_train_sentences
from data import get_dev_sentences
from data import get_test_sentences
from data import get_vocabulary
from data import max_len_sentences
from data import avg_len_sentences
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Layer, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt


def encode_pad(sentences):
    train_sentences = get_train_sentences()
    vocab_size, vocab = get_vocabulary(train_sentences)

    joined = []
    for s in sentences:
        listToStr = ' '.join([str(elem) for elem in s])
        joined.append(listToStr)

    encoded_sentences = [one_hot(d, vocab_size) for d in joined]

    #avg = round(avg_len_sentences(sentences))
    avg = round(max_len_sentences(train_sentences))

    # Pad after each sentence
    padded_sentences = pad_sequences(encoded_sentences, maxlen=avg, padding='post')

    return padded_sentences


def tokenizer_encode_pad(sentences):
    train_sentences = get_train_sentences()
    vocab_size, vocab = get_vocabulary(train_sentences)

    joined = []
    for s in sentences:
        listToStr = ' '.join([str(elem) for elem in s])
        joined.append(listToStr)

    max = round(max_len_sentences(train_sentences))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(joined)

    # 3981 distinct words
    vocab_size = len(tokenizer.word_index) + 1

    # Embed each sentence
    encoded_sentences = tokenizer.texts_to_sequences(joined)

    max = round(max_len_sentences(train_sentences))

    # Pad after each sentence
    padded_sentences = pad_sequences(encoded_sentences, maxlen=max, padding='post')

    return tokenizer, vocab_size, padded_sentences


def create_embedding_matrix(posts, tokenizer):
    total, vocabulary = data.get_vocabulary(posts)
    word_ids, posts, bios, freqs, e_freqs, pos = data.read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/train.txt')

    word_pos_glove_dict, posts_features = data.extract_features(word_ids, posts, bios, freqs, e_freqs, pos)
    train_sentences = get_train_sentences()
    tokenizer, vocab_size, padded_train_sentences = tokenizer_encode_pad(train_sentences)
    embedding_matrix = np.zeros((vocab_size, 101))
    for w, i in tokenizer.word_index.items():
        vector = word_pos_glove_dict.get(w)
        if vector is not None:
            embedding_matrix[i] = vector
        else:
            embedding_matrix[i] = np.zeros(101)

    return np.array(embedding_matrix)


def padded_prob():
    word_ids, posts, bios, freqs, e_freqs, pos = data.read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/train.txt')

    train_probabilities = e_freqs

    max = 38
    prob = []
    for sentence in train_probabilities:
        new_sentence = []
        if len(sentence) != max:
            diff = max - len(sentence)
            for vector in sentence:
                new_sentence.append(float(vector))
            for i in range(diff):
                new_sentence.append(0)

            prob.append(np.array(new_sentence))
        else:
            for vector in sentence:
                new_sentence.append(float(vector))
            prob.append(np.array(new_sentence))
    return np.array(prob)


class ChangedMSE(keras.losses.Loss):
    def __init__(self, regularization_factor = 0.1, name="mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.square(y_true - y_pred)) + \
               tf.math.reduce_mean(tf.square(0.5 - y_pred))\
               + self.regularization_factor


def padded_prob_dev():
    word_ids, posts, bios, freqs, e_freqs, pos = data.read_data(
        r'SemEval2020_Task10_Emphasis_Selection-master/train_dev_data/dev.txt')

    train_probabilities = e_freqs

    max = 38
    prob = []
    for sentence in train_probabilities:
        new_sentence = []
        if len(sentence) != max:
            diff = max - len(sentence)
            for vector in sentence:
                new_sentence.append(float(vector))
            for i in range(diff):
                new_sentence.append(0)

            prob.append(np.array(new_sentence))
        else:
            for vector in sentence:
                new_sentence.append(float(vector))
            prob.append(np.array(new_sentence))
    return np.array(prob)


def create_lstm():

    train_sentences = get_train_sentences()
    probabilities = padded_prob()
    print(padded_prob())
    tokenizer, vocab_size, padded_train_sentences = tokenizer_encode_pad(train_sentences)
    # (2742, 38), which means we have 2742 sentences and each has 38 padded words
    print(padded_train_sentences.shape)
    embedding_matrix = create_embedding_matrix(train_sentences, tokenizer)
    print(embedding_matrix.shape)
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 101, weights=[embedding_matrix], trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(38))

    model.add(Dense(76))
    model.add(Dense(76))
    model.add(Dense(76))
    #model.add(Dense(68, activation='sigmoid'))
    # # model.add(Dense(34, kernel_initializer='normal', activation='relu'))
    # # model.add(Dense(38, kernel_initializer='normal', activation='relu'))

    #model.add(Dense(1, kernel_initializer='normal', activation='softmax')) so ova 90
    #model.add(Dense(34, kernel_initializer='normal', activation='sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(38, activation='sigmoid'))

    model.summary()

    dev_sentences = get_dev_sentences()
    dev_probabilities = padded_prob_dev()
    tokenizer, vocab_size, padded_dev_sentences = tokenizer_encode_pad(dev_sentences)
    print(dev_probabilities.shape)
    print(padded_dev_sentences.shape)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')
    history = model.fit(padded_train_sentences, probabilities, epochs=1000, verbose=1,
                        validation_data=(padded_dev_sentences, dev_probabilities))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    loss, accuracy = model.evaluate(padded_train_sentences, probabilities, verbose=1)
    metrics = []


    metrics.append(model.evaluate(padded_train_sentences, probabilities, verbose=1))
    print("Loss is " + str(metrics[0][0]) + ", accuracy is " + str(metrics[0][1] * 100 ))

    test_sentences = get_test_sentences()
    tokenizer, vocab_size, padded_test_sentences = tokenizer_encode_pad(test_sentences)
    print(padded_test_sentences.shape)

    predictions = model.predict(padded_test_sentences[:3])
    print(predictions.shape)
    print(predictions)


def create_bilstm():

    train_sentences = get_train_sentences()
    probabilities = padded_prob()
    print(padded_prob())
    tokenizer, vocab_size, padded_train_sentences = tokenizer_encode_pad(train_sentences)
    # (2742, 38), which means we have 2742 sentences and each has 34 padded words
    print(padded_train_sentences.shape)
    embedding_matrix = create_embedding_matrix(train_sentences, tokenizer)
    print(embedding_matrix.shape)
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 101, weights=[embedding_matrix], trainable=False)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(38)))

    model.add(Dense(76))
    model.add(Dense(76))
    model.add(Dense(76))

    model.add(Dense(38, activation='sigmoid'))

    model.summary()

    dev_sentences = get_dev_sentences()
    dev_probabilities = padded_prob_dev()
    tokenizer, vocab_size, padded_dev_sentences = tokenizer_encode_pad(dev_sentences)
    print(dev_probabilities.shape)
    print(padded_dev_sentences.shape)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')
    history = model.fit(padded_train_sentences, probabilities, epochs=1000, verbose=1,
                        validation_data=(padded_dev_sentences, dev_probabilities))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    loss, accuracy = model.evaluate(padded_train_sentences, probabilities, verbose=1)
    metrics = []
    metrics.append(model.evaluate(padded_train_sentences, probabilities, verbose=1))
    print("Loss is " + str(metrics[0][0]) + ", accuracy is " + str(metrics[0][1] * 100 ))

    test_sentences = get_test_sentences()
    tokenizer, vocab_size, padded_test_sentences = tokenizer_encode_pad(test_sentences)
    print(padded_test_sentences.shape)

    predictions = model.predict(padded_test_sentences[:3])
    print(predictions.shape)
    print(predictions)


if __name__ == '__main__':
    create_lstm()
    create_bilstm()