import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.84:
            print("\nReached 0.84% accuracy so cancelling training!")
            self.model.stop_training = True


def imbd_binary():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

    train_data, test_data = imdb['train'], imdb['test']

    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []


    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)


    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    oov_tok = "<OOV>"
    
    tokenizer =  Tokenizer(num_words = vocab_size, oov_token=oov_tok)# YOUR CODE HERE
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

    test_sequences = tokenizer.texts_to_sequences(testing_sentences)
    test_padded = pad_sequences(test_sequences, maxlen=max_length)
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])
        
    callbacks = myCallback()
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(padded, training_labels_final, epochs=10, validation_data=(test_padded, testing_labels_final), callbacks=[callbacks])

    return model



if __name__ == '__main__':
    model = imbd_binary()
    model.save("model-IMDB-binary.h5")

