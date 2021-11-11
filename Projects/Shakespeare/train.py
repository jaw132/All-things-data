'''
Train a model to produce novel shakespearean poetry using Tensorflow on
the data extracted in get_sonnets.py
'''

#TODO remove hard coded values

# import libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np
import matplotlib.pyplot as plt


def process_data(filename):
    '''

    :param filename: path of file that contains sonnets

    Tokenizes the sonnets, creates n_grams and then the training data, that
    being a n_gram followed by the next word in the sentence, as that's what
    we are trying to predict
    '''
    data = open(filename).read()

    corpus = data.lower().split("\n")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)


    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, total_words, tokenizer, max_sequence_len

def build_model(predictors, label, total_words, max_len):
    '''

    :param predictors: n_grams set up in process_data
    :param label: labels set up in process_data
    :param total_words: total number of words in corpus needed for embedding
    :param max_len: the length of predictors

    Using the Keras API define the architecture for a neural netwrok to predict
    the next word in the sequence
    '''
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_len-1))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(128, activation="relu", activity_regularizer=regularizers.L2(0.01)))
    model.add(Dense(total_words, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['acc'])
    print(model.summary())

    model_history = model.fit(predictors, label, epochs=100, verbose=1)
    model.save("shakespeare_model.h5")
    return model_history


def plot_model(model_history):
    '''

    :param model_history: the training information from building the model

    simple plots to see how accuracy and loss changed as the model changed
    '''
    acc = model_history.history['acc']
    loss = model_history.history['loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()

    plt.show()



def generate_text(prompt, num_of_words, tokenizer, model, max_len):

    '''
    Takes a prompt and generates novel text (hopefully shakespearean)
    '''

    for _ in range(num_of_words):
        token_list = tokenizer.texts_to_sequences([prompt])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list))
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        prompt += " " + output_word
    print(prompt)

def main():
    predictors, labels, total_words, tokenizer, max_len = process_data('processed_sonnets.txt')
    model, model_history = build_model(predictors, labels, total_words, max_len)
    plot_model(model_history)
    prompt="Shall I compare thee to a summer's day"
    num_of_words=100
    generate_text(prompt, num_of_words, tokenizer, model, max_len)


if __name__ == '__main__':
    main()


