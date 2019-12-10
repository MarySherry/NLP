import gensim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import re
import string
from keras.models import Model, model_from_yaml
from keras.layers import Dense, LSTM, Input, Embedding
from keras.layers import Activation, dot, concatenate
from keras.layers import TimeDistributed
import nltk
from nltk.translate.bleu_score import corpus_bleu

def cleaning(text):
    ''' Performs cleaning of text of punctuation, digits,
        excessive spaces and transfers to lower-case
    '''
    exclude = set(string.punctuation + string.digits + '«»…‘’―–—‽')
    text = text.lower().strip()
    text = re.sub(r'\s+', " ", text)
    text = ''.join(character for character in text if character not in exclude)

    return text

def process_data(data, max_sentence_length, SOS, EOS):
    ''' Performs data cleaning, filtering of maximal allowed sentence length, appending of Start-of-String
        and End-of-String characters
    '''
    processed_data = data.copy()
    cleaner = lambda x: cleaning(x)
    processed_data.en = processed_data.en.apply(cleaner)
    processed_data.ru = processed_data.ru.apply(cleaner)

    target_seq_lens = np.array([len(sentence) for sentence in processed_data.ru])
    target_idx = np.where(target_seq_lens <= max_sentence_length)[0]
    keep_idx = target_idx
    print("{} input sentence pairs with {} or fewer words in target language".format(len(keep_idx), max_sentence_length))

    # Append Start-Of-Sequence and End-Of-Sequence tags to target sentences:
    processed_data.ru = processed_data .ru.apply(lambda x: SOS + ' ' + x + ' ' + EOS)

    # Subset initial data
    return processed_data.iloc[keep_idx]

def vocabulary_info(data):
    '''Creates a text vocabularies,
    calculates the total number of observed words 
    and the maximal sentence word length'''
    vocabulary=set()
    length_list=[]
    for line in data:
        length_list.append(len(nltk.word_tokenize(line)))
        for word in nltk.word_tokenize(line):
            if word not in vocabulary:
                vocabulary.add(word)
    
    words=set()
    for line in data:
        for word in line.split():
            if word not in words:
                words.add(word)
    # get lengths and sizes
    num_tokens = len(vocabulary)
    max_length = np.max(length_list)
    
    return num_tokens, max_length, vocabulary

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = decoder_token_index[SOS]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == EOS or
           len(decoded_sentence) > decoder_max_length):
            decoded_sentence = SOS + decoded_sentence + EOS
            stop_condition = True
        else:
            decoded_sentence += ' '+sampled_char  

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]


    return decoded_sentence

if __name__ == "__main__":
    
    EOS = '_EOS'
    SOS = 'SOS_'

    if (len(sys.argv) < 2) or (len(sys.argv) > 3):
        print("Usage:")
        print("\tnmt.py data_path [pretrained_model_path]")
        sys.exit()

    DATA_PATH = sys.argv[1]
    MODEL_PATH = None
    if len(sys.argv) == 3:
        MODEL_PATH = sys.argv[2]
    

    data = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['en', 'ru'])

    # Data preprocessing
    # Choose the sentences of word-length less than 14
    max_sentence_length = 14
    data = process_data(data, max_sentence_length, SOS, EOS)
    
    encoder_num_tokens, encoder_max_length, encoder_vocabulary = vocabulary_info(data.en)
    decoder_num_tokens, decoder_max_length, decoder_vocabulary = vocabulary_info(data.ru)
    
    X_train, X_test, y_train, y_test = train_test_split(np.array(data.en), np.array(data.ru), test_size=0.33, random_state=42)
      
    # Assign unique token to words in the vocabulary
    #your code goes here
    encoder_token_index = dict([(word,i) for i, word in enumerate(encoder_vocabulary)])
    decoder_token_index= dict([(word,i) for i, word in enumerate(decoder_vocabulary)])
    
    # loaded from https://code.google.com/archive/p/word2vec/
    news_w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # create a weight matrix for words in training sets
    embedding_matrix = np.zeros((encoder_num_tokens, news_w2v.vector_size))
    for word, i in encoder_token_index.items():
        try:
            embedding_vector = news_w2v.get_vector(word)
        except:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    # Model definition
    emb_size = 300
    lstm_hidden_size = 300
    dropout_rate = 0.2

    batch_size = 128
    epochs = 30
            
    # Create matrix with zeros for source and target data
    encoder_input_data_train = np.zeros(
        (len(X_train), encoder_max_length),
        dtype='float32')
    decoder_input_data_train = np.zeros(
        (len(y_train), decoder_max_length),
        dtype='float32')

    # Create matrix with zeros for target label
    decoder_target_data_train = np.zeros(
        (len(y_train), decoder_max_length, decoder_num_tokens),
        dtype='float32')
    
    for i, (input_text, target_text) in enumerate(zip(X_train, y_train)):
        for t, word in enumerate(nltk.word_tokenize(input_text)):
            encoder_input_data_train[i, t] = encoder_token_index[word]
        for t, word in enumerate(nltk.word_tokenize(target_text)):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data_train[i, t] = decoder_token_index[word]
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                # will update as one hot vector at the labels 
                decoder_target_data_train[i, t - 1, decoder_token_index[word]] = 1.
    
    # Define Encoder Input
    encoder_inputs = Input(shape=(None,))
    
    # Embedding (non-trainable)
    encoder = Embedding(encoder_num_tokens, emb_size , weights=[embedding_matrix],
                    input_shape=(encoder_max_length,), trainable=False)(encoder_inputs)
    # Encoder LSTM
    encoder_outputs, state_h, state_c = LSTM(lstm_hidden_size, return_state=True)(encoder)
    encoder_states = [state_h, state_c] # Encoder States
    
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_inputs = Input(shape=(None,))
    # Decoder Embedding with Normal Keras Embedding (will be trained)
    decoder = Embedding(decoder_num_tokens, emb_size)(decoder_inputs) 
    # Decoder LSTM
    decoder_outputs, _, _ = LSTM(lstm_hidden_size, return_sequences=True, return_state=True)(decoder, initial_state=encoder_states)

    # Softmax & Dense Layer
    decoder_dense = Dense(decoder_num_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Final Model
    model= Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['acc'])
    
    if MODEL_PATH:
        model.load_weights(MODEL_PATH)
        print("Loaded model from disk")
    else:
        model.fit([encoder_input_data_train, decoder_input_data_train], decoder_target_data_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.05)
        model.save('model.h5')
    
    encoder_input_data_test = np.zeros(
        (len(X_test), encoder_max_length),
        dtype='float32')
    for i, (input_text, target_text) in enumerate(zip(X_test, y_test)):
        for t, word in enumerate(nltk.word_tokenize(input_text)):
            encoder_input_data_test[i, t] = encoder_token_index[word]
            
    reverse_target_char_index = dict(
        (i, char) for char, i in decoder_token_index.items())
    
    # Encoder Model
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # Decoder Model
    # Decoder States
    decoder_state_input_h = Input(shape=(emb_size,))
    decoder_state_input_c = Input(shape=(emb_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Decoder Embedding with pre trained weights
    decoder_emb = Embedding(decoder_num_tokens, emb_size)(decoder_inputs) 

    # Decoder LSTM with Pretrained weights
    decoder_outputs2, state_h2, state_c2 = LSTM(lstm_hidden_size, return_sequences=True, return_state=True)(decoder_emb, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    # Decoder Dense with Pretrained Weights
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    # Decoder Final Model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)
            
    actual, predicted = list(), list()
    for seq_index in range(len(encoder_input_data_test)):
        input_seq = encoder_input_data_test[seq_index: seq_index + 1]
        decoded_sent = decode_sequence(input_seq)
        predicted.append(nltk.word_tokenize(decoded_sent))
        actual.append([nltk.word_tokenize(y_test[seq_index])])
        
    print("BLEU Score on Test Set : ")
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    
    