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

def decode_sequence_attention(input_seq):
    decoded_sentence = ''
    # encoder input
    encoder_input = input_seq.reshape(1,encoder_max_length)
    # create blank matrix for decoder data
    decoder_input = np.zeros(shape=(len(encoder_input), decoder_max_length))
    # update the first element with the start index
    decoder_input[:,0] = decoder_token_index[SOS]
    # loop through the max length
    for i in range(1, decoder_max_length):
        # Predict the index of the next word
        output = model_attention.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    # Reverse to get the sentence from the word index
    for d in decoder_input[:,0:][0]:
        if (reverse_target_char_index[d] == EOS):
            decoded_sentence += ' '+reverse_target_char_index[d]
            break
        decoded_sentence += ' '+reverse_target_char_index[d]
    return decoded_sentence.strip()

if __name__ == "__main__":
    
    EOS = '_EOS'
    SOS = 'SOS_'

    if (len(sys.argv) < 2) or (len(sys.argv) > 3):
        print("Usage:")
        print("\tnmt_att.py data_path [pretrained_model_path]")
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
    
    # Define Encoder and Decoder Input
    encoder_input = Input(shape=(encoder_max_length,))
    decoder_input = Input(shape=(decoder_max_length,))
    
    # Embedding (non-trainable)
    encoder = Embedding(encoder_num_tokens, emb_size , weights=[embedding_matrix] , 
                        input_length=encoder_max_length , mask_zero=True , trainable=False)(encoder_input) 
    # Encoder LSTM
    encoder = LSTM(lstm_hidden_size, return_sequences=True, go_backwards = True, unroll=True)(encoder)
    # Last word from encoder to fed to the decoder input
    encoder_last = encoder[:,-1,:]
    
    # Decoder Embedding with Normal Keras Embedding (will be trained)
    decoder = Embedding(decoder_num_tokens, emb_size, 
                        mask_zero=True, input_length=decoder_max_length)(decoder_input) 
    # Decoder LSTM
    decoder = LSTM(lstm_hidden_size, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])

    # Attention dot product of the encoder and decoder weights
    attention = dot([decoder, encoder], axes=[2, 2])
    # Attention Softmax to get the optimum weight
    attention = Activation('softmax', name='attention')(attention)

    # Context dot product of pre attention weight with encoder
    context = dot([attention, encoder], axes=[2,1])
    
    # Decoder combined context 
    decoder_combined_context = concatenate([context, decoder])

    # Time Distributed Layer with activation and softmax
    output = TimeDistributed(Dense(emb_size, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(decoder_num_tokens, activation="softmax"))(output)

    # Final Model
    model_attention = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    model_attention.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])
    
    if MODEL_PATH:
        model_attention.load_weights(MODEL_PATH)
        print("Loaded model from disk")
    else:
        model_attention.fit([encoder_input_data_train, decoder_input_data_train], decoder_target_data_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.05)
        model_attention.save('model_attention.h5')
    
    encoder_input_data_test = np.zeros(
        (len(X_test), encoder_max_length),
        dtype='float32')
    for i, (input_text, target_text) in enumerate(zip(X_test, y_test)):
        for t, word in enumerate(nltk.word_tokenize(input_text)):
            encoder_input_data_test[i, t] = encoder_token_index[word]
            
    reverse_target_char_index = dict(
        (i, char) for char, i in decoder_token_index.items())
            
    actual_attention, predicted_attention = list(), list()
    for seq_index in range(len(encoder_input_data_test)):
        input_seq = encoder_input_data_test[seq_index: seq_index + 1]
        decoded_sent = decode_sequence_attention(input_seq)
        predicted_attention.append(nltk.word_tokenize(decoded_sent))
        actual_attention.append([nltk.word_tokenize(y_test[seq_index])])
        
    print("BLEU Score on Test Set : ")
    print('BLEU-1: %f' % corpus_bleu(actual_attention, predicted_attention, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual_attention, predicted_attention, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual_attention, predicted_attention, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual_attention, predicted_attention, weights=(0.25, 0.25, 0.25, 0.25)))
    
    