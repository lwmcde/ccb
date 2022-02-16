import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq

import random

from attenvis import AttentionVis
av = AttentionVis()

def train(model, train_french, train_english, eng_padding_index, epoch):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_french: french train data (all data for training) of shape (num_sentences, 14)
    :param train_english: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :return: None
    """

    # NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
    # and english sentences to be used by the decoder
    # - The english sentences passed to the decoder have the last token in the window removed:
    #     [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
    # 
    # - When computing loss, the decoder labels should have the first word removed:
    #     [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP] 
    
    #shuffle
    '''indices = tf.range(start=0, limit=tf.shape(train_french)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    
    train_french = tf.gather(train_french, shuffled_indices)
    train_english = tf.gather(train_english, shuffled_indices)'''
    
    size = train_english.shape[1]
    
    train_english2 = train_english[:,:size-1]

    mask = (train_english != eng_padding_index)
    
    for i in range(0, len(train_english), model.batch_size):
      inputsbatch = train_french[i:i + model.batch_size]
      labelsbatch = train_english2[i:i + model.batch_size]
      labelsbatch2 = train_english[i:i + model.batch_size]
      maskbatch = mask[i:i + model.batch_size]
      
      with tf.GradientTape() as tape:
        probs = model.call(inputsbatch, labelsbatch)
        
        labelsbatch2 = labelsbatch2[:,1:]
        maskbatch = maskbatch[:,1:]
        
        loss = model.loss_function(probs, labelsbatch2, maskbatch)
        
        acc = model.accuracy_function(probs, labelsbatch2, maskbatch)
        
        if i%50 == 0:
  
            print("TRAINING ACCURACY: {}".format(acc))
            print("PROGRESS: {} %".format(int(i * 100/len(train_english))))
            print("EPOCH: {}".format(epoch))
  
      gradients = tape.gradient(loss, model.trainable_variables)
      model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      
'''
@av.test_func
def test(model, test_french, test_english, eng_padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_french: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_english: english test data (all data for testing) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
    e.g. (my_perplexity, my_accuracy)
    """

    # Note: Follow the same procedure as in train() to construct batches of data!
    
    losses = []
    accs = []
    
    size = test_english.shape[1]
    
    test_english2 = test_english[:,:size-1]
    
    mask = (test_english != eng_padding_index)
    
    for i in range(0, len(test_english), model.batch_size):
      inputsbatch = test_french[i:i + model.batch_size]
      labelsbatch = test_english2[i:i + model.batch_size]
      labelsbatch2 = test_english[i:i + model.batch_size]
      maskbatch = mask[i:i + model.batch_size]
      
      probs = model.call(inputsbatch, labelsbatch)
      
      labelsbatch2 = labelsbatch2[:,1:]
      maskbatch = maskbatch[:,1:]
        
      loss = model.loss_function(probs, labelsbatch2, maskbatch)
        
      acc = model.accuracy_function(probs, labelsbatch2, maskbatch)
      
      losses.append(loss)
      accs.append(acc * np.sum(maskbatch))
    
    perplexity = np.exp(np.sum(np.array(losses))/np.sum(mask[:,1:]))
    accuracy = np.sum(np.array(accs))/np.sum(mask[:,1:])
    
    return (perplexity, accuracy)'''

def main():    

    print("Running preprocessing...")
    #train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = get_data('../../data/fls.txt','../../data/els.txt','../../data/flt.txt','../../data/elt.txt')
    train_english, test_english, train_in, test_in, english_vocab, eng_padding_index = get_data('../data/sortedinput.txt','../data/sortedoutput.txt','../data/sortedinput.txt','../data/sortedoutput.txt')
    print("Preprocessing complete.")

    model_args = (FRENCH_WINDOW_SIZE, len(english_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))

    model = Transformer_Seq2Seq(*model_args) 
    
    # TODO:
    # Train and Test Model for 1 epoch.
    
    #model.load_weights('savedmodel')
    #print("LOADED")
    
    epochs = 1
    
    for i in range(epochs):
        train(model, train_in, train_english, eng_padding_index, i)
        #continue
        
    #perplexity, accuracy = test(model, test_french, test_english, eng_padding_index)
    
    #print("Final Accuarcy: " + str(accuracy))
    #print("Final Perplexity: " + str(perplexity))

    # Visualize a sample attention matrix from the test set
    # Only takes effect if you enabled visualizations above
    #av.show_atten_heatmap()
    
    model.save_weights('savedmodel')

    print("SAVED")
    
    #model.load_weights('savedmodel')
    #print("LOADED")

    english_vocab2 = {v: k for k, v in english_vocab.items()}
    
    test2 = ""
    
    while not test2 == "exit":
        test1 = []
        
        test2 = input("You: ")
        padded = pad_french(test2)
        idinput = convert_to_id(english_vocab, padded)
        
        for i in range(15): 
            
            
            padded2 = pad_english(test1)
            
            idinput2 = convert_to_id(english_vocab, padded2)[:,:15]
            
            
            probs = model.call(idinput, idinput2).numpy()
            
            probs[0][i] = probs[0][i]/np.sum(probs[0][i])
            
            indices = np.arange(len(english_vocab))
            
            #choose a response based on probability
            pred = np.random.choice(indices, p=probs[0][i])  
            
            output = english_vocab2[pred]
            
            test1.append(output)
        
        
        chatoutput = []
        
        for word in test1:
            if not (word == "*STOP*" or word == "*PAD*"):
                chatoutput.append(word)
            else:
                break
            
        botmsg = " ".join(x for x in chatoutput)
        print("Chatbuddy: " + botmsg)

if __name__ == '__main__':
    main()
