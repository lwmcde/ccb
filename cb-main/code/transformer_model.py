import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis

av = AttentionVis()

class Transformer_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

        ######vvv DO NOT CHANGE vvv##################
        super(Transformer_Seq2Seq, self).__init__()

        self.english_vocab_size = english_vocab_size # The size of the english vocab

        self.french_window_size = french_window_size # The french window size
        self.english_window_size = english_window_size # The english window size
        ######^^^ DO NOT CHANGE ^^^##################


        # TODO:
        # 1) Define any hyperparameters
        # 2) Define embeddings, encoder, decoder, and feed forward layers
        
        self.optimizer = tf.optimizers.Adam()

        # Define batch size and optimizer/learning rate
        self.batch_size = 64
        self.embedding_size = 216

        # Define english and french embedding layers:
            
        self.E = tf.Variable(tf.random.normal([self.english_vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32, name="e2"))
        
        # Create positional encoder layers
        
        self.PE = transformer.Position_Encoding_Layer(self.english_window_size, self.embedding_size)
        
        # Define encoder and decoder layers:
            
        self.encoder = transformer.Transformer_Block(self.embedding_size, False, True)
        
        self.decoder = transformer.Transformer_Block(self.embedding_size, True, True)
        
        self.encoder2 = transformer.Transformer_Block(self.embedding_size, False, True)
        
        self.decoder2 = transformer.Transformer_Block(self.embedding_size, True, True)
        
        self.encoder3 = transformer.Transformer_Block(self.embedding_size, False, True)
        
        self.decoder3 = transformer.Transformer_Block(self.embedding_size, True, True)
        
        self.encoder4 = transformer.Transformer_Block(self.embedding_size, False, True)
        
        self.decoder4 = transformer.Transformer_Block(self.embedding_size, True, True)
        
        self.encoder5 = transformer.Transformer_Block(self.embedding_size, False, True)
        
        self.decoder5 = transformer.Transformer_Block(self.embedding_size, True, True)
        
        self.encoder6 = transformer.Transformer_Block(self.embedding_size, False, True)
        
        self.decoder6 = transformer.Transformer_Block(self.embedding_size, True, True)
    
        # Define dense layer(s)
        
        self.D1 = tf.keras.layers.Dense(self.english_vocab_size, name="d1")

    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
        
        embedding1 = tf.nn.embedding_lookup(self.E, encoder_input)
        embedding2 = tf.nn.embedding_lookup(self.E, decoder_input)
    
        # TODO:
        #1) Add the positional embeddings to french sentence embeddings
        
        embedding1 = self.PE(embedding1)
        
        #2) Pass the french sentence embeddings to the encoder
        
        encoded = self.encoder(embedding1)
        encoded = self.encoder2(encoded)
        
        encoded = self.encoder3(encoded)
        encoded = self.encoder4(encoded)
        
        encoded = self.encoder5(encoded)
        encoded = self.encoder6(encoded)
        
        
        #3) Add positional embeddings to the english sentence embeddings
        
        embedding2 = self.PE(embedding2)
        
        #4) Pass the english embeddings and output of your encoder, to the decoder
        
        decoded = self.decoder(embedding2, encoded)
        decoded = self.decoder2(decoded, encoded)
        decoded = self.decoder3(decoded, encoded)
        decoded = self.decoder4(decoded, encoded)
        decoded = self.decoder5(decoded, encoded)
        decoded = self.decoder6(decoded, encoded)
        
        #5) Apply dense layer(s) to the decoder out to generate probabilities
        
        logits = self.D1(decoded)
        
        probs = tf.nn.softmax(logits)
    
        return probs

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy
        
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy


    def loss_function(self, prbs, labels, mask):
        prbs = tf.boolean_mask(prbs, mask)
        labels = tf.boolean_mask(labels, mask)

        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))

    @av.call_func
    def __call__(self, *args, **kwargs):
        return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)