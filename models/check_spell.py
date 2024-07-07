import numpy as np
import re
from collections import Counter
from tensorflow.keras.saving import load_model
import tensorflow as tf
import string
import config as args
from nltk.util import ngrams

class CheckSpell(tf.Module):
    def __init__(self, args):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
        super().__init__()
        self.args = args
        self.model = load_model(args.MODEL_PATH)
        self.accepted_char = list(string.digits + ''.join(args.LETTERS))

    def encoder_data(self, text):
        text = "\x00" + text
        x = np.zeros((self.args.MAXLEN, len(self.args.ALPHABET)))
        for i, c in enumerate(text[:self.args.MAXLEN]):
            x[i, self.args.ALPHABET.index(c)] = 1
        if i < self.args.MAXLEN - 1:
            for j in range(i + 1, self.args.MAXLEN):
                x[j, 0] = 1
        return x

    def decoder_data(self, x):
        x = x.argmax(axis=-1)
        return ''.join(self.args.ALPHABET[i] for i in x)


    def nltk_ngrams(self, words):
        if len(words.split()) < self.args.NGRAM:
            words = words + (self.args.NGRAM - len(words.split())) * ' \x00 '
        return ngrams(words.split(), self.args.NGRAM)
    

    def guess(self, ngram):
        text = ' '.join(ngram)
        preds = self.model.predict(np.array([self.encoder_data(text)]), verbose=0)
        return self.decoder_data(preds[0]).strip('\x00')
    
    def __call__(self, sentence):
        for i in sentence:
            if i not in self.accepted_char:
                sentence = sentence.replace(i, " ")
        ngrams = list(self.nltk_ngrams(sentence))
        guessed_ngrams = [self.guess(ngram) for ngram in ngrams]
        candidates = [Counter() for _ in range(len(guessed_ngrams) + self.args.NGRAM - 1)]
        for nid, ngram in enumerate(guessed_ngrams):
            for wid, word in enumerate(re.split(' +', ngram)):
                candidates[nid + wid].update([word])
        try:
            output = ' '.join(c.most_common(1)[0][0] for c in candidates)
        except:
            output = ' '
        return output

