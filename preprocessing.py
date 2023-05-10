import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical

class Preprocessing:

    def __init__(self, train_data, test_data, data):
        self.train_data, self.test_data, self.data = train_data, test_data, data
        self.split_dataset_into_x_y()
        self.encode_y()
        self.turn_x_y_into_sequences()

    def split_dataset_into_x_y(self):
        self.X_train = self.train_data["tweet"]
        self.X_test = self.test_data["tweet"]

        self.y_train = self.train_data["label"]
        self.y_test = self.test_data["label"]

    # clear tweets and get them ready for the model
    def remove_emojis(tweet):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r' ', tweet)

    def clean_tweet(self, tweet):
        # Remove multiple spaces
        tweet = re.sub(r'\s+', ' ', tweet)

        # remove punctuations
        punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
        translator = str.maketrans('', '', punctuations)
        tweet = tweet.translate(translator)

        # remove Tashkeel
        tweet = self.remove_diacritics(tweet)

        # remove longation
        tweet = re.sub("[إأآا]", "ا", tweet)
        tweet = re.sub("ى", "ي", tweet)
        tweet = re.sub("ؤ", "ء", tweet)
        tweet = re.sub("ئ", "ء", tweet)
        tweet = re.sub("ة", "ه", tweet)
        tweet = re.sub("گ", "ك", tweet)

        # Remove Stopwords
        stop_words = set(stopwords.words('arabic'))
        pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
        tweet = pattern.sub('', tweet)

        # remove hashtags and @usernames
        tweet = re.sub(r"(#[\d\w\.]+)", '', tweet)
        tweet = re.sub(r"(@[\d\w\.]+)", '', tweet)

        # tekenization using nltk
        tweet = word_tokenize(tweet)

        return tweet

    def remove_diacritics(self, tweet):
        arabic_diacritics = re.compile("""    | # Shadda
                                     َ    | # Fatha
                                     ً    | # Tanwin Fath
                                     ُ    | # Damma
                                     ٌ    | # Tanwin Damm
                                     ِ    | # Kasra
                                     ٍ    | # Tanwin Kasr
                                     ْ    | # Sukun
                                     ـ     # Tatwil/Kashida
                                 """, re.VERBOSE)
        return arabic_diacritics.sub(r' ', tweet)

    def get_x_y(self):
        return (self.X_train, self.X_test, self.y_train, self.y_test)

    def turn_x_y_into_sequences(self):
        texts = [' '.join(self.clean_tweet(text)) for text in self.data["tweet"]]
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(texts)

        # Use that tokenizer to transform the text messages in the training and test sets
        self.X_train = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_test = self.tokenizer.texts_to_sequences(self.X_test)

        # Pad the sequences so each sequence is the same length
        self.X_train = pad_sequences(self.X_train, 50)
        self.X_test = pad_sequences(self.X_test, 50)

        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)

        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)


    def encode_y(self):
        # encoding the labels column
        encoding = {'neg': 0, 'pos': 1}

        self.y_train = [encoding[x] for x in self.train_data["label"]]
        self.y_test = [encoding[x] for x in self.test_data["label"]]

        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def get_tokenizer_len(self):
        return len(self.tokenizer.index_word) + 1