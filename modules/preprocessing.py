import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

class Preprocessing:

    def __init__(self):
        self.stemmer = PorterStemmer()

        intab = string.punctuation
        outtab = "                                "
        self.trantab = str.maketrans(intab, outtab)

    def stem_tokens(self, tokens, stem):
        stemmed = []
        for item in tokens:
            stemmed.append(stem.stem(item))
        # print(stemmed)
        return stemmed

    def tokenize(self, text):
        tokens = word_tokenize(text=text)
        stems = self.stem_tokens(tokens, self.stemmer)
        return ' '.join(stems)

    def train_preprocess(self, train_data, test_data, idata):
        corpus = []
        test_set = []
        i_set = []
        count_vect = CountVectorizer()
        tf_idf_transformer = TfidfTransformer()

        for text in train_data:
            text = text.lower()
            text = text.translate(self.trantab)
            text = self.tokenize(text)
            corpus.append(text)

        train_data_counts = count_vect.fit_transform(corpus)
        train_data_tf_idf = tf_idf_transformer.fit_transform(train_data_counts)

        for text in test_data:
            text = text.lower()
            text = text.translate(self.trantab)
            text = self.tokenize(text)
            test_set.append(text)

        test_data_counts = count_vect.transform(test_set)
        test_data_tf_idf = tf_idf_transformer.transform(test_data_counts)

        for text in idata:
            text = text.lower()
            text = text.translate(self.trantab)
            text = self.tokenize(text)
            i_set.append(text)

        i_data_counts = count_vect.transform(i_set)
        i_data_tf_idf = tf_idf_transformer.transform(i_data_counts)

        return train_data_tf_idf, test_data_tf_idf, i_data_tf_idf

