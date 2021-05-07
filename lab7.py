import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer


def Tokenize():
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(ngram_range = (1,1), lowercase = True, stop_words = 'english', tokenizer = token.tokenize)

    text = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']
    x = cv.fit_transform(text)
    x_df = pd.DataFrame(x.toarray(), columns = cv.get_feature_names())
    print(x_df)

    tfidf = TfidfVectorizer()
    x1 = tfidf.fit_transform(text)
    x1_df = pd.DataFrame(x1.toarray(), columns = tfidf.get_feature_names())
    print('\n', x1_df)


if __name__ == '__main__':
    Tokenize()