import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import requests
import string
import nltk
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from wordcloud import WordCloud


stop_words = stopwords.words("english") # 불용어 확인용
ps = PorterStemmer()                    # 어간추출용 

def clean_text(text):
    # 문장부호 제거, 소문자로 변환
    text = "".join([word.lower() for word in text if word not in string.punctuation])   # string.punctuation - 문장부호
    tokens = re.split('\W+', text)
    # 불용어가 아니면 어간 추출
    text = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return text


def Tokenizer():
    # 단어로 쪼개서 빈도 테이블과 그래프 그리기
    text = "Now, I understand that because it's an election season expectations for what we will achieve this year are low But, Mister Speaker, I appreciate the constructive approach that you and other leaders took at the end of last year to pass a budget and make tax cuts permanent for working\
families. So I hope we can work together this year on some bipartisan priorities like criminal justice reform and helping people who are battling prescription drug abuse and heroin abuse. So, who knows, we might surprise the cynics again"
    # word_tokinize split sentence into words
    words = word_tokenize(text)
    # FreqDist counts frequency of each word
    fdist = FreqDist(words)
    fdist.most_common(2)
    fdist.plot(30, cumulative = True)
    plt.show()

    # 문장으로 나눈 다음 단어로 쪼개서 문장별 단어의 갯수 확인 및 막대그래프 그리기
    sentence = sent_tokenize(text)  # sent_tokinize split entire text to sentences
    word_num = []
    for s in sentence:
        word_num.append(len(word_tokenize(s)))

    sns.barplot(x=np.arange(1,4), y=word_num)
    plt.title('Number of Words by Sentence')
    plt.show()

    # 주소의 텍스트를 사용하여 불용어처리, 어간추출, 문장부호 제거
    url = 'http://programminghistorian.github.io/ph-submissions/assets/basic-text-processing-in-r/sotu_text/236.txt'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    text = soup.get_text()
    words = clean_text(text)
    fdist = FreqDist(words)
    fdist.plot(20)
    plt.show()

    # 워드클라우드 그리기
    wc = WordCloud(background_color = 'white').generate_from_frequencies(fdist)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    Tokenizer()
