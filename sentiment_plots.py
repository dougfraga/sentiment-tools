import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


df = pd.read_csv('sentimentos.csv', sep='\t', encoding='utf-8')

def count_sentiments(df):
    count = df['analysis'].value_counts(normalize=True).mul(100)
    return count


def count_words(df):
    words = ' '.join(df.text)
    words = words.split(' ')
    count = Counter(words).most_common(10)
    return count

def func(count):
    return "{:.1f}%\n({:d} g)".format(count)

count = count_sentiments(df)
words = count_words(df)
plt.figure()
#plt.pie(count, autopct=lambda pct: func(count),
#                                  textprops=dict(color="w"))
plt.savefig('teste.png')
print(count)
print(words)
