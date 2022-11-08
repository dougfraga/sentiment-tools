import re 
import pandas as pd
import snscrape.modules.twitter as sntwitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from googletrans import Translator


query = 'Convocação da seleção since:2022-11-07 until:2022-11-08'
maxTweets = 500


# Functions
def get_content(query, maxTweets):
    """Get tweets with scrape method"""
    tweets_list = []
    i = 0
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if i >= maxTweets:
            break
        tweets_list.append([tweet.date, tweet.user.username, tweet.content])
        if i % 10 == 0:
            print(i)
        i += 1  
    df = pd.DataFrame(tweets_list, columns=['date', 'username', 'text'])   
    df.to_csv('tweets_originais.csv', sep='\t', encoding='utf-8')
    return df


def cleanTxt(text):
    """Filter undesirable characters"""
    text = re.sub(r'@[_A-Za-z0-9*".]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r':', '', text)

    return text


def remove_emojis(data):
    "Remove emojis from text"
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def translate_content(df):
    """Translate content for English"""
    lines = []
    for line in df.text:
        print(line)
        trans = Translator()
        trans_text = trans.translate(line, src="pt", dest="en")
        lines.append(trans_text.text)
    df['text_en'] = lines
    return df


def sentAnl(text):
    """Sentiment analysis function"""
    # Load the model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    # Sentiment Analysis
    #encoded_tweet = tokenizer(df['text_en'].to_list(), return_tensors='pt')
    encoded_tweet = tokenizer(text, return_tensors='pt')

    #output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = list(softmax(scores))
    labels = {0:'Negative', 1:'Neutral', 2:'Positive'}
    max_score = scores.index(max(scores))
    analysis = labels[max_score]
    return analysis


if __name__ == '__main__':
    df = get_content(query, maxTweets)
    df['text'] = df['text'].apply(cleanTxt)
    df['text'] = df['text'].apply(remove_emojis)
    df = translate_content(df)
    df['analysis'] = df['text_en'].apply(sentAnl)
    df.to_csv('sentimentos.csv', sep='\t', encoding='utf-8')
    print(df)

    
    