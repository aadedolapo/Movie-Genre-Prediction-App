import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import pickle

#convert text to features
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
# Split data to train and test
from sklearn.model_selection import train_test_split
# classification model
from sklearn.linear_model import LogisticRegression
# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier
# Performance metric

# Load Data
data = pd.read_csv("tmdb-movies.csv")
movies = data[['id', 'original_title', 'overview', 'genres']]
movies.dropna(axis="rows", inplace=True)
movies["genres"] = movies['genres'].apply(lambda x: x.split("|"))

# Function for text cleaning
def clean_text(text):
    # Remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # Remove everything alphabets
    text = re.sub("[^a-zA-Z]"," ",text)
    # Remove whitespaces
    text = ' '.join(text.split())
    # Convert text to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(['one','two','first','second','three','four','five','six'
                       'seven','eight','nine','ten','go','gets','may','also','across',
                       'among','beside','however','yet','within','last','well'])
    stemmer = SnowballStemmer("english")
    cleaned_text = ' '.join([stemmer.stem(word) for word in text.split() if  not word in stop_words])
    return cleaned_text
movies['overview'] = movies['overview'].apply(clean_text)

# Converting Text to Features
# Instantiate the Multilabelbinarizer model
mlb = MultiLabelBinarizer()
mlb.fit(movies['genres'])
# transform target variable
y = mlb.fit_transform(movies['genres'])

vectorizer = CountVectorizer(min_df=4,ngram_range=(1,2),binary=True)
xtrain, xtest, ytrain, ytest = train_test_split(movies['overview'], y, test_size=0.1, random_state=42)

xtrain_vec = vectorizer.fit_transform(xtrain)
xtest_vec = vectorizer.transform(xtest)

lr = LogisticRegression(random_state=0)
clf_lr = OneVsRestClassifier(lr)
clf_lr.fit(xtrain_vec,ytrain)

with open("model.pkl", "wb") as f:
    pickle.dump(clf_lr, f)
model = pickle.load(open("model.pkl", "rb"))