from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re 
import pickle
#convert text to features
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
# Split data to train and test
from sklearn.model_selection import train_test_split

app = Flask(__name__)

data = pd.read_csv("tmdb-movies.csv")
movies = data[['id', 'original_title', 'overview', 'genres']]
movies.dropna(axis="rows", inplace=True)
def to_list(x):
    return x.split("|")
movies['genres'] = movies['genres'].apply(to_list)

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
movies['overview'] = movies["overview"].apply(lambda x:clean_text(x))

# Instantiate the Multilabelbinarizer model
mlb = MultiLabelBinarizer()
mlb.fit(movies['genres'])
# transform target variable
y = mlb.fit_transform(movies['genres'])
# Converting Text to Features
vectorizer = CountVectorizer(min_df=4,ngram_range=(1,2),binary=True)
# split dataset into training and test set
xtrain, xtest, ytrain, ytest = train_test_split(movies['overview'], y, test_size=0.1, random_state=42)

xtrain_vec = vectorizer.fit_transform(xtrain)
xtest_vec = vectorizer.transform(xtest)
model = pickle.load(open("model.pkl", "rb"))

def predict_genre(movie_plot):
    q = clean_text(movie_plot)
    q_vec = vectorizer.transform([q])
    q_pred = model.predict(q_vec)
    result = mlb.inverse_transform(q_pred)
    if len(result[0]) == 0:
        return "Sorry! Unable to Predict a Genre"
    else:
        return result

# Home page
@app.route("/")
def Home():
    return render_template("index.html")

# Where predict method is defined
@app.route("/Predict", methods = ["POST"])
def predict():
    movie_plot = request.form.get("movie_plot")
    prediction = predict_genre(movie_plot)
    return render_template("index.html", prediction_text="{}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
