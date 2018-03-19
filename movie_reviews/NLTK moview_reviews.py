import nltk
import random
from nltk.corpus import movie_reviews

random.seed()

document = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())

def find(documents):
    temp = {}
    word = set(documents)

    for w in word_features:
        temp[w] = (w in word)
        
    return temp

features = [(find(w) , ids) for (w , ids) in document]

training_set = features[:900] + features[1000:1900]
testing_set = features[900:1000] + features[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

