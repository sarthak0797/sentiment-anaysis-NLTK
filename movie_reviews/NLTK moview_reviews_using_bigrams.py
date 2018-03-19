import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import pickle
from nltk.corpus import movie_reviews
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


document = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words)[:5000]

def find(documents):
    temp = {}
    word = set(documents)
    n = 2
    score_fn=BigramAssocMeasures.chi_sq

    bigram_finder = BigramCollocationFinder.from_words(word)
    bigrams = bigram_finder.nbest(score_fn, n)
    
    for w in word_features:
        temp[w] = (w in word )
    for w in bigrams:
        temp[w] = True
        
    return temp

features = [(find(w) , ids) for (w , ids) in document]

training_set = features[:900] + features[1000:1900]
testing_set = features[900:1000] + features[1900:]

#Run next 4 instructions if you're running the script for first time 
classifier = NaiveBayesClassifier.train(training_set)
classify_buffer = open('movie_reviews.pickle', 'wb')
pickle.dump(classifier, classify_buffer)
classify_buffer.close()
#Comment above 4 instructions if you've run the script once

#Run next 3 instructions if you're running the script second time onwards
#classify_buffer = open('movie_reviews.pickle', 'rb')
#classifier = pickle.load(classify_buffer)
#classify_buffer.close()
#Comment aboce 3 instructions while running the script for first time

print("Classifier accuracy after using bigrams is percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
