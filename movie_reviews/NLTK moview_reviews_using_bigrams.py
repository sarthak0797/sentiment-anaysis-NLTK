import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
 

random.seed()

document = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


random.shuffle(document)

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

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent after using bigrams is:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features()
