import pandas as pd
from collections import Counter
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import math
from nltk.stem.porter import *
from nltk.corpus import stopwords
import nltk

#nltk.download()
stemmer = PorterStemmer()
vectorizer = CountVectorizer()
tfidf = TfidfTransformer()

def remove_punctuation(text):
    return re.sub(ur"[^\w\d'\s]+",'',text)


def normalize(review):
	stops = set(stopwords.words("english"))
	review = review.lower()
	review = remove_punctuation(review)
	#split_tokens = review.split()
	split_tokens = nltk.word_tokenize(review)
	split_tokens = [w for w in split_tokens if not w in stops]
	tokens = [stemmer.stem(token) for token in split_tokens]
	return tokens

def Bow(reviews):
	   
    aux_idf = [" ".join(rev) for rev in reviews]
    matrix = vectorizer.fit_transform(aux_idf)
    #idf = vectorizer.idf_
    #return dict(zip(vectorizer.get_feature_names(), idf))
    return matrix


###MAIN

df = pd.read_csv('train.tsv', sep="\t", skiprows =[0], header = None)
df.columns = ["Phraseid", "Sentenceid", "Phrase", "Sentiment"]
df.drop(df.columns[:2], axis = 1, inplace = True)

reviews = df['Phrase'].tolist()
clasif = df['Sentiment'].tolist()


reviews = [review.decode('utf-8') for review in reviews]
#words guarda lista de listas, cada sublista es un review en tokens
norm_revs = [normalize(review) for review in reviews]
#generamos nuestros tfidfs
count = Bow(norm_revs)

ttfidf = tfidf.fit_transform(count)
#####################################
clasif_count = (Counter(clasif).most_common())
clasif_count.sort(key = itemgetter(0))

#calculo las probabilidades a priori
aux_pp = [elem[1] for elem in clasif_count]
prior_probs = [elem/float(len(reviews)) for elem in aux_pp]
#######################################

classifier = MultinomialNB(alpha=1.0, class_prior=prior_probs, fit_prior=True)
nbcls  = classifier.fit(ttfidf, clasif)


#################################################################
test = pd.read_csv('test.tsv', sep="\t", skiprows =[0], header = None)
test.columns = ["Phraseid", "Sentenceid", "Phrase"]
test.drop(test.columns[1], axis = 1, inplace = True)

#####################


tests = test['Phrase'].tolist()
ids = test['Phraseid'].tolist()

norm_test = [normalize(test) for test in tests]
norm_test = [" ".join(test) for test in norm_test]
print "clasificacion"
vec = vectorizer.transform(norm_test)
sents = nbcls.predict(vec)


data = zip(ids,sents)
ans = pd.DataFrame(data)
ans.columns = ["PhraseId", "Sentiment"]

ans.to_csv('Sentiments3.csv', sep=',', index=False)