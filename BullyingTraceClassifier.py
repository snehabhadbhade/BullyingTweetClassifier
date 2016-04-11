# Classifier training for identifying Bullying Traces
import re
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import csv
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.naive_bayes import BernoulliNB

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''



lemmatiser = WordNetLemmatizer()
stemmer = SnowballStemmer("english")


def tokenizer(text):

    lemmatized_words = []
    tokens = word_tokenize(text)
    tokens_pos = pos_tag(tokens)
    count = 0
    for token in tokens:
        pos = tokens_pos[count]
        pos = get_wordnet_pos(pos[1])
        if pos != '':
            lemma = lemmatiser.lemmatize(token, pos)
        else:
            lemma = lemmatiser.lemmatize(token)
        lemmatized_words.append(lemma)
        count+=1
    return lemmatized_words


#Training text has the labelled tweets
with open("Train_Bully","r") as f:
    train_text = f.readlines()
f.close()

#Test set has the unlabelled tweets
with open("Test_Bully","r") as f:
    test_text = f.readlines()
f.close()

# Convert Training data to fit the model
train_data = []
train_target = np.ndarray(shape = len(train_text), dtype ='int64')
pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
labels = []

with open("Train_Bully","r") as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        tweet = row[0]
        tweet = tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
        tweet = re.sub('@[^\s]+','',tweet)
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = tweet.strip('\'"')
        tweet = pattern.sub(r"\1\1",tweet)
        tweet = re.sub("[^a-zA-Z]"," ",tweet)
        train_data.append(tweet)
        labels.append(int(row[1].strip('"')))

f.close()

train_target = np.asarray(labels, dtype='int64')
test_data = []
test_target = []

with open("Test_Bully","r") as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        tweet = row[0]
        tweet = tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
        tweet = re.sub('@[^\s]+','',tweet)
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = tweet.strip('\'"')
        tweet = pattern.sub(r"\1\1",tweet)
        tweet = re.sub("[^a-zA-Z]"," ",tweet)
        test_data.append(tweet)
        labels.append(int(row[1].strip('"')))

test_target = np.asarray(labels, dtype='int64')


parameters = {'vect__ngram_range': [(1, 1), (1, 2),(1,3)],'tfidf__use_idf': (True, False),}

linear_pipe = Pipeline([('vect', CountVectorizer(encoding="latin-1", ngram_range=(1,1), tokenizer=tokenizer, analyzer='word')),('tfidf', TfidfTransformer(use_idf="True")), ('clf', BernoulliNB()),])

gs_svm = GridSearchCV(linear_pipe, parameters, n_jobs=-1, scoring="accuracy")

gs_svm = gs_svm.fit(train_data, train_target)

predicted = gs_svm.predict(test_data)

#Predicted will have the labels that our classifier identified
with open("Predicted_bully", "w") as f:
    header = 'Text,Predicted_label'+"\n"
    f.write(header)
    count=0
    for label in predicted:
        tweet = test_data[count]
        row = str(tweet) +","+str(label)+"\n"
        f.write(row)
        count+=1

f.close()


print(np.mean(predicted == test_target))
print('Best score: %0.3f' % gs_svm.best_score_)
print('Best parameters set:')
best_parameters = gs_svm.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))











