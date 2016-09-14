import nltk
import os

location = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

pos_path = location + "/positive.txt"
neg_path = location + "/negative.txt"
stop_path = location + "/stopwords.txt"

pw = open(pos_path.encode("utf8"))
pos_txt = pw.readlines()

nw = open(neg_path.encode("utf8"))
neg_txt = nw.readlines()

sw = open(stop_path.encode("utf8"))
stp_txt = sw.readlines()

pos_list = []
neg_list = []
stop_list = []

for i in range(0,len(pos_txt)):
    pos_list.append('positive')

for i in range(0,len(neg_txt)):
    neg_list.append('negative')

for i in range(0,len(stp_txt)):
    stop_list.append('stop')

pos_tag = zip(pos_txt, pos_list)
neg_tag = zip(neg_txt, neg_list)

tag_sentence = pos_tag + neg_tag

sentences = []

for (sen, tag) in tag_sentence:
    word_filter = []
    for i in sen.split():
        word_filter.append(i)
    sentences.append((word_filter, tag))

def getwords(sentences):
    allwords = []
    for (words, sentiment) in sentences:
        allwords.extend(words)
    return allwords

def getwordfeatures(listofsentences):
    wordfreq = nltk.FreqDist(listofsentences)
    words = wordfreq.keys()
    return words

print(getwordfeatures(getwords(sentences)))

wordlist = getwordfeatures(getwords(sentences))
wordlist = [i for i in wordlist if not i in stp_txt]

def feature_extractor(doc):
    docwords = set(doc)
    features = {}
    for i in wordlist:
        features['contains(%s)' % i] = (i in docwords)
    return features

training_set = nltk.classify.apply_features(feature_extractor, sentences)
classifier = nltk.NaiveBayesClassifier.train(training_set)

# print(classifier.show_most_informative_features(n=30))

while True:
    input = raw_input("Enter any sentence or 'exit' to quit:")
    if input == 'exit':
        break
    elif input == 'informfeatures':
        print(classifier.show_most_informative_features(n=30))
        continue
    else:
        input = input.lower()
        input = input.split()
        print(classifier.classify(feature_extractor(input)))

pw.close()
nw.close()
sw.close()






