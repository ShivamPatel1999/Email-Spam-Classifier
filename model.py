import os
from collections import Counter
import numpy as np
import pickle

folder = 'email/'
files = os.listdir(folder)
len(files)
emails = [folder + file for file in files]
words = []
for email in emails:
    f = open(email, encoding='latin-1')
    blob = f.read()
    words = words+blob.split(" ")
for i in range(len(words)):
    if not words[i].isalpha():
        words[i] = ""

word_dict = Counter(words)

del word_dict[""]
word_dict = word_dict.most_common(3000)
#for i in word_dict:
  #  print(i[0])
features = []
labels = []
for email in emails:
    f = open(email, encoding='latin-1')
    blob = f.read().split(" ")
    data = []
    for i in word_dict:
        data.append(blob.count(i[0]))
    features.append(data)

    if 'spam' in email:
        labels.append(1)
    if 'ham' in email:
        labels.append(0)

features = np.array(features)
labels = np.array(labels)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=9)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
print(classifier.fit(X_train, y_train))

pickle.dump(classifier, open('model.pkl', 'wb'))