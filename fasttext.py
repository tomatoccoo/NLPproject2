import fasttext

# Skipgram model
model = fasttext.skipgram('fasttext_data.txt', 'fasttext_word_model')
print(model['solved']) # list of words in dictionary

classifier = fasttext.supervised('fasttext_train_data.txt', 'fasttext_classifier_model')
result = classifier.test('fasttext_test_data.txt')
