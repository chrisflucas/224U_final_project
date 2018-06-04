import os
import re
import nltk
from nltk import word_tokenize
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


remove_begin = re.compile('^M[0-9]+\.[0-9]+\. ')

def readInXMLs(directory_in_str):

	dialog_annotations = []
	dialog = []

	directory = os.fsencode(directory_in_str)
	
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		filename = os.path.join(os.fsdecode(directory), filename)

		with open(filename, 'r', encoding = "ISO-8859-1") as fp:
			whole_file = fp.readlines()

		with open(filename, 'r', encoding = "ISO-8859-1") as fp:

			for i, line in enumerate(fp):
				if line.startswith('M') and line[1].isnumeric():
					line = line.strip('\n')
					line = re.sub(remove_begin, '', line)
					dialog.append(line)

					flag = False
					for e in range(26):
						if whole_file[i+e].startswith('[') and flag == False:
							tag_line = whole_file[i+e]
							flag = True

					dialog_annotations.append(tag_line.strip('\n'))


	return dialog, dialog_annotations


# return array of 0/1 class dep on if statement contains '[Y]'
def get_yvals(dialogArr):
	yvals = []

	for dialog in dialogArr:
		if '[Y]' in dialog:
			yvals.append(1)
		else:
			yvals.append(0)

	for i in range(len(dialogArr)):
		if '[Y]' in dialogArr[i]:
			dialogArr[i] = dialogArr[i].strip('[Y]')

	return yvals, dialogArr

def get_dialog_acts(dialog_annotations):
	dialog_acts = []

	# Request-Information, Inform, Conventional, Request-Action, Commit
	# \[(\w+):

	for dialog_annotation in dialog_annotations:
		dialog_act = re.findall('\[(\w+-?\w*)', dialog_annotation)[0]
		if dialog_act == 'Inform-AnswerOffline':
			dialog_acts.append('Inform')
		else:
			dialog_acts.append(dialog_act)

	return dialog_acts



def get_pos(dialogArr):

	pos_tags = []

	for dialog in dialogArr:
		pos_tags.append(nltk.pos_tag(word_tokenize(dialog)))

	only_pos_tags = []

	for pos_list in pos_tags:
		only_pos_tags.append(' '.join([x[1] for x in pos_list]))

	# return in stringified format - for sklearn ... might not have to do this
	return only_pos_tags


def get_mixed(dialogs, pos_tags):
	mixed = []

	# first tokenize both - then iterate through each to create mixed
	for dialog, pos_tag in zip(dialogs, pos_tags):
		tokenized_dialog = word_tokenize(dialog)
		tokenized_pos = [x[1] for x in nltk.pos_tag(tokenized_dialog)]

		# we put in open-class pos tokens, else just the dialog
		tmp = []
		for i in range(len(tokenized_pos)):
			pos = tokenized_pos[i]
			if pos.startswith(('NN', 'VB', 'RB', 'JJ')):
				tmp.append(pos)
			else:
				tmp.append(tokenized_dialog[i])

		mixed.append(' '.join(tmp))

	return mixed


def my_tokenizer(s):
	return s.split()


def preprocess():
	# we have 2 more utterances than the paper... not sure where they are - and the numbers are off (not 5%)
	dialog_train, dialog_annotations_train = readInXMLs('/Users/MichaelSmith/Desktop/PowerAnnotations_V1.0/AnnotatedThreads/Train')
	dialog_test, dialog_annotations_test = readInXMLs('/Users/MichaelSmith/Desktop/PowerAnnotations_V1.0/AnnotatedThreads/Test')

	# combine dialog, dialog_annotations
	dialog_full = dialog_train + dialog_test
	dialog_annotations_full = dialog_annotations_train + dialog_annotations_test 

	# randomize - this doesn't feel scientifically accurate - might want to remove.
	# dialog_full, dialog_annotations_full = shuffle(dialog_full, dialog_annotations_full, random_state = 42)

	# get yvals for training and 
	Y, dialog_full = get_yvals(dialog_full)

	# get dialog acts
	dialog_acts_full = get_dialog_acts(dialog_annotations_full)

	# get part of speech tags
	pos_tags = get_pos(dialog_full)

	# get mixed class
	mixed_train = get_mixed(dialog_full, pos_tags) 

	# Vectorize the data - into sparse matricies 
	pos_vectorizer = CountVectorizer(tokenizer=my_tokenizer, ngram_range=(2, 2))
	pos_matrix = pos_vectorizer.fit_transform(pos_tags)

	mixed_vectorizer = CountVectorizer(tokenizer=my_tokenizer, ngram_range=(4,4))
	mixed_matrix = mixed_vectorizer.fit_transform(mixed_train)

	# sparse concatenation
	pos_matrix = coo_matrix(pos_matrix)
	mixed_matrix = coo_matrix(mixed_matrix)

	# this section one-hot encodes the dialog acts into a matrix
	le = LabelEncoder()
	dialog_acts_full = le.fit_transform(dialog_acts_full)

	temp_matr1 = []
	for dialog_act in dialog_acts_full:
		temp_matr1.append([dialog_act])

	enc = OneHotEncoder()
	dialog_acts_full = enc.fit_transform(temp_matr1)

	# this concatenates the matricies together
	X = hstack([pos_matrix, mixed_matrix, dialog_acts_full]).toarray()


	scalar = StandardScaler(with_mean=False)
	X = scalar.fit_transform(X)

	# separate back into test and train:
	X_train = X[0:1091, :]
	X_test = X[1091:, :]
	y_train = Y[0:1091]
	y_test = Y[1091:]

	return X_train, X_test, y_train, y_test


def main():
	
	X_train, X_test, y_train, y_test = preprocess()

	clf = SVC(kernel='linear', probability=True)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print("F1-score: ", f1_score(y_test, y_pred))

	clf = LinearSVC()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print("F1-score: ", f1_score(y_test, y_pred))




if __name__ == "__main__":
	main()
