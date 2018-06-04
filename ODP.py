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

	pos_tags_train = []

	for dialog in dialogArr:
		pos_tags_train.append(nltk.pos_tag(word_tokenize(dialog)))

	only_pos_tags_train = []

	for pos_list in pos_tags_train:
		only_pos_tags_train.append(' '.join([x[1] for x in pos_list]))

	# return in stringified format - for sklearn ... might not have to do this
	return only_pos_tags_train


def get_mixed(dialogs, pos_tags_train):
	mixed = []

	# first tokenize both - then iterate through each to create mixed
	for dialog, pos_tag in zip(dialogs, pos_tags_train):
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
	#dialog_train = dialog_train + dialog_test
	#dialog_annotations_train = dialog_annotations_train + dialog_annotations_test 

	# randomize - this doesn't feel scientifically accurate - might want to remove.
	# dialog_train, dialog_annotations_train = shuffle(dialog_train, dialog_annotations_train, random_state = 42)

	# get yvals for training and 
	y_train, dialog_train = get_yvals(dialog_train)
	y_test, dialog_test = get_yvals(dialog_test)

	# get dialog acts
	dialog_acts_train = get_dialog_acts(dialog_annotations_train)
	dialog_acts_test = get_dialog_acts(dialog_annotations_test)

	# get part of speech tags
	pos_tags_train = get_pos(dialog_train)
	pos_tags_test = get_pos(dialog_test)

	# get mixed class
	mixed_train = get_mixed(dialog_train, pos_tags_train)
	mixed_test = get_mixed(dialog_test, pos_tags_test) 

	# Vectorize the data - into sparse matricies 
	pos_vectorizer = CountVectorizer(tokenizer=my_tokenizer, ngram_range=(2, 2))
	pos_matrix_train = pos_vectorizer.fit_transform(pos_tags_train)
	pos_matrix_test = pos_vectorizer.transform(pos_tags_test)

	mixed_vectorizer = CountVectorizer(ngram_range=(4,4))
	mixed_matrix_train = mixed_vectorizer.fit_transform(mixed_train)
	mixed_matrix_test = mixed_vectorizer.transform(mixed_test)

	# sparse concatenation

	# this section one-hot encodes the dialog acts into a matrix

	le = LabelEncoder()
	dialog_acts_train = le.fit_transform(dialog_acts_train)
	dialog_acts_test = le.transform(dialog_acts_test)

	temp_matr1 = []
	for dialog_act in dialog_acts_train:
		temp_matr1.append([dialog_act])
	temp_matr2 = []
	for dialog_act in dialog_acts_test:
		temp_matr2.append([dialog_act])

	enc = OneHotEncoder()
	dialog_acts_train = enc.fit_transform(temp_matr1)
	dialog_acts_test = enc.transform(temp_matr2)

	# this concatenates the matricies together
	X_train = hstack([pos_matrix_train, mixed_matrix_train, dialog_acts_train]).toarray()
	X_test = hstack([pos_matrix_test, mixed_matrix_test, dialog_acts_test]).toarray()

	scalar = StandardScaler(with_mean=False)
	X_train = scalar.fit_transform(X_train)
	X_test = scalar.transform(X_test)


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
