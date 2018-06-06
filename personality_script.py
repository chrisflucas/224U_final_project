#####################################################
# This file sets up the pipeline for and analyzes
# text in string format to judge it in 10 
# emotional categories using the LIWC 2007 database
# as obtained by Chris Potts in the Piazza forum 
# for CS 224U at Stanford University.
# No right are held for the usage of the LIWC dataset
# outisde of a Stanford class environment.
#
# Author: Ramin Ahmari (Jun 2018)
####################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np 
from collections import defaultdict
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import sys, collections
sys.path.append("/Users/raminahmari/Classes/CS 224U/snorkel")
import os
import re
from snorkel import SnorkelSession
from models import Section, Statement, ReconSentence, ReconDocument
from snorkel import SnorkelSession
import hearing_parser
import csv

doc_feature_dict = {}

class LIWC_Extractor():

	rudeagress_words = []
	emotive_words = []
	perceptive_words = []
	social_words = []
	friendly_words = []
	negative_words = []
	positive_words = []
	confrontational_words = []
	indecisive_words = []
	selfish_words = []
	ps = PorterStemmer()

	def __init__(self):
		self.doc_mappings = {} 						#caching system
		self.commissioner_thresholds = None 		#statistic thresholds for commissioners
		self.inmate_thresholds = None 				#statistic thresholds for inmates

	def _determine_speaker(self, sentence, commissioner):
		''' Decides if we need to toggle our understanding of who is speaking '''
		if 'PRESIDING COMMISSIONER' in sentence: return True
		if 'INMATE' in sentence: return False
		return commissioner

	def _count_occurrences(self, arr, sentence):
		''' 
			Sums up over the number of times any phrase in arr occurs in sentence.
			Ex. sentence = 'hi I apologize and I am sorry', arr = APOLOGIZE then
			this returns 2 for 'apologize' and 'sorry'. 
		'''
		return sum([1 for phrase in arr if phrase in sentence])

	def setup_personality_reader(self):

		df = pd.read_csv('LIWC2007.csv')
		headers = list(df)
		LIWC_dict = {}
		for header in headers:
			LIWC_dict[header] = []
			for elem in df[header]:
				if pd.isnull(elem) == False:
					if elem[-2:] == ".*":
						elem = elem[:-2]
					LIWC_dict[header].append(elem)

		self.agression_words = LIWC_dict["Anger"]
		self.agression_words.append(LIWC_dict["Anger.1"])
		self.swear_words = LIWC_dict["Swear"]
		self.sexual_words = LIWC_dict["Sexual"]
		self.rudeagress_words = self.agression_words
		self.rudeagress_words.append(self.swear_words)
		self.rudeagress_words.append(self.sexual_words)


		self.emotive_words = LIWC_dict["Affect"]
		self.emotive_words.append(LIWC_dict["Affect.1"])
		self.emotive_words.append(LIWC_dict["Affect.2"])
		self.emotive_words.append(LIWC_dict["Affect.3"])
		self.emotive_words.append(LIWC_dict["Affect.4"])
		self.emotive_words.append(LIWC_dict["Affect.5"])
		self.emotive_words.append(LIWC_dict["Feel"])

		self.perceptive_words = LIWC_dict["Percept"]
		self.perceptive_words.append(LIWC_dict["Percept.1"])

		self.team_words = LIWC_dict["We"]
		self.social_words = LIWC_dict["Social"]
		self.social_words.append(LIWC_dict["Social.1"])
		self.social_words.append(LIWC_dict["Social.2"])
		self.social_words.append(self.team_words)

		self.friendly_words = LIWC_dict["Friends"]

		self.negative_words = LIWC_dict["Negemo"]
		self.negative_words.append(LIWC_dict["Negemo.1"])
		self.negative_words.append(LIWC_dict["Negemo.2"])
		self.negative_words.append(LIWC_dict["Negemo.3"])
		self.death_words = LIWC_dict["Death"]
		self.sad_words = LIWC_dict["Sad"]
		self.negative_words.append(self.death_words)
		self.negative_words.append(self.sad_words)

		self.positive_words = LIWC_dict["Posemo"]
		self.positive_words.append(LIWC_dict["Posemo.1"])
		self.positive_words.append(LIWC_dict["Posemo.2"])

		self.confrontational_words = LIWC_dict["You"]

		self.indecisive_words = LIWC_dict["Tentat"]

		self.selfish_words = LIWC_dict["I"]


	def analyze_text(self, text, doc, commissioner):
		words = word_tokenize(text)
		stemmed_words = []
		for word in words:
			stemmed_words.append(self.ps.stem(word))
		emotion_dict = defaultdict(int)
		#print(self.rudeagress_words)
		#print(stemmed_words)

		for word in stemmed_words:
			if word in self.rudeagress_words:
				emotion_dict["Rude/Agressive"] = emotion_dict["Rude/Agressive"] + 1
			if word in self.emotive_words:
				emotion_dict["Emotive"] = emotion_dict["Emotive"] + 1
			if word in self.perceptive_words:
				emotion_dict["Perceptive"] = emotion_dict["Perceptive"] + 1
			if word in self.social_words:
				emotion_dict["Social"] = emotion_dict["Social"] + 1
			if word in self.friendly_words:
				emotion_dict["Friendly"] = emotion_dict["Friendly"] + 1
			if word in self.negative_words:
				emotion_dict["Negative"] = emotion_dict["Negative"] + 1
			if word in self.positive_words:
				emotion_dict["Positive"] = emotion_dict["Positive"] + 1
			if word in self.confrontational_words:
				emotion_dict["Confrontational"] = emotion_dict["Confrontational"] + 1
			if word in self.indecisive_words:
				emotion_dict["Indecisive"] = emotion_dict["Indecisive"] + 1
			if word in self.selfish_words:
				emotion_dict["Selfish"] = emotion_dict["Selfish"] + 1

		rudeagress_count = emotion_dict["Rude/Agressive"]
		emotive_count = emotion_dict["Emotive"]
		perceptive_count = emotion_dict["Perceptive"]
		social_count = emotion_dict["Social"]
		friendly_count = emotion_dict["Friendly"]
		negative_count = emotion_dict["Negative"]
		positive_count = emotion_dict["Positive"]
		confrontational_count = emotion_dict["Confrontational"]
		indecisive_count = emotion_dict["Indecisive"]
		selfish_count = emotion_dict["Selfish"]

		overall_count = rudeagress_count + emotive_count + perceptive_count + social_count + friendly_count + negative_count + positive_count + confrontational_count + indecisive_count + selfish_count
		overall_count = float(overall_count)
		overall_count += 1

		rudeagress_percentage = int(rudeagress_count/overall_count * 100)
		emotive_percentage = int(emotive_count/overall_count * 100)
		perceptive_percentage = int(perceptive_count/overall_count * 100)
		social_percentage = int(social_count/overall_count * 100)
		friendly_percentage = int(friendly_count/overall_count * 100)
		negative_percentage = int(negative_count/overall_count * 100)
		positive_percentage = int(positive_count/overall_count * 100)
		confrontational_percentage = int(confrontational_count/overall_count * 100)
		indecisive_percentage = int(indecisive_count/overall_count * 100)
		selfish_percentage = int(selfish_count/overall_count * 100)

		if commissioner:
			doc_feature_dict[doc] = [rudeagress_percentage, emotive_percentage, perceptive_percentage, social_percentage, friendly_percentage, negative_percentage, positive_percentage, confrontational_percentage, indecisive_percentage, selfish_percentage]
		else:
			doc_feature_dict[doc] = doc_feature_dict[doc] + [rudeagress_percentage, emotive_percentage, perceptive_percentage, social_percentage, friendly_percentage, negative_percentage, positive_percentage, confrontational_percentage, indecisive_percentage, selfish_percentage]

		'''
		print("Text Psychologically Perceived As\n")
		print("" + str(rudeagress_percentage) + "% Rude/Aggressive\n")
		print("" + str(emotive_percentage) + "% Emotive\n")
		print("" + str(perceptive_percentage) + "% Perceptive\n")
		print("" + str(social_percentage) + "% Social\n")
		print("" + str(friendly_percentage) + "% Friendly\n")
		print("" + str(negative_percentage) + "% Negative\n")
		print("" + str(positive_percentage) + "% Positive\n")
		print("" + str(confrontational_percentage) + "% Confrontational\n")
		print("" + str(indecisive_percentage) + "% Indecisive\n")
		print("" + str(selfish_percentage) + "% Selfish\n")
		'''


	def evaluate_doc(self, doc):
		#if doc.id in self.doc_mappings: 
		#	if print_stats:
		#		print('Commissioner Counts') 
		#		print(self.doc_mappings[doc.id][0])
		#		print('Inmate Counts')
		#		print(self.doc_mappings[doc.id][1])
		#	return self.doc_mappings[doc.id]


		print("Starting Analysis of", end=" ")
		print(doc)
		commissioner = None
		commissioner_sentences = []
		inmate_sentences = []
		old_progress = -10

		for idx, section in enumerate(doc.sections):
			print("TOTAL PROGRESS: ", end=" ")
			print("" + str(int((idx+1)*100/float(len(doc.sections)))) + "%")
			print("SECTION " + str(idx) + " PROGRESS:")
			for idx_2, statement in enumerate(section.statements):
				progress = int((idx_2+1)*100/float(len(section.statements)))
				if progress >= (old_progress + 10):
					print("" + str(progress) + "%")
					old_progress = progress
				for sentence in statement.sentences:
					joined_sentence = ' '.join(sentence.words)
					commissioner = self._determine_speaker(joined_sentence, commissioner)
					if commissioner == None: continue

					joined_sentence = joined_sentence.lower()
					if commissioner:
						commissioner_sentences.append(joined_sentence)
					else:
						inmate_sentences.append(joined_sentence)
			old_progress = 0
	
		#self.doc_mappings[doc.id] = (commissioner_report, inmate_report)
		print("\n")
		commissioner_text = ' '.join(commissioner_sentences)
		inmate_text = ' '.join(inmate_sentences)
		print("Analyzing Text Bodies ...")
		print("Analyzing Commissioner Text Body ...")
		self.analyze_text(commissioner_text, str(doc), True)
		print("Analyzing Inmate Text Body ...")
		self.analyze_text(inmate_text, str(doc), False)


if __name__ == '__main__':
	session = SnorkelSession()
	docs = session.query(ReconDocument)
	liwc_extractor = LIWC_Extractor()
	liwc_extractor.setup_personality_reader()
	i = 0
	#print(' '.join([' '.join(sentence.words) for sentence in docs[0].sections[1].statements[1].sentences]))
	liwc_extractor.evaluate_doc(docs[0])

	while True:
		print(i)
		try:
			liwc_extractor.evaluate_doc(docs[i])
			i += 1
		except:
			i += 1
			if(i == 1000):
				break

	f = open('category_parser.csv', 'w')
	with f:
		writer = csv.writer(f)
		for key, value in doc_feature_dict.items():
			writer.writerow(value)

