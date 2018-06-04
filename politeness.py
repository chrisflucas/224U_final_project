from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, collections
sys.path.append("/Users/cflucas/cs224U_final_project/snorkel")
import os
import numpy as np
import re
from snorkel import SnorkelSession
from models import Section, Statement, ReconSentence, ReconDocument
from snorkel import SnorkelSession
import hearing_parser


# Define some global variables representing our lexicons. 
APOLOGIZE = ['sorry', 'oops', 'woops', 'excuse me', 'forgive me', 'apologies', 'apologize', 'my bad', 'my fault']
ASK_AGENCY = ['do me a favor', 'let me', 'allow me', 'can i', 'should i', 'may i', 'might i', 'could i']
GIVE_AGENCY = ['let you', 'allow you', 'you can', 'you may', 'you could']
GRATITUDE = ['thank', 'thanks', 'appreciate']
PLEASE = ['please']
KEYS = ['apologize', 'ask_agency', 'give_agency', 'gratitude', 'please']


class PolitenessExtractor():

	def __init__(self):
		self.doc_mappings = {} # caching system
		self.commissioner_thresholds = None # computed thresholds
		self.inmate_thresholds = None


	def compute_score(self, doc, print_stats=True):
		'''
			Given a document, this method iterates over all sentences in the document, 
			determines who is speaking, and then counts the number of times there are any 
			occurrences of any words or phrases in any of the aforementioned lexicons. 

			We maintain separate counts for each lexicon and then increment the total number 
			of occurrences of those lexical words over the entire document for either the inmate
			or the commissioner speaker. 

			@param doc: Parole Hearing document
			@param print_stats (opt): boolean
			@return tuple: dict of commissioner lexicon counts, dict of inmate lexicon counts
		'''

		def _determine_speaker(sentence, commissioner):
			''' Decides if we need to toggle our understanding of who is speaking '''
			if 'PRESIDING COMMISSIONER' in sentence: return True
			if 'INMATE' in sentence: return False
			return commissioner

		def _count_occurrences(arr, sentence):
			''' 
				Sums up over the number of times any phrase in arr occurs in sentence.
				Ex. sentence = 'hi I apologize and I am sorry', arr = APOLOGIZE then
				this returns 2 for 'apologize' and 'sorry'. 
			'''
			return sum([1 for phrase in arr if phrase in sentence])

		if doc.id in self.doc_mappings: 
			if print_stats:
				print('Commissioner Counts') 
				print(self.doc_mappings[doc.id][0])
				print('Inmate Counts')
				print(self.doc_mappings[doc.id][1])
			return self.doc_mappings[doc.id]

		sections = doc.sections
		statements = sections[0].statements
		sentences = statements[0].sentences
		apologize_comm=ask_agency_comm=give_agency_comm=gratitude_comm=please_comm=0
		apologize_inm=ask_agency_inm=give_agency_inm=gratitude_inm=please_inm=0
		commissioner = None

		for section in doc.sections:
			for statement in section.statements:
				for sentence in statement.sentences:
					joined_sentence = ' '.join(sentence.words)
					commissioner = _determine_speaker(joined_sentence, commissioner)
					if commissioner == None: continue

					apologize_count = _count_occurrences(APOLOGIZE, joined_sentence.lower())
					ask_agency_count = _count_occurrences(ASK_AGENCY, joined_sentence.lower())
					give_agency_count = _count_occurrences(GIVE_AGENCY, joined_sentence.lower())
					gratitude_count = _count_occurrences(GRATITUDE, joined_sentence.lower())
					please_count = _count_occurrences(PLEASE, joined_sentence.lower())

					if commissioner:
						apologize_comm += apologize_count
						ask_agency_comm += ask_agency_count
						give_agency_comm += give_agency_count
						gratitude_comm += gratitude_count
						please_comm += please_count
					else:
						apologize_inm += apologize_count
						ask_agency_inm += ask_agency_count
						give_agency_inm += give_agency_count
						gratitude_inm += gratitude_count
						please_inm += please_count

		commissioner_report = self.generate_report('COMMISSIONER', apologize_comm, ask_agency_comm, \
									give_agency_comm, gratitude_comm, please_comm, print_stats)
		inmate_report = self.generate_report('INMATE', apologize_inm, ask_agency_inm, give_agency_inm, \
									gratitude_inm, please_inm, print_stats)
		
		self.doc_mappings[doc.id] = (commissioner_report, inmate_report)
		return commissioner_report, inmate_report

	def calculate_feature_thresholds(self):
		'''
			This iterates over all of the documents the Extractor has seen
			and computes the average counts for apology, ask agency, etc. for 
			both the commissioner and inmate separately. (10 averages, 5 for 
			commissioner apology, commissioner ask agency, etc. and 5 for 
			inmate apology, inmate ask agency, etc.)

			Stores the values in self.commissioner_thresholds and self.inmate_thresholds
			so that the Extractor can start using these values to featurize documents.

			@param self: PolitenessExtractor
			@return None 
		'''
		commissioner_total_counts = collections.defaultdict(lambda: 0.0)
		inmate_total_counts = collections.defaultdict(lambda: 0.0)


		for doc_id, count_tuple in self.doc_mappings.items():
			commissioner_counts, inmate_counts = count_tuple

			for k in KEYS:
				commissioner_total_counts[k] += commissioner_counts[k]
				inmate_total_counts[k] += inmate_counts[k]

		number_of_documents = float(len(self.doc_mappings.keys()))

		self.commissioner_thresholds = collections.defaultdict(lambda: 0.0)
		self.inmate_thresholds = collections.defaultdict(lambda: 0.0)
		
		for k in KEYS:
			self.commissioner_thresholds[k] = commissioner_total_counts[k]/number_of_documents
			self.inmate_thresholds[k] = inmate_total_counts[k]/number_of_documents

		print( 'Done calculating averages, use these as thresholds for '+\
				'featurizing politeness as a function of apologize, ask_agency, etc.' )

		print(self.commissioner_thresholds)
		print(self.inmate_thresholds)

	def featurize_document(self, doc):
		'''
			Given a document and an instance of a PolitenessExtractor, this method will 
			return a tuple of dictionary feature representations of the commissioner and 
			the inmate in the document where the value is 1 if the number of occurrences 
			of 'apology', 'ask agency', etc. is greater than or equal to the average 
			occurrence count for the 'training data' -- the set of documents that have 
			been included in the PolitenessExtractor doc_mappings field.


			Note: Instance of PolitenessExtractor must already have calculated the feature
			thresholds to use -- if not, an error is thrown. 

			@param doc: Parole Hearing document
			@return tuple: commissioner feature vector, inmate feature vector 
		'''
		if self.commissioner_thresholds is None or self.inmate_thresholds is None:
			raise Exception('Must calculate feature thresholds first.  \
				Use PolitenessExtractor.calculate_feature_thresholds() method.')

		commissioner_counts, inmate_counts = self.compute_score(doc, print_stats=True)
		featurized_vector_comm = collections.defaultdict(lambda: 0)
		featurized_vector_inm = collections.defaultdict(lambda: 0)
		
		for k in KEYS:
			featurized_vector_comm[k] = 1 if commissioner_counts[k] >= self.commissioner_thresholds[k] else 0
			featurized_vector_inm[k] = 1 if inmate_counts[k] >= self.inmate_thresholds[k] else 0

		return featurized_vector_comm, featurized_vector_inm


	def print_report(self, speaker, apologize, ask_agency, give_agency, \
						gratitude, please):
		'''
			Given the politeness stats and speaker (commissioner or inmate typically), this prints
			an easily digestible snapshot of the lexical counts. 

			@param speaker: string
			@param apologize ... please: int
		'''
		intro = '='*25 + ' ' + speaker + ' ' + '='*25
		print( intro )
		print( '='*5 + ' Apologize: {}'.format(apologize) )
		print( '='*5 + ' Ask Agency: {}'.format(ask_agency) )
		print( '='*5 + ' Give Agency: {}'.format(give_agency) )
		print( '='*5 + ' Gratitude: {}'.format(gratitude) )
		print( '='*5 + ' Please: {}'.format(please) )
		print( '='*len(intro) )
		print( '\n' )

	def generate_report(self, speaker, apologize, ask_agency, give_agency, \
						gratitude, please, print_stats):
		'''
			Given the politeness stats, speaker and option to print extra statistic 
			information, this method creates a map representing the counts of each 
			kind of 'polite' language.

			@param speaker: string
			@param apologize ... please: int
			@return dict mapping lexicon name to int occurrence count. 
		'''

		if print_stats:
			self.print_report(speaker, apologize, ask_agency, give_agency, \
							gratitude, please)

		return {
			'apologize': apologize,
			'ask_agency': ask_agency, 
			'give_agency': give_agency,
			'gratitude': gratitude,
			'please': please
		}


if __name__ == '__main__':
	session = SnorkelSession()
	docs = session.query(ReconDocument)
	print(docs)
	i = 0
	p = PolitenessExtractor()

	'''
		Stupid code because I don't know how to get the length 
		of the array of documents returned by the query. 

		i.e. for i in range(len(docs)): does not work because
			 len(docs) is not a thing...
	'''
	while True:
		try:
			p.compute_score(docs[i], print_stats=False)
			i+=1
		except:
			print('i is: {}'.format(i))
			p.calculate_feature_thresholds()
			break





