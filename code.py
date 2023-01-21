from collections import defaultdict
from collections import Counter
import itertools
from math import log2
import random
from os.path import exists
import os
import re
import Stemmer
import string
import json
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from gensim.models import LdaModel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy import sparse
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

class Eval():

	def __init__(self, retrieved_path, relevant_path):
		self.relevant_path = relevant_path
		self.retrieved_path = retrieved_path
		self.relevant_docs = self.get_relevant_docs()

	def get_relevant_docs(self):
		relevant_dict = defaultdict(lambda : defaultdict(dict))
		with open(self.relevant_path, "r") as file:
			next(file) # skip header
			for line in file:
				qid, did, rel = line.split(",")
				relevant_dict[int(qid)][int(did)] = int(rel)
		return relevant_dict

	def get_retrieved_results(self, cutoff):
		retrieved_dict = defaultdict(lambda : defaultdict(dict))
		with open(self.retrieved_path) as file:
			next(file) # skip header
			for line in file:
				snum, qnum, dnum, rank, score = line.split(",")
				if cutoff is None:
					cutoff = len(self.relevant_docs[int(qnum)])
				if int(rank) <= cutoff or cutoff == 0:
					retrieved_dict[int(snum)][int(qnum)][int(dnum)] = float(score)
		return retrieved_dict
		
	def get_mean(self, metric_dict):
		for snum, qnum in metric_dict.items():
			metric_dict[snum]["mean"] = sum(qnum.values())/len(qnum)
		return metric_dict

	def get_precision_k(self, cutoff):
		retrieved_results = self.get_retrieved_results(cutoff)
		p_k = defaultdict(lambda : defaultdict(dict))
		for snum, qnum in retrieved_results.items():
			for q, dnum in qnum.items():
				true = 0
				for doc in dnum.keys():
					if doc in self.relevant_docs[q]:
						true += 1
				p_k[snum][q] = true/cutoff
		p_k = self.get_mean(p_k)
		return p_k
		
	def get_recall_k(self, cutoff):
		retrieved_results = self.get_retrieved_results(cutoff)
		r_k = defaultdict(lambda : defaultdict(dict))
		for snum, qnum in retrieved_results.items():
			for q, dnum in qnum.items():
				true = 0
				for doc in dnum.keys():
					if doc in self.relevant_docs[q]:
						true += 1
				r_k[snum][q] = true/len(self.relevant_docs[q])
		r_k = self.get_mean(r_k)
		return r_k
		
	def get_r_precision(self):
		retrieved_results = self.get_retrieved_results(None)
		r_p = defaultdict(lambda : defaultdict(dict))
		for snum, qnum in retrieved_results.items():
			for q, dnum in qnum.items():
				true = 0
				for doc in dnum.keys():
					if doc in self.relevant_docs[q]:
						true += 1
				r_p[snum][q] = true/len(self.relevant_docs[q])
		r_p = self.get_mean(r_p)
		return r_p
		
	def get_AP(self):
		retrieved_results = self.get_retrieved_results(cutoff=0)
		a_p = defaultdict(lambda : defaultdict(dict))
		for snum, qnum in retrieved_results.items():
			for q, dnum in qnum.items():
				true = 0
				retrieved=0
				ap = 0
				for doc in dnum.keys():
					retrieved += 1
					if doc in self.relevant_docs[q]:
						true += 1
						ap += true/retrieved
				a_p[snum][q] = ap/len(self.relevant_docs[q])
		a_p = self.get_mean(a_p)
		return a_p
		
	def get_iDCG(self, cutoff):
		iDCG_k = defaultdict(lambda : defaultdict(dict))
		relevant_dict = self.get_relevant_docs()
		for qid, did in relevant_dict.items():
			dcg = did.pop(list(did.keys())[0])
			for i, doc in enumerate(did.keys(), 2):
				if i > cutoff:
					break
				dcg += did[doc]/log2(i)
			iDCG_k[qid] = dcg
		return iDCG_k
        
	def get_DCG(self, cutoff):
		retrieved_results = self.get_retrieved_results(cutoff)
		relevant_docs = self.relevant_docs
		DCG_k = defaultdict(lambda : defaultdict(dict))
		for snum, qnum in retrieved_results.items():
			for q, dnum in qnum.items():
				dcg = 0
				for i, doc in enumerate(dnum.keys(), 1):
					rel = relevant_docs[q][doc]
					if rel == {}:
						continue
					if i == 1:
						dcg = rel
					elif i > 1:
						dcg += rel/log2(i)
				DCG_k[snum][q] = dcg
		return DCG_k
			
	def get_nDCG(self, cutoff):
		DCG_k = self.get_DCG(cutoff)
		iDCG_k = self.get_iDCG(cutoff)
		nDCG_k = defaultdict(lambda : defaultdict(dict))
		for snum , qnum in DCG_k.items():
			for qid, dcg in qnum.items():
				nDCG_k[snum][qid] = dcg/iDCG_k[qid]
		nDCG_k = self.get_mean(nDCG_k)
		return nDCG_k

	def write_results(self, p10, r50, r_p, AP, nDCG_10, nDCG_20):
		with open("ir_eval.csv", "w") as file:
			file.write("system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n")
			for snum, qnum in p10.items():
				for q in qnum.keys():
					file.write(str(snum) + "," + str(q) + 
						"," + "{:.3f}".format(round(p10[snum][q], 3)) +
						"," + "{:.3f}".format(round(r50[snum][q], 3)) +
						"," + "{:.3f}".format(round(r_p[snum][q], 3)) +
						"," + "{:.3f}".format(round(AP[snum][q], 3)) +
						"," + "{:.3f}".format(round(nDCG_10[snum][q], 3)) +
						"," + "{:.3f}".format(round(nDCG_20[snum][q], 3)) +
						"\n")

class Preprocessor():

	def __init__(self):
        	self.stopwords = self.get_stopwords()

	def trim(self, text):
		text = text.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')
		return text

	def remove_punctuation(self, term):
		x = ''.join(character for character in term if character not in string.punctuation)
		return x

	def tokenise(self, text):
		tokens = re.split('\W+', text)
		lower_text = list(map(str.lower, tokens))
		return list(map(self.remove_punctuation, lower_text))

	def get_stopwords(self):
		with open('stopwords.txt') as f:
			stopwords = f.read().split('\n')
		return stopwords

	def remove_stopwords(self, terms):
		return [term for term in terms if term not in self.stopwords]

	def stem(self, terms):
		stemmer = Stemmer.Stemmer('english')	
		return [stemmer.stemWord(term) for term in terms]

	def remove_empty(self, terms):
		return [term for term in terms if term != '']

	def preprocess(self, raw_text):
		text = self.trim(raw_text)
		tokens = self.tokenise(text)
		words_without_ST = self.remove_stopwords(tokens)
		stemmed = self.stem(words_without_ST)
		no_empties = self.remove_empty(stemmed)
		return no_empties

	def unique_from_array(self, items):
		items_1d = list(itertools.chain.from_iterable(items.values()))
		vocab = {}
		for i, x in enumerate(items_1d):
			if x not in vocab.keys():
				vocab[x] = 0

		for i, k in enumerate(vocab.keys()):
			vocab[k] = i
		return vocab

	# convert word list to dictionary for speeding purposes
	def dictionify(self, items):
		word_dict = {}
		for i, word in enumerate(items):
			word_dict[i] = word
		return word_dict

	def encode_labels(self, labels):
		labels_encoded = []
		for l in labels:
			if l == 'positive':
				labels_encoded.append(0)
			elif l == 'negative':
				labels_encoded.append(1)
			else:
				labels_encoded.append(2)
		#print("encode labels: ", len(labels_encoded))
		return labels_encoded

	def create_count_matrix(self, docs, vocab, mode):
		# print(len(vocab))
		count_mtx = sparse.dok_matrix((len(docs), len(vocab)), dtype='uint8')
		for i in docs.keys():
			count_dict = Counter(docs[i])
			for word in count_dict.keys():
				if mode == 'baseline':
					try:
						# print(count_dict[word])
						count_mtx[i, vocab[word]] = count_dict[word]
					except:
						continue
				elif mode == 'improved':
					try:
						count_mtx[i, vocab[word]] = count_dict[word] * 1000
					except:
						continue
		return count_mtx

	def preprocess_baseline(self, raw_text):
		text = self.trim(raw_text)
		tokens = self.tokenise(text)
		no_empties = self.remove_empty(tokens)
		return no_empties


class Analyze_text():

	def __init__(self, path):
		self.preprocessor = Preprocessor()
		self.corpus = self.read_corpora(path)
		self.all_corpora = self.get_all_corpora()

	def read_corpora(self, path):
		preprocessor = self.preprocessor
		corpus = defaultdict(lambda : defaultdict(dict))
		verse = 0
		current_name = 'OT'
		with open(path, 'r') as file:
			for line in file:
				corpus_name, raw_text = line.split("\t", 1)
				if corpus_name != current_name:
					verse = 0
					current_name = corpus_name
				text = preprocessor.preprocess(raw_text)
				corpus[corpus_name][verse] = text
				verse += 1
		return corpus

	def get_count_N(self, term, corpus_name):
		other_corpus = {"OT", "NT", "Quran"}-{corpus_name}
		corpus = self.corpus
		N00, N01, N10, N11 = 0, 0, 0, 0
		for verse, text in corpus[corpus_name].items():
			if term in text:
				N11 += 1
			else:
				N01 += 1
		for corpus_i in other_corpus:
			for verse, text in corpus[corpus_i].items():
				if term in text:
					N10 += 1
				else:
					N00 += 1
		return [N00, N01, N10, N11]

	def sort_dict(self, unsorted_dict):
		return dict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=True))

	def display_results(self, result_dict):
		for i, item in enumerate(result_dict.items()):
			term = item[0]
			score = item[1]
			print(term + ': ' + str(score))
			if i > 10:
				break
	
	def compute_mutual_chi(self):
		result = defaultdict(lambda : defaultdict(dict))
		result_chi = defaultdict(lambda : defaultdict(dict))
		for corpus_name, verses in self.corpus.items():
			for verse, text in verses.items():
				for word in text:
					if word in result[corpus_name]:
						continue
					N00, N01, N10, N11 = self.get_count_N(word, corpus_name)
					N = N00 + N01 + N10 + N11
					# compute mi
					try:
						eq1 = (N11 / N) * log2((N * N11) / ((N11 + N10) * (N01 + N11)))
					except:
						eq1 = 0
					try:
						eq2 = (N01 / N) * log2((N * N01) / ((N01 + N00) * (N01 + N11)))
					except:
						eq2 = 0
					try:
						eq3 = (N10 / N) * log2((N * N10) / ((N10 + N11) * (N10 + N00)))
					except:
						eq3 = 0
					try:
						eq4 = (N00 / N) * log2((N * N00) / ((N00 + N01) * (N10 + N00)))
					except:
						eq4 = 0
					result[corpus_name][word] = eq1+eq2+eq3+eq4
					# compute chi-squared
					eq_chi1 = N * ((N11 * N00) - (N10 * N01))**2
					eq_chi2 = (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00)
					result_chi[corpus_name][word] = eq_chi1/eq_chi2
			result[corpus_name] = self.sort_dict(result[corpus_name])
			result_chi[corpus_name] = self.sort_dict(result_chi[corpus_name])
			print("-------Mutual Information - ", corpus_name,"-------")
			self.display_results(result[corpus_name])
			print("-------Chi-Squared - ", corpus_name,"-------")
			self.display_results(result_chi[corpus_name])

	def get_all_corpora(self):
		return [text for corpus, verses in self.corpus.items() for verse, text in verses.items()]

	def get_lda_corpus(self):
		corp_dictionary = Dictionary(self.all_corpora)
		return [corp_dictionary.doc2bow(text) for text in self.all_corpora]

	def get_lda(self):
		if exists(datapath('lda_model')):
			#os.remove(datapath('lda_model'))
			return LdaModel.load(datapath('lda_model'))
		corpus = self.get_lda_corpus()
		lda = LdaModel(corpus=corpus, num_topics=20)
		save_loc = datapath('lda_model')
		lda.save(save_loc)
		return lda

	def extract_tokens(self, lda_token):
		ids = {}
		#get token ID : word dictionary to retrieve words
		corp_dictionary = Dictionary(self.all_corpora)
		dictionary = corp_dictionary.token2id
		ks, vs = dictionary.keys(), dictionary.values()
		word_to_id = dict(zip(vs,ks))
		probs = lda_token.replace(' ', '').replace('\"', '').split('+')
		for prob_num in probs:
			prob, num = prob_num.split('*')
			ids[word_to_id[int(num)]] = prob
		ids_sorted = {k: v for k, v in sorted(ids.items(), key=lambda item: item[1], reverse=True)}
		return ids_sorted

	def lda_calc_average_score(self):
		len_ot = len(self.corpus['OT'])
		len_nt = len(self.corpus['NT'])
		len_qu = len(self.corpus['Quran'])
		lda_result = defaultdict(lambda : defaultdict(dict))
		corpus = self.get_lda_corpus()
		lda = self.get_lda()
		lda_distrib = lda.get_document_topics(corpus)
		#add results for each corpus to get average score for each topic
		for i, line in enumerate(lda_distrib):
			line_dict = dict(line)
			if i < len_ot:
				lda_result['OT'][i] = line_dict
			elif len_ot <= i < len_ot + len_nt:
				lda_result['NT'][i] = line_dict
			elif len_ot + len_nt <= i:
				lda_result['Quran'][i] = line_dict

		#set probability to 0 if a topic probability does not appear
		for corpus_name in lda_result.keys():
			for doc in lda_result[corpus_name].keys():
				for topic in range(0, 20):
					try:
						if lda_result[corpus_name][doc][topic] == 0:
        	                    			lda_result[corpus_name][doc][topic] = 0
					except:
						lda_result[corpus_name][doc][topic] = 0

		avg_scores = defaultdict(lambda : defaultdict(dict))

		#calculate average probability 1) sum up the values
		for corpus_name in lda_result.keys():
			for doc in lda_result[corpus_name].keys():
				for topic in lda_result[corpus_name][doc].keys():
					try:
						avg_scores[corpus_name][topic] += lda_result[corpus_name][doc][topic]
					except:
						avg_scores[corpus_name][topic] = lda_result[corpus_name][doc][topic]

		#calculate average probability 2) average the values by the total number of documents in each corpus
		for corpus_name in avg_scores.keys():
			for topic in avg_scores[corpus_name].keys():
				avg_scores[corpus_name][topic] = avg_scores[corpus_name][topic] / len(lda_result[corpus_name])

		#sort each corpus by the probability of each topic candidate
		for corpus_name, topics in avg_scores.items():
			avg_scores[corpus_name] = {k: v for k, v in sorted(topics.items(), key=lambda item: item[1], reverse=True)}

		best_ot = list(avg_scores['OT'].keys())[0]
		best_nt = list(avg_scores['NT'].keys())[0]
		best_quran = list(avg_scores['Quran'].keys())[0]

		topic_ot = lda.print_topic(best_ot)
		topic_nt = lda.print_topic(best_nt)
		topic_quran = lda.print_topic(best_quran)

		tokens_ot = self.extract_tokens(topic_ot)
		tokens_nt = self.extract_tokens(topic_nt)
		tokens_quran = self.extract_tokens(topic_quran)

		print("Best topics and tokens")
		print('OT: ' + str(best_ot))
		print(tokens_nt)
		print('NT: ' + str(best_nt))
		print(tokens_ot)
		print('Quran: ' + str(best_quran))
		print(tokens_quran)

class Classifier():

	def __init__(self, ):
		self.raw_data = self.load_raw_data()
		self.raw_test_data = self.load_raw_test_data()

	def load_raw_data(self):
		with open('train.txt', 'r') as f:
			train_text = f.readlines()
		return train_text

	def load_raw_test_data(self):
		with open('test.txt', 'r') as f:
			test_text = f.readlines()
		return test_text

	def shuffle_and_split(self, split, X, y):
		dataset = list(zip(X.todense(), y))  # zip the count matrix and labels
		random.shuffle(dataset)  # shuffle the cm-label tuples
		if split == 'train':  # if training set is given, split to training and validation
			X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1)
			X_train_sparse = sparse.dok_matrix(X_train)
			X_dev_sparse = sparse.dok_matrix(X_dev)
			return X_train_sparse, X_dev_sparse, y_train, y_dev

		elif split == 'test':
			splitted = [list(t) for t in zip(*dataset)]  # unzip the list of tuples of [(dense_matrix, labels)]
			X_shuffled = splitted[0]
			y_shuffled = splitted[1]
			X_sparse = sparse.dok_matrix(np.concatenate(X_shuffled, axis=0))  # convert back to sparse matrix from dense
			return X_sparse, y_shuffled

	def collect_words_from_raw_text(self, mode, raw_text):
		p = Preprocessor()
		docs = []
		labels = []
		for i,line in enumerate(raw_text):
			_, s, document = line.split('\t', 2)
			if mode == 'baseline':
				docs.append(p.preprocess_baseline(document))
			elif mode == 'improved':
				docs.append(p.preprocess(document))
			labels.append(s)
		return docs, labels

	# create vocabulary using the docs
	def create_vocab(self, docs):
		p = Preprocessor()
		vocab = p.unique_from_array(p.dictionify(docs))  # convert docs to be in dictionary form and create vocab
		return vocab

	def run_count_matrix_creator(self, mode, docs, vocab, labels):
		p = Preprocessor()
		docs = p.dictionify(docs)
		#print(len(labels))
		count_mtx = p.create_count_matrix(docs, vocab, mode)
		encoded_labels = p.encode_labels(labels)
		#print("run count: ", len(encoded_labels))
		return count_mtx, encoded_labels

	def prepare_data(self, mode):
		raw_text = self.raw_data
		raw_test_text = self.raw_test_data
		docs, labels = self.collect_words_from_raw_text(mode, raw_text)
		test_docs, test_labels = self.collect_words_from_raw_text(mode, raw_test_text)
		vocab = self.create_vocab(docs)
		count_mtx, encoded_labels = self.run_count_matrix_creator(mode, docs, vocab, labels)
		count_mtx_test, encoded_labels_test = self.run_count_matrix_creator(mode, test_docs, vocab, test_labels)
		X_train, X_dev, y_train, y_dev = self.shuffle_and_split('train', count_mtx, encoded_labels)
		X_test, y_test = self.shuffle_and_split('test', count_mtx_test, encoded_labels_test)

		# save shuffled and splitted data to disk
		with open('X_train_{}.pkl'.format(mode), 'wb') as f:
			pickle.dump(X_train, f)
		with open('X_test_{}.pkl'.format(mode), 'wb') as f:
			pickle.dump(X_test, f)
		with open('X_dev_{}.pkl'.format(mode), 'wb') as f:
			pickle.dump(X_dev, f)
		with open('y_train_{}.pkl'.format(mode), 'wb') as f:
			pickle.dump(y_train, f)
		with open('y_dev_{}.pkl'.format(mode), 'wb') as f:
			pickle.dump(y_dev, f)
		with open('y_test_{}.pkl'.format(mode), 'wb') as f:
			pickle.dump(y_test, f)
		return X_train, X_dev, X_test, y_train, y_dev, y_test

	def rm(self, mode):
		os.remove('X_train_{}.pkl'.format(mode))
		os.remove('X_test_{}.pkl'.format(mode))
		os.remove('X_dev_{}.pkl'.format(mode))
		os.remove('y_train_{}.pkl'.format(mode))
		os.remove('y_dev_{}.pkl'.format(mode))
		os.remove('y_test_{}.pkl'.format(mode))

	def load_data(self, mode):
		with open('X_train_{}.pkl'.format(mode), 'rb') as f:
			X_train = pickle.load(f)
		with open('X_dev_{}.pkl'.format(mode), 'rb') as f:
			X_dev = pickle.load(f)
		with open('X_test_{}.pkl'.format(mode), 'rb') as f:
			X_test = pickle.load(f)
		with open('y_train_{}.pkl'.format(mode), 'rb') as f:
			y_train = pickle.load(f)
		with open('y_dev_{}.pkl'.format(mode), 'rb') as f:
			y_dev = pickle.load(f)
		with open('y_test_{}.pkl'.format(mode), 'rb') as f:
			y_test = pickle.load(f)
		return X_train, X_dev, X_test, y_train, y_dev, y_test

	def get_metrics_str(self, mode, split, y_true, y_pred):
		precision = self.precision(y_true, y_pred)
		recall = self.recall(y_true, y_pred)
		f1 = self.f1_score(y_true, y_pred)
		metrics_string = ''
		metrics_string += mode + ',' + split + ','  # add system and split
		metrics_string += str(precision[0]) + ',' + str(recall[0]) + ',' + str(f1[0]) + ','
		metrics_string += str(precision[1]) + ',' + str(recall[1]) + ',' + str(f1[1]) + ','
		metrics_string += str(precision[2]) + ',' + str(recall[2]) + ',' + str(f1[2]) + ','
		metrics_string += str(precision['macro']) + ',' + str(recall['macro']) + ',' + str(f1['macro'])
		return metrics_string

	def load_svm_model(self, mode, classifier='svm'):
		with open('svm_model_{}.pkl'.format(mode), 'rb') as f:
			model = pickle.load(f)
		return model

	def evaluate_predictions(self, mode):
		print(mode)
		model = self.load_svm_model(mode)
		if not exists('X_train_{}.pkl'.format(mode)):
			self.prepare_data(mode)
		#else:
			#self.rm(mode)
			#self.prepare_data(mode)
		X_train, X_dev, X_test, y_train, y_dev, y_test = self.load_data(mode)
		sc = StandardScaler(with_mean=False)
		sc.fit(X_train)
		X_train_std = sc.transform(X_train)
		svm = SVC(kernel='linear', random_state=1, C=0.1)
		svm.fit(X_train_std, y_train)
		y_train_pred = svm.predict(X_train)
		y_dev_pred = svm.predict(X_dev)
		y_test_pred = svm.predict(X_test)

		if not exists('classification.csv'):
			with open('classification.csv', 'a') as f:
				f.write('system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro\n')

		with open('classification.csv', 'a') as f:
			f.write(self.get_metrics_str(mode, 'train', y_train, y_train_pred) + '\n')
			f.write(self.get_metrics_str(mode, 'dev', y_dev, y_dev_pred) + '\n')
			f.write(self.get_metrics_str(mode, 'test', y_test, y_test_pred) + '\n')

	def train_model(self, mode):
		if mode == 'baseline':
			c = 1000
		elif mode == 'improved':
			c = 10
		X_train, X_dev, X_test, y_train, y_dev, y_test = self.prepare_data(mode)
		model = SVC(C=c, verbose=True)  # init sklearn.svm.SVC
		with open('svm_model_{}.pkl'.format(mode), 'wb') as f:
			pickle.dump(model, f)
		self.evaluate_predictions(mode)


	# initialise metrics dictinary for easier additions
	def init_metric_dict(self):
		lookup = defaultdict(lambda : defaultdict(dict))
		for i in range(3):
			lookup[i]['tp'] = 0
			lookup[i]['fp'] = 0
			lookup[i]['fn'] = 0
		return lookup

	def precision(self, y_true, y_pred):
		# initialise metrics dictionary
		lookup = self.init_metric_dict()
		for true, pred in zip(y_true, y_pred):
			#print(true, pred)
			if true == pred:
				lookup[pred]['tp'] += 1
			else:
				lookup[pred]['fp'] += 1

		#print(lookup)
		precisions = {}
		for i in range(3):
			try:
				precisions[i] = round(lookup[i]['tp'] / (lookup[i]['tp'] + lookup[i]['fp']), 3)
			except:
				precisions[i] = 0
		precisions['macro'] = round((precisions[0] + precisions[1] + precisions[2]) / 3, 3)
		return precisions

	def recall(self, y_true, y_pred):
		# initialise metrics dictionary
		lookup = self.init_metric_dict()
		for true, pred in zip(y_true, y_pred):
			if true == pred:
				lookup[true]['tp'] += 1
			else:
				lookup[true]['fn'] += 1
		recall = {}
		for i in range(3):
			try:
				recall[i] = round(lookup[i]['tp'] / (lookup[i]['tp'] + lookup[i]['fn']), 3)
			except:
				recall[i] = 0
		recall['macro'] = round((recall[0] + recall[1] + recall[2]) / 3, 3)
		return recall

	def f1_score(self, y_true, y_pred):
		precision = self.precision(y_true, y_pred)
		recall = self.recall(y_true, y_pred)
		f1 = {}
		for i in range(3):
			try:
				f1[i] = round(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]), 3)
			except:
				f1[i] = 0
		f1['macro'] = round((f1[0] + f1[1] + f1[2]) / 3, 3)
		return f1


##################################################################
ir = Eval("system_results.csv", "qrels.csv")
p10 = ir.get_precision_k(10)
r50 = ir.get_recall_k(50)
r_precision = ir.get_r_precision()
ap = ir.get_AP()
nDCG_10 = ir.get_nDCG(10)
nDCG_20 = ir.get_nDCG(20)
ir.write_results(p10, r50, r_precision, ap, nDCG_10, nDCG_20)
##################################################################

#################################################################
a = Analyze_text("train_and_dev.tsv")
a.compute_mutual_chi()
a.lda_calc_average_score()
#################################################################

#################################################################
c = Classifier()
c.train_model('baseline')
c.train_model('improved')
#################################################################

