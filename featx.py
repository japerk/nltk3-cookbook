import collections
from nltk.corpus import stopwords, reuters
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

def bag_of_words(words):
	'''
	>>> bag_of_words(['the', 'quick', 'brown', 'fox'])
	{'quick': True, 'brown': True, 'the': True, 'fox': True}
	'''
	return dict([(word, True) for word in words])

def bag_of_words_not_in_set(words, badwords):
	'''
	>>> bag_of_words_not_in_set(['the', 'quick', 'brown', 'fox'], ['the'])
	{'quick': True, 'brown': True, 'fox': True}
	'''
	return bag_of_words(set(words) - set(badwords))

def bag_of_non_stopwords(words, stopfile='english'):
	'''
	>>> bag_of_non_stopwords(['the', 'quick', 'brown', 'fox'])
	{'quick': True, 'brown': True, 'fox': True}
	'''
	badwords = stopwords.words(stopfile)
	return bag_of_words_not_in_set(words, badwords)

def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
	'''
	>>> bag_of_bigrams_words(['the', 'quick', 'brown', 'fox'])
	{'brown': True, ('brown', 'fox'): True, ('the', 'quick'): True, 'quick': True, ('quick', 'brown'): True, 'the': True, 'fox': True}
	'''
	bigram_finder = BigramCollocationFinder.from_words(words)
	bigrams = bigram_finder.nbest(score_fn, n)
	return bag_of_words(words + bigrams)

def bag_of_words_in_set(words, goodwords):
	return bag_of_words(set(words) & set(goodwords))

def label_feats_from_corpus(corp, feature_detector=bag_of_words):
	label_feats = collections.defaultdict(list)
	
	for label in corp.categories():
		for fileid in corp.fileids(categories=[label]):
			feats = feature_detector(corp.words(fileids=[fileid]))
			label_feats[label].append(feats)
	
	return label_feats

def split_label_feats(lfeats, split=0.75):
	train_feats = []
	test_feats = []
	
	for label, feats in lfeats.items():
		cutoff = int(len(feats) * split)
		train_feats.extend([(feat, label) for feat in feats[:cutoff]])
		test_feats.extend([(feat, label) for feat in feats[cutoff:]])
	
	return train_feats, test_feats

def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
	word_fd = FreqDist()
	label_word_fd = ConditionalFreqDist()
	
	for label, words in labelled_words:
		for word in words:
			word_fd[word] += 1
			label_word_fd[label][word] += 1
	
	n_xx = label_word_fd.N()
	high_info_words = set()
	
	for label in label_word_fd.conditions():
		n_xi = label_word_fd[label].N()
		word_scores = collections.defaultdict(int)
		
		for word, n_ii in label_word_fd[label].items():
			n_ix = word_fd[word]
			score = score_fn(n_ii, (n_ix, n_xi), n_xx)
			word_scores[word] = score
		
		bestwords = [word for word, score in word_scores.items() if score >= min_score]
		high_info_words |= set(bestwords)
	
	return high_info_words

def reuters_high_info_words(score_fn=BigramAssocMeasures.chi_sq):
	labeled_words = []
	
	for label in reuters.categories():
		labeled_words.append((label, reuters.words(categories=[label])))
	
	return high_information_words(labeled_words, score_fn=score_fn)

def reuters_train_test_feats(feature_detector=bag_of_words):
	train_feats = []
	test_feats = []
	
	for fileid in reuters.fileids():
		if fileid.startswith('training'):
			featlist = train_feats
		else: # fileid.startswith('test')
			featlist = test_feats
		
		feats = feature_detector(reuters.words(fileid))
		labels = reuters.categories(fileid)
		featlist.append((feats, labels))
	
	return train_feats, test_feats

if __name__ == '__main__':
	import doctest
	doctest.testmod()