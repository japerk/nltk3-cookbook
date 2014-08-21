import collections, itertools
from nltk import metrics
from nltk.classify import util, ClassifierI, MultiClassifierI
from nltk.probability import FreqDist

def precision_recall(classifier, testfeats):
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)
	
	for i, (feats, label) in enumerate(testfeats):
		refsets[label].add(i)
		observed = classifier.classify(feats)
		testsets[observed].add(i)
	
	precisions = {}
	recalls = {}
	
	for label in classifier.labels():
		precisions[label] = metrics.precision(refsets[label], testsets[label])
		recalls[label] = metrics.recall(refsets[label], testsets[label])
	
	return precisions, recalls

class MaxVoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
		self._labels = sorted(set(itertools.chain(*[c.labels() for c in classifiers])))
	
	def labels(self):
		return self._labels
	
	def classify(self, feats):
		counts = FreqDist()
		
		for classifier in self._classifiers:
			counts[classifier.classify(feats)] += 1
		
		return counts.max()

class MultiBinaryClassifier(MultiClassifierI):
	def __init__(self, *label_classifiers):
		self._label_classifiers = dict(label_classifiers)
		self._labels = sorted(self._label_classifiers.keys())
	
	def labels(self):
		return self._labels
	
	def classify(self, feats):
		lbls = set()
		
		for label, classifier in self._label_classifiers.items():
			if classifier.classify(feats) == label:
				lbls.add(label)
		
		return lbls

def train_binary_classifiers(trainf, labelled_feats, labelset):
	pos_feats = collections.defaultdict(list)
	neg_feats = collections.defaultdict(list)
	classifiers = {}
	
	for feat, labels in labelled_feats:
		for label in labels:
			pos_feats[label].append(feat)
		
		for label in labelset - set(labels):
			neg_feats[label].append(feat)
	
	for label in labelset:
		postrain = [(feat, label) for feat in pos_feats[label]]
		negtrain = [(feat, '!%s' % label) for feat in neg_feats[label]]
		classifiers[label] = trainf(postrain + negtrain)
	
	return classifiers

def multi_metrics(multi_classifier, test_feats):
	mds = []
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)
	
	for i, (feat, labels) in enumerate(test_feats):
		for label in labels:
			refsets[label].add(i)
		
		guessed = multi_classifier.classify(feat)
		
		for label in guessed:
			testsets[label].add(i)
		
		mds.append(metrics.masi_distance(set(labels), guessed))
	
	avg_md = sum(mds) / float(len(mds))
	precisions = {}
	recalls = {}
	
	for label in multi_classifier.labels():
		precisions[label] = metrics.precision(refsets[label], testsets[label])
		recalls[label] = metrics.recall(refsets[label], testsets[label])
	
	return precisions, recalls, avg_md