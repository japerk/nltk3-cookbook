import itertools
from nltk.tbl import Template
from nltk.tag import brill, brill_trainer
from nltk.probability import FreqDist, ConditionalFreqDist

def backoff_tagger(train_sents, tagger_classes, backoff=None):
	for cls in tagger_classes:
		backoff = cls(train_sents, backoff=backoff)
	
	return backoff

def word_tag_model(words, tagged_words, limit=200):
	fd = FreqDist(words)
	cfd = ConditionalFreqDist(tagged_words)
	most_freq = (word for word, count in fd.most_common(limit))
	return dict((word, cfd[word].max()) for word in most_freq)

patterns = [
	(r'^\d+$', 'CD'),
	(r'.*ing$', 'VBG'), # gerunds, i.e. wondering
	(r'.*ment$', 'NN'), # i.e. wonderment
	(r'.*ful$', 'JJ') # i.e. wonderful
]

def train_brill_tagger(initial_tagger, train_sents, **kwargs):
	templates = [
		brill.Template(brill.Pos([-1])),
		brill.Template(brill.Pos([1])),
		brill.Template(brill.Pos([-2])),
		brill.Template(brill.Pos([2])),
		brill.Template(brill.Pos([-2, -1])),
		brill.Template(brill.Pos([1, 2])),
		brill.Template(brill.Pos([-3, -2, -1])),
		brill.Template(brill.Pos([1, 2, 3])),
		brill.Template(brill.Pos([-1]), brill.Pos([1])),
		brill.Template(brill.Word([-1])),
		brill.Template(brill.Word([1])),
		brill.Template(brill.Word([-2])),
		brill.Template(brill.Word([2])),
		brill.Template(brill.Word([-2, -1])),
		brill.Template(brill.Word([1, 2])),
		brill.Template(brill.Word([-3, -2, -1])),
		brill.Template(brill.Word([1, 2, 3])),
		brill.Template(brill.Word([-1]), brill.Word([1])),
	]
	
	trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates, deterministic=True)
	return trainer.train(train_sents, **kwargs)

def unigram_feature_detector(tokens, index, history):
	return {'word': tokens[index]}