import nltk.tag
from nltk.chunk import ChunkParserI
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tag import UnigramTagger, BigramTagger, ClassifierBasedTagger
from nltk.corpus import names, ieer, gazetteers
from tag_util import backoff_tagger

def conll_tag_chunks(chunk_sents):
	'''Convert each chunked sentence to list of (tag, chunk_tag) tuples,
	so the final result is a list of lists of (tag, chunk_tag) tuples.
	>>> from nltk.tree import Tree
	>>> t = Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')])])
	>>> conll_tag_chunks([t])
	[[('DT', 'B-NP'), ('NN', 'I-NP')]]
	'''
	tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
	return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

class TagChunker(ChunkParserI):
	'''Chunks tagged tokens using Ngram Tagging.'''
	def __init__(self, train_chunks, tagger_classes=[UnigramTagger, BigramTagger]):
		'''Train Ngram taggers on chunked sentences'''
		train_sents = conll_tag_chunks(train_chunks)
		self.tagger = backoff_tagger(train_sents, tagger_classes)
	
	def parse(self, tagged_sent):
		'''Parsed tagged tokens into parse Tree of chunks'''
		if not tagged_sent: return None
		(words, tags) = zip(*tagged_sent)
		chunks = self.tagger.tag(tags)
		# create conll str for tree parsing
		wtc = zip(words, chunks)
		return conlltags2tree([(w,t,c) for (w,(t,c)) in wtc])

def chunk_trees2train_chunks(chunk_sents):
	tag_sents = [tree2conlltags(sent) for sent in chunk_sents]
	return [[((w,t),c) for (w,t,c) in sent] for sent in tag_sents]

def prev_next_pos_iob(tokens, index, history):
	word, pos = tokens[index]
	
	if index == 0:
		prevword, prevpos, previob = ('<START>',)*3
	else:
		prevword, prevpos = tokens[index-1]
		previob = history[index-1]
	
	if index == len(tokens) - 1:
		nextword, nextpos = ('<END>',)*2
	else:
		nextword, nextpos = tokens[index+1]
	
	feats = {
		'word': word,
		'pos': pos,
		'nextword': nextword,
		'nextpos': nextpos,
		'prevword': prevword,
		'prevpos': prevpos,
		'previob': previob
	}
	
	return feats

class ClassifierChunker(ChunkParserI):
	def __init__(self, train_sents, feature_detector=prev_next_pos_iob, **kwargs):
		if not feature_detector:
			feature_detector = self.feature_detector
		
		train_chunks = chunk_trees2train_chunks(train_sents)
		self.tagger = ClassifierBasedTagger(train=train_chunks,
			feature_detector=feature_detector, **kwargs)
	
	def parse(self, tagged_sent):
		if not tagged_sent: return None
		chunks = self.tagger.tag(tagged_sent)
		return conlltags2tree([(w,t,c) for ((w,t),c) in chunks])

def sub_leaves(tree, label):
	return [t.leaves() for t in tree.subtrees(lambda s: s.label() == label)]

class PersonChunker(ChunkParserI):
	'''
	>>> from nltk.corpus import treebank_chunk
	>>> chunker = PersonChunker()
	>>> sub_leaves(chunker.parse(treebank_chunk.tagged_sents()[0]), 'PERSON')
	[[('Pierre', 'NNP')]]
	'''
	def __init__(self):
		self.name_set = set(names.words())
	
	def parse(self, tagged_sent):
		iobs = []
		in_person = False
		
		for word, tag in tagged_sent:
			if word in self.name_set and in_person:
				iobs.append((word, tag, 'I-PERSON'))
			elif word in self.name_set:
				iobs.append((word, tag, 'B-PERSON'))
				in_person = True
			else:
				iobs.append((word, tag, 'O'))
				in_person = False
		
		return conlltags2tree(iobs)

class LocationChunker(ChunkParserI):
	'''Chunks locations based on the gazetteers corpus.
	>>> loc = LocationChunker()
	>>> t = loc.parse([('San', 'NNP'), ('Francisco', 'NNP'), ('CA', 'NNP'), ('is', 'BE'), ('cold', 'JJ'), ('compared', 'VBD'), ('to', 'TO'), ('San', 'NNP'), ('Jose', 'NNP'), ('CA', 'NNP')])
	>>> sub_leaves(t, 'LOCATION')
	[[('San', 'NNP'), ('Francisco', 'NNP'), ('CA', 'NNP')], [('San', 'NNP'), ('Jose', 'NNP'), ('CA', 'NNP')]]
	'''
	def __init__(self):
		# gazetteers is a WordListCorpusReader of many different location words
		self.locations = set(gazetteers.words())
		self.lookahead = 0
		# need to know how many words to lookahead in the tagged sentence to find a location
		for loc in self.locations:
			nwords = loc.count(' ')
			
			if nwords > self.lookahead:
				self.lookahead = nwords
	
	def iob_locations(self, tagged_sent):
		i = 0
		l = len(tagged_sent)
		inside = False
		
		while i < l:
			word, tag = tagged_sent[i]
			j = i + 1
			k = j + self.lookahead
			nextwords, nexttags = [], []
			loc = False
			# lookahead in the sentence to find multi-word locations
			while j < k:
				if ' '.join([word] + nextwords) in self.locations:
					# combine multiple separate locations into single location chunk
					if inside:
						yield word, tag, 'I-LOCATION'
					else:
						yield word, tag, 'B-LOCATION'
					# every next word is inside the location chunk
					for nword, ntag in zip(nextwords, nexttags):
						yield nword, ntag, 'I-LOCATION'
					# found a location, so we're inside a chunk
					loc, inside = True, True
					# move forward to the next word since the current words
					# are already chunked
					i = j
					break
				
				if j < l:
					nextword, nexttag = tagged_sent[j]
					nextwords.append(nextword)
					nexttags.append(nexttag)
					j += 1
				else:
					break
			# if no location found, then we're outside the location chunk
			if not loc:
				inside = False
				i += 1
				yield word, tag, 'O'
	
	def parse(self, tagged_sent):
		iobs = self.iob_locations(tagged_sent)
		return conlltags2tree(iobs)

def ieertree2conlltags(tree, tag=nltk.tag.pos_tag):
	# tree.pos() flattens the tree and produces [(word, node)] where node is
	# from the word's parent tree node. words in a chunk therefore get the
	# chunk tag, while words outside a chunk get the same tag as the tree's
	# top node
	words, ents = zip(*tree.pos())
	iobs = []
	prev = None
	# construct iob tags from entity names
	for ent in ents:
		# any entity that is the same as the tree's top node is outside a chunk
		if ent == tree.label():
			iobs.append('O')
			prev = None
		# have a previous entity that is equal so this is inside the chunk
		elif prev == ent:
			iobs.append('I-%s' % ent)
		# no previous equal entity in the sequence, so this is the beginning of
		# an entity chunk
		else:
			iobs.append('B-%s' % ent)
			prev = ent
	# get tags for each word, then construct 3-tuple for conll tags
	words, tags = zip(*tag(words))
	return zip(words, tags, iobs)

def ieer_chunked_sents(tag=nltk.tag.pos_tag):
	for doc in ieer.parsed_docs():
		tagged = ieertree2conlltags(doc.text, tag)
		yield conlltags2tree(tagged)

if __name__ == '__main__':
	import doctest
	doctest.testmod()