import re, itertools
import nltk.tag
from nltk.tree import Tree

def filter_insignificant(chunk, tag_suffixes=['DT', 'CC']):
	'''Remove insignificant words from the chunk.
	>>> filter_insignificant([('the', 'DT'), ('terrible', 'JJ'), ('movie', 'NN')])
	[('terrible', 'JJ'), ('movie', 'NN')]
	'''
	good = []
	
	for word, tag in chunk:
		ok = True
		
		for suffix in tag_suffixes:
			if tag.endswith(suffix):
				ok = False
				break
		
		if ok:
			good.append((word, tag))
	
	return good

def tag_startswith(prefix):
	def f(wt):
		return wt[1].startswith(prefix)
	return f

def tag_equals(tag):
	def f(wt):
		return wt[1] == tag
	return f

def first_chunk_index(chunk, pred, start=0, step=1):
	'''Go through chunk and return the first index where pred(chunk[index])
	returns True.
	>>> first_chunk_index([('61', 'CD'), ('years', 'NNS')], tag_equals('CD'))
	0
	>>> first_chunk_index([('61', 'CD'), ('years', 'NNS')], tag_equals('NNS'))
	1
	>>> first_chunk_index([('61', 'CD'), ('years', 'NNS')], tag_equals('CD'), start=1, step=-1)
	0
	>>> first_chunk_index([('61', 'CD'), ('years', 'NNS')], tag_equals('VB'))
	'''
	l = len(chunk)
	end = l if step > 0 else -1
	
	for i in range(start, end, step):
		if pred(chunk[i]):
			return i
	
	return None

plural_verb_forms = {
	('is', 'VBZ'): ('are', 'VBP'),
	('was', 'VBD'): ('were', 'VBD')
}

singular_verb_forms = {
	('are', 'VBP'): ('is', 'VBZ'),
	('were', 'VBD'): ('was', 'VBD')
}

def correct_verbs(chunk):
	'''Correct plural/singular verb mistakes.
	>>> correct_verbs([('is', 'VBZ'), ('our', 'PRP$'), ('children', 'NNS'), ('learning', 'VBG')])
	[('are', 'VBP'), ('our', 'PRP$'), ('children', 'NNS'), ('learning', 'VBG')]
	>>> correct_verbs([('our', 'PRP$'), ('children', 'NNS'), ('is', 'VBZ'), ('learning', 'VBG')])
	[('our', 'PRP$'), ('children', 'NNS'), ('are', 'VBP'), ('learning', 'VBG')]
	>>> correct_verbs([('our', 'PRP$'), ('child', 'NN'), ('were', 'VBD'), ('learning', 'VBG')])
	[('our', 'PRP$'), ('child', 'NN'), ('was', 'VBD'), ('learning', 'VBG')]
	>>> correct_verbs([('our', 'PRP$'), ('child', 'NN'), ('is', 'VBZ'), ('learning', 'VBG')])
	[('our', 'PRP$'), ('child', 'NN'), ('is', 'VBZ'), ('learning', 'VBG')]
	'''
	vbidx = first_chunk_index(chunk, tag_startswith('VB'))
	# if no verb found, do nothing
	if vbidx is None:
		return chunk
	
	verb, vbtag = chunk[vbidx]
	nnpred = tag_startswith('NN')
	# find nearest noun to the right of verb
	nnidx = first_chunk_index(chunk, nnpred, start=vbidx+1)
	# if no noun found to right, look to the left
	if nnidx is None:
		nnidx = first_chunk_index(chunk, nnpred, start=vbidx-1, step=-1)
	# if no noun found, do nothing
	if nnidx is None:
		return chunk
	
	noun, nntag = chunk[nnidx]
	# get correct verb form and insert into chunk
	if nntag.endswith('S'):
		chunk[vbidx] = plural_verb_forms.get((verb, vbtag), (verb, vbtag))
	else:
		chunk[vbidx] = singular_verb_forms.get((verb, vbtag), (verb, vbtag))
	
	return chunk

def swap_verb_phrase(chunk):
	'''Move modifier phrase after verb to front of chunk and drop the verb.
	>>> swap_verb_phrase([('the', 'DT'), ('book', 'NN'), ('was', 'VBD'), ('great', 'JJ')])
	[('great', 'JJ'), ('the', 'DT'), ('book', 'NN')]
	>>> swap_verb_phrase([('this', 'DT'), ('gripping', 'VBG'), ('book', 'NN'), ('is', 'VBZ'), ('fantastic', 'JJ')])
	[('fantastic', 'JJ'), ('this', 'DT'), ('gripping', 'VBG'), ('book', 'NN')]
	'''
	# find location of verb
	def vbpred(wt):
		word, tag = wt
		return tag != 'VBG' and tag.startswith('VB') and len(tag) > 2
	
	vbidx = first_chunk_index(chunk, vbpred)
	
	if vbidx is None:
		return chunk
	
	return chunk[vbidx+1:] + chunk[:vbidx]

def swap_noun_cardinal(chunk):
	'''Move a cardinal that occurs after a noun to immediately before the noun.
	>>> swap_noun_cardinal([('Dec.', 'NNP'), ('10', 'CD')])
	[('10', 'CD'), ('Dec.', 'NNP')]
	>>> swap_noun_cardinal([('the', 'DT'), ('top', 'NN'), ('10', 'CD')])
	[('the', 'DT'), ('10', 'CD'), ('top', 'NN')]
	'''
	cdidx = first_chunk_index(chunk, tag_equals('CD'))
	# cdidx must be > 0 and there must be a noun immediately before it
	if not cdidx or not chunk[cdidx-1][1].startswith('NN'):
		return chunk
	
	noun, nntag = chunk[cdidx-1]
	chunk[cdidx-1] = chunk[cdidx]
	chunk[cdidx] = noun, nntag
	return chunk

def swap_infinitive_phrase(chunk):
	'''Move subject to before the noun preceding the infinitive.
	>>> swap_infinitive_phrase([('book', 'NN'), ('of', 'IN'), ('recipes', 'NNS')])
	[('recipes', 'NNS'), ('book', 'NN')]
	>>> swap_infinitive_phrase([('tastes', 'VBZ'), ('like', 'IN'), ('chicken', 'NN')])
	[('tastes', 'VBZ'), ('like', 'IN'), ('chicken', 'NN')]
	>>> swap_infinitive_phrase([('delicious', 'JJ'), ('book', 'NN'), ('of', 'IN'), ('recipes', 'NNS')])
	[('delicious', 'JJ'), ('recipes', 'NNS'), ('book', 'NN')]
	'''
	def inpred(wt):
		word, tag = wt
		return tag == 'IN' and word != 'like'
	
	inidx = first_chunk_index(chunk, inpred)
	
	if inidx is None:
		return chunk
	
	nnidx = first_chunk_index(chunk, tag_startswith('NN'), start=inidx, step=-1) or 0
	return chunk[:nnidx] + chunk[inidx+1:] + chunk[nnidx:inidx]

def singularize_plural_noun(chunk):
	'''If a plural noun is followed by another noun, singularize the plural noun.
	>>> singularize_plural_noun([('recipes', 'NNS'), ('book', 'NN')])
	[('recipe', 'NN'), ('book', 'NN')]
	'''
	nnsidx = first_chunk_index(chunk, tag_equals('NNS'))
	
	if nnsidx is not None and nnsidx+1 < len(chunk) and chunk[nnsidx+1][1][:2] == 'NN':
		noun, nnstag = chunk[nnsidx]
		chunk[nnsidx] = (noun.rstrip('s'), nnstag.rstrip('S'))
	
	return chunk

def transform_chunk(chunk, chain=[filter_insignificant, swap_verb_phrase, swap_infinitive_phrase, singularize_plural_noun], trace=0):
	'''
	>>> transform_chunk([('the', 'DT'), ('book', 'NN'), ('of', 'IN'), ('recipes', 'NNS'), ('is', 'VBZ'), ('delicious', 'JJ')])
	[('delicious', 'JJ'), ('recipe', 'NN'), ('book', 'NN')]
	'''
	for f in chain:
		chunk = f(chunk)
		
		if trace:
			print('%s : %s' % (f.__name__, chunk))
	
	return chunk

punct_re = re.compile(r'\s([,\.;\?])')

def chunk_tree_to_sent(tree, concat=' '):
	'''Convert a parse tree to a sentence, with correct punctuation.
	>>> from nltk.tree import Tree
	>>> chunk_tree_to_sent(Tree('S', [Tree('NP', [('Pierre', 'NNP'), ('Vinken', 'NNP')]), (',', ','), Tree('NP', [('61', 'CD'), ('years', 'NNS')]), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), Tree('NP', [('the', 'DT'), ('board', 'NN')]), ('as', 'IN'), Tree('NP', [('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD')]), ('.', '.')]))
	'Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.'
	'''
	s = concat.join(nltk.tag.untag(tree.leaves()))
	return re.sub(punct_re, r'\g<1>', s)

def flatten_childtrees(trees):
	children = []
	
	for t in trees:
		if t.height() < 3:
			children.extend(t.pos())
		elif t.height() == 3:
			children.append(Tree(t.label(), t.pos()))
		else:
			children.extend(flatten_childtrees([c for c in t]))
	
	return children

def flatten_deeptree(tree):
	'''
	>>> flatten_deeptree(Tree('S', [Tree('NP-SBJ', [Tree('NP', [Tree('NNP', ['Pierre']), Tree('NNP', ['Vinken'])]), Tree(',', [',']), Tree('ADJP', [Tree('NP', [Tree('CD', ['61']), Tree('NNS', ['years'])]), Tree('JJ', ['old'])]), Tree(',', [','])]), Tree('VP', [Tree('MD', ['will']), Tree('VP', [Tree('VB', ['join']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['board'])]), Tree('PP-CLR', [Tree('IN', ['as']), Tree('NP', [Tree('DT', ['a']), Tree('JJ', ['nonexecutive']), Tree('NN', ['director'])])]), Tree('NP-TMP', [Tree('NNP', ['Nov.']), Tree('CD', ['29'])])])]), Tree('.', ['.'])]))
	Tree('S', [Tree('NP', [('Pierre', 'NNP'), ('Vinken', 'NNP')]), (',', ','), Tree('NP', [('61', 'CD'), ('years', 'NNS')]), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), Tree('NP', [('the', 'DT'), ('board', 'NN')]), ('as', 'IN'), Tree('NP', [('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN')]), Tree('NP-TMP', [('Nov.', 'NNP'), ('29', 'CD')]), ('.', '.')])
	'''
	return Tree(tree.label(), flatten_childtrees([c for c in tree]))

def shallow_tree(tree):
	'''
	>>> shallow_tree(Tree('S', [Tree('NP-SBJ', [Tree('NP', [Tree('NNP', ['Pierre']), Tree('NNP', ['Vinken'])]), Tree(',', [',']), Tree('ADJP', [Tree('NP', [Tree('CD', ['61']), Tree('NNS', ['years'])]), Tree('JJ', ['old'])]), Tree(',', [','])]), Tree('VP', [Tree('MD', ['will']), Tree('VP', [Tree('VB', ['join']), Tree('NP', [Tree('DT', ['the']), Tree('NN', ['board'])]), Tree('PP-CLR', [Tree('IN', ['as']), Tree('NP', [Tree('DT', ['a']), Tree('JJ', ['nonexecutive']), Tree('NN', ['director'])])]), Tree('NP-TMP', [Tree('NNP', ['Nov.']), Tree('CD', ['29'])])])]), Tree('.', ['.'])]))
	Tree('S', [Tree('NP-SBJ', [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ',')]), Tree('VP', [('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD')]), ('.', '.')])
	'''
	children = []
	
	for t in tree:
		if t.height() < 3:
			children.extend(t.pos())
		else:
			children.append(Tree(t.label(), t.pos()))
	
	return Tree(tree.label(), children)

def convert_tree_labels(tree, mapping):
	'''
	>>> convert_tree_labels(Tree('S', [Tree('NP-SBJ', [('foo', 'NN')])]), {'NP-SBJ': 'NP'})
	Tree('S', [Tree('NP', [('foo', 'NN')])])
	'''
	children = []
	
	for t in tree:
		if isinstance(t, Tree):
			children.append(convert_tree_labels(t, mapping))
		else:
			children.append(t)
	
	label = mapping.get(tree.label(), tree.label())
	return Tree(label, children)

if __name__ == '__main__':
	import doctest
	doctest.testmod()