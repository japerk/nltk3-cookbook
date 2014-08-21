"""
==============================================
Chunking and Chinking with Regular Expressions
==============================================

>>> from nltk.chunk.regexp import tag_pattern2re_pattern
>>> tag_pattern2re_pattern('<DT>?<NN.*>+')
'(<(DT)>)?(<(NN[^\\\{\\\}<>]*)>)+'

>>> from nltk.chunk import RegexpParser
>>> chunker = RegexpParser(r'''
... NP:
...		{<DT><NN.*><.*>*<NN.*>}
...		}<VB.*>{
... ''')
>>> chunker.parse([('the', 'DT'), ('book', 'NN'), ('has', 'VBZ'), ('many', 'JJ'), ('chapters', 'NNS')])
Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')]), ('has', 'VBZ'), Tree('NP', [('many', 'JJ'), ('chapters', 'NNS')])])

>>> from nltk.chunk.regexp import ChunkString, ChunkRule, ChinkRule
>>> from nltk.tree import Tree
>>> t = Tree('S', [('the', 'DT'), ('book', 'NN'), ('has', 'VBZ'), ('many', 'JJ'), ('chapters', 'NNS')])
>>> cs = ChunkString(t)
>>> cs
<ChunkString: '<DT><NN><VBZ><JJ><NNS>'>
>>> ur = ChunkRule('<DT><NN.*><.*>*<NN.*>', 'chunk determiners and nouns')
>>> ur.apply(cs)
>>> cs
<ChunkString: '{<DT><NN><VBZ><JJ><NNS>}'>
>>> ir = ChinkRule('<VB.*>', 'chink verbs')
>>> ir.apply(cs)
>>> cs
<ChunkString: '{<DT><NN>}<VBZ>{<JJ><NNS>}'>
>>> cs.to_chunkstruct()
Tree('S', [Tree('CHUNK', [('the', 'DT'), ('book', 'NN')]), ('has', 'VBZ'), Tree('CHUNK', [('many', 'JJ'), ('chapters', 'NNS')])])

>>> from nltk.chunk import RegexpChunkParser
>>> chunker = RegexpChunkParser([ur, ir])
>>> chunker.parse(t)
Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')]), ('has', 'VBZ'), Tree('NP', [('many', 'JJ'), ('chapters', 'NNS')])])

>>> from nltk.chunk import RegexpChunkParser
>>> chunker = RegexpChunkParser([ur, ir], chunk_label='CP')
>>> chunker.parse(t)
Tree('S', [Tree('CP', [('the', 'DT'), ('book', 'NN')]), ('has', 'VBZ'), Tree('CP', [('many', 'JJ'), ('chapters', 'NNS')])])

>>> chunker = RegexpParser(r'''
... NP:
...		{<DT><NN.*>}
...		{<JJ><NN.*>}
... ''')
>>> chunker.parse(t)
Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')]), ('has', 'VBZ'), Tree('NP', [('many', 'JJ'), ('chapters', 'NNS')])])

>>> chunker = RegexpParser(r'''
... NP:
...		{(<DT>|<JJ>)<NN.*>}
... ''')
>>> chunker.parse(t)
Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')]), ('has', 'VBZ'), Tree('NP', [('many', 'JJ'), ('chapters', 'NNS')])])

>>> from nltk.chunk.regexp import ChunkRuleWithContext
>>> ctx = ChunkRuleWithContext('<DT>', '<NN.*>', '<.*>', 'chunk nouns only after determiners')
>>> cs = ChunkString(t)
>>> cs
<ChunkString: '<DT><NN><VBZ><JJ><NNS>'>
>>> ctx.apply(cs)
>>> cs
<ChunkString: '<DT>{<NN>}<VBZ><JJ><NNS>'>
>>> cs.to_chunkstruct()
Tree('S', [('the', 'DT'), Tree('CHUNK', [('book', 'NN')]), ('has', 'VBZ'), ('many', 'JJ'), ('chapters', 'NNS')])

>>> chunker = RegexpParser(r'''
... NP:
...		<DT>{<NN.*>}
... ''')
>>> chunker.parse(t)
Tree('S', [('the', 'DT'), Tree('NP', [('book', 'NN')]), ('has', 'VBZ'), ('many', 'JJ'), ('chapters', 'NNS')])


=====================================================
Merging and Splitting Chunks with Regular Expressions
=====================================================

>>> chunker = RegexpParser(r'''
... NP:
...     {<DT><.*>*<NN.*>}
...     <NN.*>}{<.*>
...     <.*>}{<DT>
...     <NN.*>{}<NN.*>
... ''')
>>> sent = [('the', 'DT'), ('sushi', 'NN'), ('roll', 'NN'), ('was', 'VBD'), ('filled', 'VBN'), ('with', 'IN'), ('the', 'DT'), ('fish', 'NN')]
>>> chunker.parse(sent)
Tree('S', [Tree('NP', [('the', 'DT'), ('sushi', 'NN'), ('roll', 'NN')]), Tree('NP', [('was', 'VBD'), ('filled', 'VBN'), ('with', 'IN')]), Tree('NP', [('the', 'DT'), ('fish', 'NN')])])

>>> from nltk.chunk.regexp import MergeRule, SplitRule
>>> cs = ChunkString(Tree('S', sent))
>>> cs
<ChunkString: '<DT><NN><NN><VBD><VBN><IN><DT><NN>'>
>>> ur = ChunkRule('<DT><.*>*<NN.*>', 'chunk determiner to noun')
>>> ur.apply(cs)
>>> cs
<ChunkString: '{<DT><NN><NN><VBD><VBN><IN><DT><NN>}'>
>>> sr1 = SplitRule('<NN.*>', '<.*>', 'split after noun')
>>> sr1.apply(cs)
>>> cs
<ChunkString: '{<DT><NN>}{<NN>}{<VBD><VBN><IN><DT><NN>}'>
>>> sr2 = SplitRule('<.*>', '<DT>', 'split before determiner')
>>> sr2.apply(cs)
>>> cs
<ChunkString: '{<DT><NN>}{<NN>}{<VBD><VBN><IN>}{<DT><NN>}'>
>>> mr = MergeRule('<NN.*>', '<NN.*>', 'merge nouns')
>>> mr.apply(cs)
>>> cs
<ChunkString: '{<DT><NN><NN>}{<VBD><VBN><IN>}{<DT><NN>}'>
>>> cs.to_chunkstruct()
Tree('S', [Tree('CHUNK', [('the', 'DT'), ('sushi', 'NN'), ('roll', 'NN')]), Tree('CHUNK', [('was', 'VBD'), ('filled', 'VBN'), ('with', 'IN')]), Tree('CHUNK', [('the', 'DT'), ('fish', 'NN')])])

>>> from nltk.chunk.regexp import RegexpChunkRule
>>> RegexpChunkRule.fromstring('{<DT><.*>*<NN.*>}')
<ChunkRule: '<DT><.*>*<NN.*>'>
>>> RegexpChunkRule.fromstring('<.*>}{<DT>')
<SplitRule: '<.*>', '<DT>'>
>>> RegexpChunkRule.fromstring('<NN.*>{}<NN.*>')
<MergeRule: '<NN.*>', '<NN.*>'>

>>> RegexpChunkRule.fromstring('{<DT><.*>*<NN.*>} # chunk everything').descr()
'chunk everything'
>>> RegexpChunkRule.fromstring('{<DT><.*>*<NN.*>}').descr()
''


======================================================
Expanding and Removing Chunks with Regular Expressions
======================================================

>>> from nltk.chunk.regexp import ChunkRule, ExpandLeftRule, ExpandRightRule, UnChunkRule
>>> from nltk.chunk import RegexpChunkParser
>>> ur = ChunkRule('<NN>', 'single noun')
>>> el = ExpandLeftRule('<DT>', '<NN>', 'get left determiner')
>>> er = ExpandRightRule('<NN>', '<NNS>', 'get right plural noun')
>>> un = UnChunkRule('<DT><NN.*>*', 'unchunk everything')
>>> chunker = RegexpChunkParser([ur, el, er, un])
>>> sent = [('the', 'DT'), ('sushi', 'NN'), ('rolls', 'NNS')]
>>> chunker.parse(sent)
Tree('S', [('the', 'DT'), ('sushi', 'NN'), ('rolls', 'NNS')])

>>> from nltk.chunk.regexp import ChunkString
>>> from nltk.tree import Tree
>>> cs = ChunkString(Tree('S', sent))
>>> cs
<ChunkString: '<DT><NN><NNS>'>
>>> ur.apply(cs)
>>> cs
<ChunkString: '<DT>{<NN>}<NNS>'>
>>> el.apply(cs)
>>> cs
<ChunkString: '{<DT><NN>}<NNS>'>
>>> er.apply(cs)
>>> cs
<ChunkString: '{<DT><NN><NNS>}'>
>>> un.apply(cs)
>>> cs
<ChunkString: '<DT><NN><NNS>'>


========================================
Partial Parsing with Regular Expressions
========================================

>>> chunker = RegexpParser(r'''
... NP:
... 	{<DT>?<NN.*>+}	# chunk optional determiner with nouns
... 	<JJ>{}<NN.*>	# merge adjective with noun chunk
... PP:
... 	{<IN>}			# chunk preposition
... VP:
... 	{<MD>?<VB.*>}	# chunk optional modal with verb
... ''')
>>> from nltk.corpus import conll2000
>>> score = chunker.evaluate(conll2000.chunked_sents())
>>> score.accuracy()
0.6148573545757688

>>> from nltk.corpus import treebank_chunk
>>> treebank_score = chunker.evaluate(treebank_chunk.chunked_sents())
>>> treebank_score.accuracy()
0.49033970276008493

>>> score.precision()
0.60201948127375
>>> score.recall()
0.606072502505847

>>> len(score.missed())
47161
>>> len(score.incorrect())
47967
>>> len(score.correct())
119720
>>> len(score.guessed())
120526


===============================
Training a Tagger Based Chunker
===============================

>>> from nltk.chunk.util import tree2conlltags, conlltags2tree
>>> from nltk.tree import Tree
>>> t = Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')])])
>>> tree2conlltags(t)
[('the', 'DT', 'B-NP'), ('book', 'NN', 'I-NP')]
>>> conlltags2tree([('the', 'DT', 'B-NP'), ('book', 'NN', 'I-NP')])
Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')])])

>>> from chunkers import TagChunker
>>> from nltk.corpus import treebank_chunk
>>> train_chunks = treebank_chunk.chunked_sents()[:3000]
>>> test_chunks = treebank_chunk.chunked_sents()[3000:]
>>> chunker = TagChunker(train_chunks)
>>> score = chunker.evaluate(test_chunks)
>>> score.accuracy()
0.9732039335251428
>>> score.precision()
0.9166534370535006
>>> score.recall()
0.9465573770491803

>>> from nltk.corpus import conll2000
>>> conll_train = conll2000.chunked_sents('train.txt')
>>> conll_test = conll2000.chunked_sents('test.txt')
>>> chunker = TagChunker(conll_train)
>>> score = chunker.evaluate(conll_test)
>>> score.accuracy()
0.8950545623403762
>>> score.precision()
0.8114841974355675
>>> score.recall()
0.8644191676944863

>>> from nltk.tag import UnigramTagger
>>> uni_chunker = TagChunker(train_chunks, tagger_classes=[UnigramTagger])
>>> score = uni_chunker.evaluate(test_chunks)
>>> score.accuracy()
0.9674925924335466


=============================
Classification Based Chunking
=============================

>>> from chunkers import ClassifierChunker
>>> chunker = ClassifierChunker(train_chunks)
>>> score = chunker.evaluate(test_chunks)
>>> score.accuracy()
0.9721733155838022
>>> score.precision()
0.9258838793383068
>>> score.recall()
0.9359016393442623

>>> chunker = ClassifierChunker(conll_train)
>>> score = chunker.evaluate(conll_test)
>>> score.accuracy()
0.9264622074002153
>>> score.precision()
0.8737924310910219
>>> score.recall()
0.9007354620620346

>>> from nltk.classify import MaxentClassifier
>>> builder = lambda toks: MaxentClassifier.train(toks, trace=0, max_iter=10, min_lldelta=0.01)
>>> me_chunker = ClassifierChunker(train_chunks, classifier_builder=builder)
>>> score = me_chunker.evaluate(test_chunks)
>>> score.accuracy()
0.9743204362949285
>>> score.precision()
0.9334423548650859
>>> score.recall()
0.9357377049180328


=========================
Extracting Named Entities
=========================

>>> from nltk.chunk import ne_chunk
>>> ne_chunk(treebank_chunk.tagged_sents()[0])
Tree('S', [Tree('PERSON', [('Pierre', 'NNP')]), Tree('ORGANIZATION', [('Vinken', 'NNP')]), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')])

>>> tree = ne_chunk(treebank_chunk.tagged_sents()[0])
>>> from chunkers import sub_leaves
>>> sub_leaves(tree, 'PERSON')
[[('Pierre', 'NNP')]]
>>> sub_leaves(tree, 'ORGANIZATION')
[[('Vinken', 'NNP')]]

>>> from nltk.chunk import ne_chunk_sents
>>> trees = ne_chunk_sents(treebank_chunk.tagged_sents()[:10])
>>> [sub_leaves(t, 'ORGANIZATION') for t in trees]
[[[('Vinken', 'NNP')]], [[('Elsevier', 'NNP')]], [[('Consolidated', 'NNP'), ('Gold', 'NNP'), ('Fields', 'NNP')]], [], [], [[('Inc.', 'NNP')], [('Micronite', 'NN')]], [[('New', 'NNP'), ('England', 'NNP'), ('Journal', 'NNP')]], [[('Lorillard', 'NNP')]], [], []]

>>> ne_chunk(treebank_chunk.tagged_sents()[0], binary=True)
Tree('S', [Tree('NE', [('Pierre', 'NNP'), ('Vinken', 'NNP')]), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')])

>>> sub_leaves(ne_chunk(treebank_chunk.tagged_sents()[0], binary=True), 'NE')
[[('Pierre', 'NNP'), ('Vinken', 'NNP')]]


===============================
Extracting Proper Noun Entities
===============================

>>> chunker = RegexpParser(r'''
... NAME:
... 	{<NNP>+}
... ''')
>>> sub_leaves(chunker.parse(treebank_chunk.tagged_sents()[0]), 'NAME')
[[('Pierre', 'NNP'), ('Vinken', 'NNP')], [('Nov.', 'NNP')]]


===============================
Training a Named Entity Chunker
===============================

>>> from chunkers import ieer_chunked_sents
>>> ieer_chunks = list(ieer_chunked_sents())
>>> len(ieer_chunks)
94
>>> chunker = ClassifierChunker(ieer_chunks[:80])
>>> chunker.parse(treebank_chunk.tagged_sents()[0])
Tree('S', [Tree('LOCATION', [('Pierre', 'NNP'), ('Vinken', 'NNP')]), (',', ','), Tree('DURATION', [('61', 'CD'), ('years', 'NNS')]), Tree('MEASURE', [('old', 'JJ')]), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), Tree('DATE', [('Nov.', 'NNP'), ('29', 'CD')]), ('.', '.')])

>>> score = chunker.evaluate(ieer_chunks[80:])
>>> score.accuracy()
0.8829018388070625
>>> score.precision()
0.4088717454194793
>>> score.recall()
0.5053635280095352

>>> from nltk.corpus import ieer
>>> ieer.parsed_docs()[0].headline
Tree('DOCUMENT', ['Kenyans', 'protest', 'tax', 'hikes'])

"""

if __name__ == '__main__':
	import doctest
	doctest.testmod()