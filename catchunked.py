from nltk.corpus.reader import CategorizedCorpusReader, ChunkedCorpusReader
from nltk.corpus.reader import ConllCorpusReader, ConllChunkCorpusReader

class CategorizedChunkedCorpusReader(CategorizedCorpusReader, ChunkedCorpusReader):
	"""
	A reader for chunked corpora whose documents are divided into categories
	based on their file identifiers.
	"""
	# code adapted from CategorizedTaggedCorpusReader
	def __init__(self, *args, **kwargs):
		CategorizedCorpusReader.__init__(self, kwargs)
		ChunkedCorpusReader.__init__(self, *args, **kwargs)
	
	def _resolve(self, fileids, categories):
		if fileids is not None and categories is not None:
			raise ValueError('Specify fileids or categories, not both')
		if categories is not None:
			return self.fileids(categories)
		else:
			return fileids
	
	def raw(self, fileids=None, categories=None):
		return ChunkedCorpusReader.raw(self, self._resolve(fileids, categories))
	
	def words(self, fileids=None, categories=None):
		return ChunkedCorpusReader.words(self, self._resolve(fileids, categories))
	
	def sents(self, fileids=None, categories=None):
		return ChunkedCorpusReader.sents(self, self._resolve(fileids, categories))
	
	def paras(self, fileids=None, categories=None):
		return ChunkedCorpusReader.paras(self, self._resolve(fileids, categories))
	
	def tagged_words(self, fileids=None, categories=None):
		return ChunkedCorpusReader.tagged_words(self, self._resolve(fileids, categories))
	
	def tagged_sents(self, fileids=None, categories=None):
		return ChunkedCorpusReader.tagged_sents(self, self._resolve(fileids, categories))
		
	def tagged_paras(self, fileids=None, categories=None):
		return ChunkedCorpusReader.tagged_paras(self, self._resolve(fileids, categories))
	
	def chunked_words(self, fileids=None, categories=None):
		return ChunkedCorpusReader.chunked_words(
			self, self._resolve(fileids, categories))
	
	def chunked_sents(self, fileids=None, categories=None):
		return ChunkedCorpusReader.chunked_sents(
			self, self._resolve(fileids, categories))
	
	def chunked_paras(self, fileids=None, categories=None):
		return ChunkedCorpusReader.chunked_paras(
			self, self._resolve(fileids, categories))

class CategorizedConllChunkCorpusReader(CategorizedCorpusReader, ConllChunkCorpusReader):
	"""
	A reader for conll chunked corpora whose documents are divided into
	categories based on their file identifiers.
	"""
	def __init__(self, *args, **kwargs):
		# NOTE: in addition to cat_pattern, ConllChunkCorpusReader also requires
		# chunk_types as third argument, which defaults to ('NP','VP','PP')
		CategorizedCorpusReader.__init__(self, kwargs)
		ConllChunkCorpusReader.__init__(self, *args, **kwargs)
	
	def _resolve(self, fileids, categories):
		if fileids is not None and categories is not None:
			raise ValueError('Specify fileids or categories, not both')
		if categories is not None:
			return self.fileids(categories)
		else:
			return fileids
	
	def raw(self, fileids=None, categories=None):
		return ConllCorpusReader.raw(self, self._resolve(fileids, categories))
	
	def words(self, fileids=None, categories=None):
		return ConllCorpusReader.words(self, self._resolve(fileids, categories))
	
	def sents(self, fileids=None, categories=None):
		return ConllCorpusReader.sents(self, self._resolve(fileids, categories))
	
	def tagged_words(self, fileids=None, categories=None):
		return ConllCorpusReader.tagged_words(self, self._resolve(fileids, categories))
	
	def tagged_sents(self, fileids=None, categories=None):
		return ConllCorpusReader.tagged_sents(self, self._resolve(fileids, categories))
	
	def chunked_words(self, fileids=None, categories=None, chunk_types=None):
		return ConllCorpusReader.chunked_words(
			self, self._resolve(fileids, categories), chunk_types)
	
	def chunked_sents(self, fileids=None, categories=None, chunk_types=None):
		return ConllCorpusReader.chunked_sents(
			self, self._resolve(fileids, categories), chunk_types)
	
	def parsed_sents(self, fileids=None, categories=None, pos_in_tree=None):
		return ConllCorpusReader.parsed_sents(
			self, self._resolve(fileids, categories), pos_in_tree)
	
	def srl_spans(self, fileids=None, categories=None):
		return ConllCorpusReader.srl_spans(self, self._resolve(fileids, categories))
	
	def srl_instances(self, fileids=None, categories=None, pos_in_tree=None, flatten=True):
		return ConllCorpusReader.srl_instances(
			self, self._resolve(fileids, categories), pos_in_tree, flatten)
	
	def iob_words(self, fileids=None, categories=None):
		return ConllCorpusReader.iob_words(self, self._resolve(fileids, categories))
	
	def iob_sents(self, fileids=None, categories=None):
		return ConllCorpusReader.iob_sents(self, self._resolve(fileids, categories))