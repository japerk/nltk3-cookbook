import re, csv, yaml, enchant
from nltk.corpus import wordnet
from nltk.metrics import edit_distance

##################################################
## Replacing Words Matching Regular Expressions ##
##################################################

replacement_patterns = [
	(r'won\'t', 'will not'),
	(r'can\'t', 'cannot'),
	(r'i\'m', 'i am'),
	(r'ain\'t', 'is not'),
	(r'(\w+)\'ll', '\g<1> will'),
	(r'(\w+)n\'t', '\g<1> not'),
	(r'(\w+)\'ve', '\g<1> have'),
	(r'(\w+)\'s', '\g<1> is'),
	(r'(\w+)\'re', '\g<1> are'),
	(r'(\w+)\'d', '\g<1> would'),
]

class RegexpReplacer(object):
	""" Replaces regular expression in a text.
	>>> replacer = RegexpReplacer()
	>>> replacer.replace("can't is a contraction")
	'cannot is a contraction'
	>>> replacer.replace("I should've done that thing I didn't do")
	'I should have done that thing I did not do'
	"""
	def __init__(self, patterns=replacement_patterns):
		self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
	
	def replace(self, text):
		s = text
		
		for (pattern, repl) in self.patterns:
			s = re.sub(pattern, repl, s)
		
		return s

####################################
## Replacing Repeating Characters ##
####################################

class RepeatReplacer(object):
	""" Removes repeating characters until a valid word is found.
	>>> replacer = RepeatReplacer()
	>>> replacer.replace('looooove')
	'love'
	>>> replacer.replace('oooooh')
	'ooh'
	>>> replacer.replace('goose')
	'goose'
	"""
	def __init__(self):
		self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
		self.repl = r'\1\2\3'

	def replace(self, word):
		if wordnet.synsets(word):
			return word
		
		repl_word = self.repeat_regexp.sub(self.repl, word)
		
		if repl_word != word:
			return self.replace(repl_word)
		else:
			return repl_word

######################################
## Spelling Correction with Enchant ##
######################################

class SpellingReplacer(object):
	""" Replaces misspelled words with a likely suggestion based on shortest
	edit distance.
	>>> replacer = SpellingReplacer()
	>>> replacer.replace('cookbok')
	'cookbook'
	"""
	def __init__(self, dict_name='en', max_dist=2):
		self.spell_dict = enchant.Dict(dict_name)
		self.max_dist = max_dist
	
	def replace(self, word):
		if self.spell_dict.check(word):
			return word
		
		suggestions = self.spell_dict.suggest(word)
		
		if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
			return suggestions[0]
		else:
			return word

class CustomSpellingReplacer(SpellingReplacer):
	""" SpellingReplacer that allows passing a custom enchant dictionary, such
	a DictWithPWL.
	>>> d = enchant.DictWithPWL('en_US', 'mywords.txt')
	>>> replacer = CustomSpellingReplacer(d)
	>>> replacer.replace('nltk')
	'nltk'
	"""
	def __init__(self, spell_dict, max_dist=2):
		self.spell_dict = spell_dict
		self.max_dist = max_dist

########################
## Replacing Synonyms ##
########################

class WordReplacer(object):
	""" WordReplacer that replaces a given word with a word from the word_map,
	or if the word isn't found, returns the word as is.
	>>> replacer = WordReplacer({'bday': 'birthday'})
	>>> replacer.replace('bday')
	'birthday'
	>>> replacer.replace('happy')
	'happy'
	"""
	def __init__(self, word_map):
		self.word_map = word_map
	
	def replace(self, word):
		return self.word_map.get(word, word)

class CsvWordReplacer(WordReplacer):
	""" WordReplacer that reads word mappings from a csv file.
	>>> replacer = CsvWordReplacer('synonyms.csv')
	>>> replacer.replace('bday')
	'birthday'
	>>> replacer.replace('happy')
	'happy'
	"""
	def __init__(self, fname):
		word_map = {}
		
		for line in csv.reader(open(fname)):
			word, syn = line
			word_map[word] = syn
		
		super(CsvWordReplacer, self).__init__(word_map)

class YamlWordReplacer(WordReplacer):
	""" WordReplacer that reads word mappings from a yaml file.
	>>> replacer = YamlWordReplacer('synonyms.yaml')
	>>> replacer.replace('bday')
	'birthday'
	>>> replacer.replace('happy')
	'happy'
	"""
	def __init__(self, fname):
		word_map = yaml.load(open(fname))
		super(YamlWordReplacer, self).__init__(word_map)

#######################################
## Replacing Negations with Antonyms ##
#######################################

class AntonymReplacer(object):
	def replace(self, word, pos=None):
		""" Returns the antonym of a word, but only if there is no ambiguity.
		>>> replacer = AntonymReplacer()
		>>> replacer.replace('good')
		>>> replacer.replace('uglify')
		'beautify'
		>>> replacer.replace('beautify')
		'uglify'
		"""
		antonyms = set()
		
		for syn in wordnet.synsets(word, pos=pos):
			for lemma in syn.lemmas():
				for antonym in lemma.antonyms():
					antonyms.add(antonym.name())
		
		if len(antonyms) == 1:
			return antonyms.pop()
		else:
			return None
	
	def replace_negations(self, sent):
		""" Try to replace negations with antonyms in the tokenized sentence.
		>>> replacer = AntonymReplacer()
		>>> replacer.replace_negations(['do', 'not', 'uglify', 'our', 'code'])
		['do', 'beautify', 'our', 'code']
		>>> replacer.replace_negations(['good', 'is', 'not', 'evil'])
		['good', 'is', 'not', 'evil']
		"""
		i, l = 0, len(sent)
		words = []
		
		while i < l:
			word = sent[i]
			
			if word == 'not' and i+1 < l:
				ant = self.replace(sent[i+1])
				
				if ant:
					words.append(ant)
					i += 2
					continue
			
			words.append(word)
			i += 1
		
		return words

class AntonymWordReplacer(WordReplacer, AntonymReplacer):
	""" AntonymReplacer that uses a custom mapping instead of WordNet.
	Order of inheritance is very important, this class would not work if
	AntonymReplacer comes before WordReplacer.
	>>> replacer = AntonymWordReplacer({'evil': 'good'})
	>>> replacer.replace_negations(['good', 'is', 'not', 'evil'])
	['good', 'is', 'good']
	"""
	pass

if __name__ == '__main__':
	import doctest
	doctest.testmod()