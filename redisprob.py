from nltk.probability import ConditionalFreqDist
from rediscollections import RedisHashMap, encode_key

class RedisHashFreqDist(RedisHashMap):
	'''
	>>> from redis import Redis
	>>> r = Redis()
	>>> rhfd = RedisHashFreqDist(r, 'test')
	>>> rhfd.items()
	[]
	>>> rhfd.values()
	[]
	>>> len(rhfd)
	0
	>>> rhfd['foo']
	0
	>>> rhfd['foo'] += 1
	>>> rhfd['foo']
	1
	>>> rhfd.items()
	[(b'foo', 1)]
	>>> rhfd.values()
	[1]
	>>> len(rhfd)
	1
	>>> rhfd.clear()
	'''
	def N(self):
		return int(sum(self.values()))
	
	def __missing__(self, key):
		return 0
	
	def __getitem__(self, key):
		return int(RedisHashMap.__getitem__(self, key) or 0)
	
	def values(self):
		return [int(v) for v in RedisHashMap.values(self)]
	
	def items(self):
		return [(k, int(v)) for (k, v) in RedisHashMap.items(self)]

class RedisConditionalHashFreqDist(ConditionalFreqDist):
	'''
	>>> from redis import Redis
	>>> r = Redis()
	>>> rchfd = RedisConditionalHashFreqDist(r, 'condhash')
	>>> rchfd.N()
	0
	>>> rchfd.conditions()
	[]
	>>> rchfd['cond1']['foo'] += 1
	>>> rchfd.N()
	1
	>>> rchfd['cond1']['foo']
	1
	>>> rchfd.conditions()
	['cond1']
	>>> rchfd.clear()
	'''
	def __init__(self, r, name, cond_samples=None):
		self._r = r
		self._name = name
		ConditionalFreqDist.__init__(self, cond_samples)
		
		for key in self._r.keys(encode_key('%s:*' % name)):
			condition = key.split(b':')[1].decode()
			self[condition] # calls self.__getitem__(condition)
	
	def __getitem__(self, condition):
		if condition not in self:
			key = '%s:%s' % (self._name, condition)
			val = RedisHashFreqDist(self._r, key)
			super(RedisConditionalHashFreqDist, self).__setitem__(condition, val)
		
		return super(RedisConditionalHashFreqDist, self).__getitem__(condition)
	
	def clear(self):
		for fdist in self.values():
			fdist.clear()

if __name__ == '__main__':
	import doctest
	doctest.testmod()