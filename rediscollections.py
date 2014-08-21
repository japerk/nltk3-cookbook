import collections, re

white = re.compile('[\s&]+')

def encode_key(key):
	return white.sub('_', key.strip())

class RedisHashMap(collections.MutableMapping):
	def __init__(self, r, name):
		self._r = r
		self._name = encode_key(name)
	
	def __iter__(self):
		return self.items()
	
	def __len__(self):
		return self._r.hlen(self._name)
	
	def __contains__(self, key):
		return self._r.hexists(self._name, encode_key(key))
	
	def __getitem__(self, key):
		return self._r.hget(self._name, encode_key(key))
	
	def __setitem__(self, key, val):
		self._r.hset(self._name, encode_key(key), val)
	
	def __delitem__(self, key):
		self._r.hdel(self._name, encode_key(key))
	
	def keys(self):
		return self._r.hkeys(self._name)
	
	def values(self):
		return self._r.hvals(self._name)
	
	def items(self):
		return self._r.hgetall(self._name).items()
	
	def get(self, key, default=0):
		return self[key] or default
	
	def clear(self):
		self._r.delete(self._name)

class RedisOrderedDict(collections.MutableMapping):
	'''
	>>> from redis import Redis
	>>> r = Redis()
	>>> rod = RedisOrderedDict(r, 'test')
	>>> rod.get('bar')
	0
	>>> len(rod)
	0
	>>> rod['bar'] = 5.2
	>>> rod['bar']
	5.2
	>>> len(rod)
	1
	>>> rod.items()
	[(b'bar', 5.2)]
	>>> rod.clear()
	'''
	def __init__(self, r, name):
		self._r = r
		self._name = encode_key(name)
	
	def __iter__(self):
		return iter(self.items())
	
	def __len__(self):
		return self._r.zcard(self._name)
	
	def __getitem__(self, key):
		return self._r.zscore(self._name, encode_key(key))
	
	def __setitem__(self, key, score):
		self._r.zadd(self._name, encode_key(key), score)
	
	def __delitem__(self, key):
		self._r.zrem(self._name, encode_key(key))
	
	def keys(self, start=0, end=-1):
		# we use zrevrange to get keys sorted by high value instead of by lowest
		return self._r.zrevrange(self._name, start, end)
	
	def values(self, start=0, end=-1):
		return [v for (k, v) in self.items(start=start, end=end)]
	
	def items(self, start=0, end=-1):
		return self._r.zrevrange(self._name, start, end, withscores=True)
	
	def get(self, key, default=0):
		return self[key] or default
	
	def iteritems(self):
		return iter(self)
	
	def clear(self):
		self._r.delete(self._name)

if __name__ == '__main__':
	import doctest
	doctest.testmod()