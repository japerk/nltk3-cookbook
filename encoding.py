# -*- coding: utf-8 -*-
import charade

def detect(s):
	'''
	>>> detect('ascii')
	{'confidence': 1.0, 'encoding': 'ascii'}
	>>> detect('abcdé')
	{'confidence': 0.505, 'encoding': 'utf-8'}
	>>> detect(bytes('abcdé', 'utf-8'))
	{'confidence': 0.505, 'encoding': 'utf-8'}
	>>> detect(bytes('\222\222\223\225', 'latin-1'))
	{'confidence': 0.5, 'encoding': 'windows-1252'}
	'''
	try:
		if isinstance(s, str):
			return charade.detect(s.encode())
		else:
			return charade.detect(s)
	except UnicodeDecodeError:
		return charade.detect(s.encode('utf-8'))

def convert(s):
	'''
	>>> convert('ascii')
	'ascii'
	>>> convert('abcdé')
	'abcdé'
	>>> convert(bytes('abcdé', 'utf-8'))
	'abcdé'
	>>> convert(bytes('\222\222\223\225', 'latin-1'))
	'\u2019\u2019\u201c\u2022'
	'''
	if isinstance(s, str):
		s = s.encode()
	
	encoding = detect(s)['encoding']
	
	if encoding == 'utf-8':
		return s.decode()
	else:
		return s.decode(encoding)

if __name__ == '__main__':
	import doctest
	doctest.testmod()