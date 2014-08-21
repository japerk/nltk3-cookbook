# -*- coding: utf-8 -*-
'''
===================================
Parsing Dates & Times with Dateutil
===================================

>>> from dateutil import parser
>>> parser.parse('Thu Sep 25 10:36:28 2010')
datetime.datetime(2010, 9, 25, 10, 36, 28)
>>> parser.parse('Thursday, 25. September 2010 10:36AM')
datetime.datetime(2010, 9, 25, 10, 36)
>>> parser.parse('9/25/2010 10:36:28')
datetime.datetime(2010, 9, 25, 10, 36, 28)
>>> parser.parse('9/25/2010')
datetime.datetime(2010, 9, 25, 0, 0)
>>> parser.parse('2010-09-25T10:36:28Z')
datetime.datetime(2010, 9, 25, 10, 36, 28, tzinfo=tzutc())

>>> parser.parse('25/9/2010', dayfirst=True)
datetime.datetime(2010, 9, 25, 0, 0)

>>> parser.parse('10-9-25')
datetime.datetime(2025, 10, 9, 0, 0)
>>> parser.parse('10-9-25', yearfirst=True)
datetime.datetime(2010, 9, 25, 0, 0)

>>> try:
...    parser.parse('9/25/2010 at about 10:36AM')
... except ValueError:
...    'cannot parse'
'cannot parse'
>>> parser.parse('9/25/2010 at about 10:36AM', fuzzy=True)
datetime.datetime(2010, 9, 25, 10, 36)


==============================
Timezone Lookup and Conversion
==============================

>>> from dateutil import tz
>>> tz.tzutc()
tzutc()
>>> import datetime
>>> tz.tzutc().utcoffset(datetime.datetime.utcnow())
datetime.timedelta(0)

>>> tz.gettz('US/Pacific')
tzfile('America/Los_Angeles')
>>> tz.gettz('US/Pacific').utcoffset(datetime.datetime.utcnow())
datetime.timedelta(-1, 61200)
>>> tz.gettz('Europe/Paris')
tzfile('Europe/Paris')
>>> tz.gettz('Europe/Paris').utcoffset(datetime.datetime.utcnow())
datetime.timedelta(0, 7200)

>>> pst = tz.gettz('US/Pacific')
>>> dt = datetime.datetime(2010, 9, 25, 10, 36)
>>> dt.tzinfo
>>> dt.astimezone(tz.tzutc())
Traceback (most recent call last):
  File "/usr/lib/python2.6/doctest.py", line 1248, in __run
	compileflags, 1) in test.globs
  File "<doctest __main__[22]>", line 1, in <module>
	dt.astimezone(tz.tzutc())
ValueError: astimezone() cannot be applied to a naive datetime
>>> dt.replace(tzinfo=pst)
datetime.datetime(2010, 9, 25, 10, 36, tzinfo=tzfile('America/Los_Angeles'))
>>> dt.replace(tzinfo=pst).astimezone(tz.tzutc())
datetime.datetime(2010, 9, 25, 17, 36, tzinfo=tzutc())

>>> parser.parse('Wednesday, Aug 4, 2010 at 6:30 p.m. (CDT)', fuzzy=True)
datetime.datetime(2010, 8, 4, 18, 30)
>>> tzinfos = {'CDT': tz.gettz('US/Central')}
>>> parser.parse('Wednesday, Aug 4, 2010 at 6:30 p.m. (CDT)', fuzzy=True, tzinfos=tzinfos)
datetime.datetime(2010, 8, 4, 18, 30, tzinfo=tzfile('America/Chicago'))

>>> tz.tzoffset('custom', 3600)
tzoffset('custom', 3600)

===================================
Extracting URLs from HTML with lxml
===================================

>>> from lxml import html
>>> doc = html.fromstring('Hello <a href="/world">world</a>')
>>> links = list(doc.iterlinks())
>>> len(links)
1
>>> (el, attr, link, pos) = links[0]
>>> attr
'href'
>>> link
'/world'
>>> pos
0

>>> doc.make_links_absolute('http://hello')
>>> abslinks = list(doc.iterlinks())
>>> (el, attr, link, pos) = abslinks[0]
>>> link
'http://hello/world'

>>> links = list(html.iterlinks('Hello <a href="/world">world</a>'))
>>> links[0][2]
'/world'

>>> doc.xpath('//a/@href')[0]
'http://hello/world'


===========================
Cleaning and Stripping HTML
===========================

>>> import lxml.html.clean
>>> lxml.html.clean.clean_html('<html><head></head><body onload=loadfunc()>my text</body></html>')
'<div><body>my text</body></div>'

>>> from bs4 import BeautifulSoup
>>> BeautifulSoup('<div><body>my text</body></div>').get_text()
'my text'


===========================================
Converting HTML Entities with BeautifulSoup
===========================================

>>> from bs4 import BeautifulSoup
>>> BeautifulSoup('&lt;').string
'<'
>>> BeautifulSoup('&amp;').string
'&'

>>> BeautifulSoup('<').string

>>> from bs4 import BeautifulSoup
>>> soup = BeautifulSoup('Hello <a href="/world">world</a>')
>>> [a['href'] for a in soup.findAll('a')]
['/world']

============================================
Detecting and Converting Character Encodings
============================================

>>> import unicodedata
>>> unicodedata.normalize('NFKD', 'abcd\xe9').encode('ascii', 'ignore')
b'abcde'

>>> from bs4 import UnicodeDammit
>>> UnicodeDammit('abcd\xe9').unicode_markup
'abcdé'

'''

if __name__ == '__main__':
	import doctest
	doctest.testmod()
