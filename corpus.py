import lockfile, tempfile, shutil
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, read_blankline_block

class IgnoreHeadingCorpusView(StreamBackedCorpusView):
	def __init__(self, *args, **kwargs):
		StreamBackedCorpusView.__init__(self, *args, **kwargs)
		# open self._stream
		self._open()
		# skip the heading block
		read_blankline_block(self._stream)
		# reset the start position to the current position in the stream
		self._filepos = [self._stream.tell()]

class IgnoreHeadingCorpusReader(PlaintextCorpusReader):
	CorpusView = IgnoreHeadingCorpusView

def append_line(fname, line):
	# lock for writing, released when fp is closed
	with lockfile.FileLock(fname):
		fp = open(fname, 'a+')
		fp.write(line)
		fp.write('\n')
		fp.close()

def remove_line(fname, line):
	'''Remove line from file by creating a temporary file containing all lines
	from original file except those matching the given line, then copying the
	temporary file back into the original file, overwriting its contents.
	'''
	with lockfile.FileLock(fname):
		tmp = tempfile.TemporaryFile()
		fp = open(fname, 'rw+')
		# write all lines from orig file, except if matches given line
		for l in fp:
			if l.strip() != line:
				tmp.write(l)
		
		# reset file pointers so entire files are copied
		fp.seek(0)
		tmp.seek(0)
		# copy tmp into fp, then truncate to remove trailing line(s)
		shutil.copyfileobj(tmp, fp)
		fp.truncate()
		fp.close()
		tmp.close()