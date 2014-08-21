import itertools, execnet, remote_word_count
from nltk.metrics import BigramAssocMeasures
from redis import Redis
from redisprob import RedisHashFreqDist, RedisConditionalHashFreqDist
from rediscollections import RedisOrderedDict

def score_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, host='localhost', specs=[('popen', 2)]):
	gateways = []
	channels = []
	
	for spec, count in specs:
		for i in range(count):
			gw = execnet.makegateway(spec)
			gateways.append(gw)
			channel = gw.remote_exec(remote_word_count)
			channel.send((host, 'word_fd', 'label_word_fd'))
			channels.append(channel)
	
	cyc = itertools.cycle(channels)
	
	for label, words in labelled_words:
		channel = next(cyc)
		channel.send((label, list(words)))
	
	for channel in channels:
		channel.send('done')
		assert 'done' == channel.receive()
		channel.waitclose(5)
	
	for gateway in gateways:
		gateway.exit()
	
	r = Redis(host)
	fd = RedisHashFreqDist(r, 'word_fd')
	cfd = RedisConditionalHashFreqDist(r, 'label_word_fd')
	word_scores = RedisOrderedDict(r, 'word_scores')
	n_xx = cfd.N()
	
	for label in cfd.conditions():
		n_xi = cfd[label].N()
		
		for word, n_ii in cfd[label].items():
			word = word.decode() # must convert to string from bytes
			n_ix = fd[word]
			
			if n_ii and n_ix and n_xi and n_xx:
				score = score_fn(n_ii, (n_ix, n_xi), n_xx)
				word_scores[word] = score
	
	return word_scores