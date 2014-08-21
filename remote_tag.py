import pickle

if __name__ == '__channelexec__':
	tagger = pickle.loads(channel.receive())
	
	for sentence in channel:
		channel.send(tagger.tag(sentence))