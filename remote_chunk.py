import pickle

if __name__ == '__channelexec__':
	tagger = pickle.loads(channel.receive())
	chunker = pickle.loads(channel.receive())
	
	for sent in channel:
		tree = chunker.parse(tagger.tag(sent))
		channel.send(pickle.dumps(tree))