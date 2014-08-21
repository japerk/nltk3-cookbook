from redis import Redis
from redisprob import RedisHashFreqDist, RedisConditionalHashFreqDist

if __name__ == '__channelexec__':
	host, fd_name, cfd_name = channel.receive()
	r = Redis(host)
	fd = RedisHashFreqDist(r, fd_name)
	cfd = RedisConditionalHashFreqDist(r, cfd_name)
	
	for data in channel:
		if data == 'done':
			channel.send('done')
			break
		
		label, words = data
		
		for word in words:
			fd[word] += 1
			cfd[label][word] += 1