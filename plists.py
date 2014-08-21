import itertools, execnet

def map(mod, args, specs=[('popen', 2)]):
	gateways = []
	channels = []
	
	for spec, count in specs:
		for i in range(count):
			gw = execnet.makegateway(spec)
			gateways.append(gw)
			channels.append(gw.remote_exec(mod))
	
	cyc = itertools.cycle(channels)
	
	for i, arg in enumerate(args):
		channel = next(cyc)
		channel.send((i, arg))
	
	mch = execnet.MultiChannel(channels)
	queue = mch.make_receive_queue()
	l = len(args)
	results = [None] * l
	
	for j in range(l):
		channel, (i, result) = queue.get()
		results[i] = result
	
	for gw in gateways:
		gw.exit()
	
	return results