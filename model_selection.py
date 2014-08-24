usage = """ a module to contain various model selection routines """
# R. Essick (ressick@mit.edu)
# R. Lynch 


import Posteriors

#=================================================
#
# model selection algorithms
#
#=================================================

def variable_bandwidth(self, g_array, fmin, fmax, ms_params, max_processors=1):
	"""
	varies which frequencies are included to maximize the bayes factor
           
	right now, this is a trivial delegation
	"""
	#Initialize model selection windows             
	win_bins = ms_params  #number of frequency bins to use as model selection window

	bmin = np.where(self.freqs==fmin)[0][0]  #bin index of fmin
	bmax = np.where(self.freqs==fmax)[0][0]  #bin index of fmax

	array_length = bmax + 1 - bmin - win_bins  #span of frequencies over which to model select

	ms_array = np.zeros((array_length, 3))  #2-D array (window position * (f_low, f_up, log bfactor))
	ms_array[:,0] = self.freqs[bmin:(bmax-win_bins+1)]
	ms_array[:,1] = self.freqs[(bmin+win_bins):(bmax+1)]

	#Set up parallelization
	#Divide windows up among processes
	processes = []  #holders for process identification
	finished = 0  #number of finished events

	#Launch processes, limiting the number of active processes to max_processors 
	for iproc in xrange(array_length):
		if len(processes) < max_processors:  #launch another process if there are empty processors

			fl=ms_array[iproc,0]
			fh=ms_array[iproc,1]

			con1, con2 = mp.Pipe()
			args = (g_array, fl, fh, con2)

			p = mp.Process(target=self.calculate_log_bfactor, args=args)
			p.start()
			con2.close()  # this way the process is the only thing that can write to con2
			processes.append((p, iproc, con1))

		else:
			while len(processes) >=  max_processors:  #wait for processes to finish if processors are full
				for ind, (p, _, _) in enumerate(processes):
					if not p.is_alive():  #update ms_array with results of finished processes
						p, ifill, con1 = processes.pop(ind)
						ms_array[ifill,2] = con1.recv()
						finished += 1
						print "Finished %s out of %s model select processes"%(finished, array_length)

			#Launch next process once a process has finished
			fl=ms_array[iproc,0]
			fh=ms_array[iproc,1]

			con1, con2 = mp.Pipe()
			args = (g_array, fl, fh, con2)

			p = mp.Process(target=self.calculate_log_bfactor, args=args)
			p.start()
			con2.close()  # this way the process is the only thing that can write to con2
			processes.append((p, iproc, con1))

	#Wait for processes to all finish, update ms_array as they do finish                            
	while len(processes):
		for ind, (p, _, _) in enumerate(processes):
			if not p.is_alive():
				p, ifill, con1 = processes.pop(ind)
				ms_array[ifill,2] = con1.recv()
				finished += 1
				print "Finished %s out of %s model select processes"%(finished, array_length)

	#Choose window with highest Bayes factor                
	max_log_B = np.amax(ms_array[:,2])
	imax = np.where(ms_array[:,2]==max_log_B)[0][0]

	f_low = ms_array[imax,0]
	f_up = ms_array[imax,1]

	return f_low, f_up, max_log_B

