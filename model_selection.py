usage = """ a module to contain various model selection routines """
# R. Essick (ressick@mit.edu)
# R. Lynch 

import posteriors
mp = posteriors.mp
np = posteriors.np

print """WARNING
	need to implement model selection (via parallelization if desired)
	develop several algorithms, with varying levels of complexity
	include templated searches (a la ryan's heavyside templates), 
		which may include computing and storing additional terms in posteriors.Posterior
		==> d*B needs to be stored for convenient look-up

	NEED TO IMPLEMENT CONDITION: if max_proc == 1: just compute directly and don't go through mp
	mirror the implementation in posteriors.py? have a single algorithm and then *_mp algorithms
		==> this will allow us to remove the need for posteriors.log_bayes_from_log_posterior_elements, which is awkward 
"""
#=================================================
#
# general utility functions
#
#=================================================

#=================================================
#
# model selection algorithms
#
#=================================================
def variable_bandwidth(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, max_proc=1, n_bins=1):
	raise StandardError, "WRITE ME"

def variable_bandwidth_mp(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, max_proc=1, n_bins=1):
	"""
	basic model selection by sliding a window of fixed width (n_bins) throughout the spectrum defined by freq_truth
	returns a boolean array for the best model and that model's log_bayes
	"""
	if max_proc == 1:
		return variable_bandwidth(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, n_bins=1)


	n_pix, thetas, phis, psis = posterior.__check_theta_phi_psi(thetas, phis, 0.0)
	n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.__check_log_posterior_elements(log_posterior_elements, n_pix)

	### define the sliding frequency ranges for each possible model
	binNos = np.arange(n_freqs)[freq_truth]

	n_models = np.sum(freq_truth)-n_bins
	models = np.zeros((n_models,n_freqs), bool)
	for modelNo in xrange(n_models):
		models[modelNo][binNos[modelNo:modelNo+n_bins]] = True

	log_bayes = np.empty((n_models,), float)

	### launch and reap processes
	procs = []
	for iproc in xrange(n_models):
		if len(procs):
			while len(procs) >= max_proc: ### reap process
				for ind, (p, _, _) in enumerate(procs):
					if not p.is_alive():
						p, modelNo, con1 = procs.pop(ind)
						log_bayes[modelNo] = con1.recv()
						break
		### launch new process
		con1, con2 = mp.Pipe()
		p = mp.Process(target=posteriors.log_bayes_from_log_posterior_elements, args=(posterior, thetas, phis, log_posterior_elements, n_pol_eff, models[iproc], con2))
		p.start()
		con2.close()
		procs.append( (p, iproc, con1) )
	### reap remaining processes
	while len(procs):
		for ind, (p, _, _) in enumerate(procs):
			if not p.is_alive():
				p, modelNo, con1 = procs.pop(ind)
				log_bayes[modelNo] = con1.recv()
				break
	
	### find best model
	best_modelNo = np.argmax(log_bayes)

	return models[best_modelNo], log_bayes[best_modelNo]

###
def log_bayes_cut(log_bayes_thr, posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth):
	raise StandardError, "WRITE ME"


###
def log_bayes_cut_mp(log_bayes_thr, posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, max_proc=1):
	""" 
	keeps only those frequencies with bayes factors larger than the specified threshold
	returns freq_truth respresenting the model and the associated log_bayes
	"""
	if max_proc == 1:
		return log_bayes_cut(log_bayes_thr, posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth)

	n_pix, thetas, phis, psis = posterior.__check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.__check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

        n_models = np.sum(freq_truth)
        models = np.zeros((n_models,n_freqs), bool)
        for modelNo in xrange(n_models):
                models[modelNo][binNos[modelNo]] = True

        log_bayes = np.empty((n_models,), float)

        ### launch and reap processes
        procs = []
        for iproc in xrange(n_models):
                if len(procs):
                        while len(procs) >= max_proc: ### reap process
                                for ind, (p, _, _) in enumerate(procs):
                                        if not p.is_alive():
                                                p, modelNo, con1 = procs.pop(ind)
                                                log_bayes[modelNo] = con1.recv()
                                                break
                ### launch new process
                con1, con2 = mp.Pipe()
                p = mp.Process(target=posteriors.log_bayes_from_log_posterior_elements, args=(posterior, thetas, phis, log_posterior_elements, n_pol_eff, models[iproc], con2))
                p.start()
                con2.close()
                procs.append( (p, iproc, con1) )

        ### reap remaining processes
        while len(procs):
                for ind, (p, _, _) in enumerate(procs):
                        if not p.is_alive():
                                p, modelNo, con1 = procs.pop(ind)
                                log_bayes[modelNo] = con1.recv()
                                break

        ### keep only those bayes factors above the threshold
	keepers = log_bayes >= log_bayes_thr

	model = np.zeros((n_freqs,), bool)
	freq_truth[binNos[log_bayes >= log_bayes_thr]] = True

        return model, posteriors.log_bayes_from_log_posterior_elements(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model)










"""
other possible model selection algorithms
        expanding frequency windows around a single peak
        expanding frequency windows around multiple peaks
        variable bin width based on rate-of-change of signal amplitude (and phase?)
		use mle_strain weighted by the bayes factor at that pixel?
"""










