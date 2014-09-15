usage = """ a module to contain various model selection routines """
# R. Essick (ressick@mit.edu)
# R. Lynch 

import posteriors
mp = posteriors.mp
np = posteriors.np

print """WARNING
	include templated searches (a la ryan's heavyside templates), 
		which may include computing and storing additional terms in posteriors.Posterior
		==> d*B needs to be stored for convenient look-up

	There's a lot of repetition between the different algorithms. Can we clean this up with delegation to a few helper functions?

	We also want the *_mp method to call the associated single-CPU functions, mirroring the setup in posteriors.py
		==> IMPLEMENT THIS (really just logic change for mp calls/model sets)

	other possible model selection algorithms
	        expanding frequency windows around a single peak
        	expanding frequency windows around multiple peaks
	        variable bin width based on rate-of-change of signal amplitude (and phase?)
        	        use mle_strain weighted by the bayes factor at that pixel? ==> weighted average MLE strain defines bin widths and spacing?
			will need to manipulate posterior.dataB and posterior.P directly (a la ryan's templated stuff)
			need to marginalize over time-of-arrival? We can do this numerically as part of this module?
"""
#=================================================
#
# general utility functions
#
#=================================================
def log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, connection=None):
        """ computes the log_bays from log_posterior_elements using posterior class """
        if connection:
                connection.send( posterior.log_bayes( posterior.log_posterior(thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=False) ) )
        else:
                return posterior.log_bayes( posterior.log_posterior(thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=False) )

#=================================================
#
# model selection algorithms
#
#=================================================
###
def fixed_bandwidth(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, connection=False, max_array_size=100):
	"""
	basic model selection by sliding a window of fixed width (n_bins) throughout the spectrum defined by freq_truth
        returns a boolean array for the best model and that model's log_bayes
	"""
        n_pix, thetas, phis, psis = posterior.__check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.__check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

        n_models = np.sum(freq_truth)-n_bins
        if n_models <= 0:
                raise ValueError, "n_models <= 0\n\teither supply more possible bins or lower n_bins"

        models = np.zeros((n_models,n_freqs), bool)
        for modelNo in xrange(n_models):
                models[modelNo][binNos[modelNo:modelNo+n_bins]] = True

	log_bayes = np.array([log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, m) for m in models])

        ### find best model
        best_modelNo = np.argmax(log_bayes)

	if connection:
		utils.flatten_and_send(connection, models[best_modelNo], max_array_size=max_array_size)
		connection.send( log_bayes[best_modelNo] )
	else:
	        return models[best_modelNo], log_bayes[best_modelNo]

###
def fixed_bandwidth_mp(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, n_bins=1, max_proc=1, max_array_size=100):
	"""
	basic model selection by sliding a window of fixed width (n_bins) throughout the spectrum defined by freq_truth
	returns a boolean array for the best model and that model's log_bayes
	parallelization achieved through computing log_bayes for each model as a separate job
	"""
	if max_proc == 1:
		return fixed_bandwidth(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, n_bins=1)

	n_pix, thetas, phis, psis = posterior.__check_theta_phi_psi(thetas, phis, 0.0)
	n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.__check_log_posterior_elements(log_posterior_elements, n_pix)

	### define the sliding frequency ranges for each possible model
	binNos = np.arange(n_freqs)[freq_truth]

	n_models = np.sum(freq_truth)-n_bins
	if n_models <= 0:
		raise ValueError, "n_models <= 0\n\teither supply more possible bins or lower n_bins"

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
		p = mp.Process(target=log_posterior_elements_to_log_bayes, args=(posterior, thetas, phis, log_posterior_elements, n_pol_eff, models[iproc], con2))
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
def variable_bandwidth(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=1, max_n_bins=1, dn_bins=1, connection=False, max_array_size=100):
        """
        slides a frequency window with variable width throughout the spectrum defined by freq_truth
        returns a boolean array for the best model and that model's log_bayes
        """
	if min_n_bins < 1:
		raise ValueError, "min_n_bins must be >= 1"
	if max_n_bins < min_n_bins:
		raise ValueError, "max_n_bins must be >= min_n_bins"
	if dn_bins < 1:
		raise ValueError, "dn_bins must be >=1"
	if not isinstance(dn_bins, int):
		raise TypeError, "dn_bins must be an \"int\""

        n_pix, thetas, phis, psis = posterior.__check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.__check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

	### build models
	sum_freq_truth = np.sum(freq_truth)
	models = []
	for n_bins in np.arange(min_n_bins, max_n_bins, dn_bins, int):
		n_models = sum_freq_truth-n_bins
		if n_models <= 0:
			continue

	        _models = np.zeros((n_models, n_freqs), bool)
        	for modelNo in xrange(n_models):
                	_models[modelNo][binNos[modelNo:modelNo+n_bins]] = True
		models.append( _models )
	if not len(models):
		raise ValueError, "len(models) <= 0\n\tnothing to do"
	models = np.array(models, bool)

        log_bayes = np.array([log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, m) for m in models])

        ### find best model
        best_modelNo = np.argmax(log_bayes)

        if connection:
                utils.flatten_and_send(connection, models[best_modelNo], max_array_size=max_array_size)
                connection.send( log_bayes[best_modelNo] )
        else:
                return models[best_modelNo], log_bayes[best_modelNo]

###
def variable_bandwidth_mp(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=1, max_n_bins=1, dn_bins=1, max_array_size=100, max_proc=1):
        """
        slides a frequency window with variable width throughout the spectrum defined by freq_truth
        returns a boolean array for the best model and that model's log_bayes
	parallelization achieved through computing log_bayes for each model separately
        """
        if min_n_bins < 1:
                raise ValueError, "min_n_bins must be >= 1"
        if max_n_bins < min_n_bins:
                raise ValueError, "max_n_bins must be >= min_n_bins"
        if dn_bins < 1:
                raise ValueError, "dn_bins must be >=1"
        if not isinstance(dn_bins, int):
                raise TypeError, "dn_bins must be an \"int\""

        n_pix, thetas, phis, psis = posterior.__check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.__check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

        ### build models
        sum_freq_truth = np.sum(freq_truth)
        models = []
        for n_bins in np.arange(min_n_bins, max_n_bins, dn_bins, int):
                n_models = sum_freq_truth-n_bins
                if n_models <= 0:
                        continue

                _models = np.zeros((n_models, n_freqs), bool)
                for modelNo in xrange(n_models):
                        _models[modelNo][binNos[modelNo:modelNo+n_bins]] = True
                models.append( _models )
	n_models = len(models)
        if not n_models:
                raise ValueError, "len(models) <= 0\n\tnothing to do"
        models = np.array(models, bool)

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
                p = mp.Process(target=log_posterior_elements_to_log_bayes, args=(posterior, thetas, phis, log_posterior_elements, n_pol_eff, models[iproc], con2))
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
def log_bayes_cut(log_bayes_thr, posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, connection=None, max_array_size=100):
        """ 
        keeps only those frequencies with bayes factors larger than the specified threshold
        returns freq_truth respresenting the model and the associated log_bayes
	"""
        n_pix, thetas, phis, psis = posterior.__check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.__check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

        n_models = np.sum(freq_truth)
        models = np.zeros((n_models,n_freqs), bool)
        for modelNo in xrange(n_models):
                models[modelNo][binNos[modelNo]] = True

        log_bayes = np.array([log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, m) for m in models])

	### keep only those bayes factors above the threshold
        model = np.zeros((n_freqs,), bool)[binNos[log_bayes >= log_bayes_thr]] = True

	if connection:
		utils.flatten_and_send(connection, model, max_array_size=max_array_size)
		connection.send( log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model) )
	else:
	        return model, log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model)

###
def log_bayes_cut_mp(log_bayes_thr, posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, max_proc=1):
	""" 
	keeps only those frequencies with bayes factors larger than the specified threshold
	returns freq_truth respresenting the model and the associated log_bayes
	parallelization achieved through computation of log_bayes for each frequency bin separately
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
                p = mp.Process(target=log_posterior_elements_to_log_bayes, args=(posterior, thetas, phis, log_posterior_elements, n_pol_eff, models[iproc], con2))
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
	model = np.zeros((n_freqs,), bool)[binNos[log_bayes >= log_bayes_thr]] = True

        return model, log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model)



