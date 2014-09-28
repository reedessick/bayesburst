usage = """ a module to contain various model selection routines """
# R. Essick (ressick@mit.edu)
# R. Lynch 

import utils
np = utils.np
hp = utils.hp
import multiprocessing as mp

print """WARNING
	include templated searches (a la ryan's heavyside templates), 
		which may include computing and storing additional terms in posteriors.Posterior
		==> d*B needs to be stored for convenient look-up

	There's a lot of repetition between the different algorithms. Can we clean this up with delegation to a few helper functions?

	We also want the *_mp method to call the associated single-CPU functions, mirroring the setup in posteriors.py
		==> IMPLEMENT THIS (really just logic change for mp calls/model sets)

	other possible model selection algorithms
		variable resolution (change df for more parsimonious models)
	        expanding frequency windows around a single peak
        	expanding frequency windows around multiple peaks
		brute force combinatorics of all possible models
			extremely expensive...will require massive parallelization (across a cluster?)
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
# model averaging
#
#=================================================
def model_average(posterior, thetas, phis, log_posterior_elements, n_pol_eff, models, connection=None, max_array_size=100):
	""" 
	compute the model-averaged log_posterior using 'models' 
	returns log_posterior
	"""
	n_pix, thetas, phis, psis = posterior.check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.check_log_posterior_elements(log_posterior_elements, n_pix)

	if len(np.shape(models)) != 2:
		raise ValueError, "bad shape for models"
	n_models, n_f = np.shape(models)

	log_posterior = np.zeros((n_pix, n_models), float)
	log_bayes = np.zeros((n_models,), float)

	### compute posterior and log_bayes for each model
	for modelNo, model in enumerate(models):
		this_log_posterior = posterior.log_posterior(thetas, phis, log_posterior_elements, n_pol_eff, model, normalize=False)
		log_posterior[:,modelNo] = this_log_posterior - utils.sum_logs( this_log_posterior ) ### normalize
		log_bayes[modelNo] = posterior.log_bayes( this_log_posterior ) ### compute log_Bayes

	### average!
	###            ### weighted sum                        ### normalize the sum
	averaged_log_posterior = utils.sum_logs( log_posterior+log_bayes ) - utils.sum_logs( log_bayes )
	if connection:
		utils.flatten_and_send(connection, averaged_log_posterior, max_array_size=max_array_size)
	else:
		return averaged_log_posterior

#=================================================
#
# model selection algorithms
#
#=================================================
def log_bayes_cut(log_bayes_thr, posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, connection=None, max_array_size=100, joint_log_bayes=False):
        """ 
        keeps only those frequencies with bayes factors larger than the specified threshold
        returns freq_truth respresenting the model and the associated log_bayes
        """
        n_pix, thetas, phis, psis = posterior.check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

        n_models = np.sum(freq_truth)
        models = np.zeros((n_models, n_freqs), bool)
        for modelNo in xrange(n_models):
                models[modelNo][binNos[modelNo]] = True

        log_bayes = np.array([log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, m) for m in models])

        ### keep only those bayes factors above the threshold
        model = np.zeros((n_freqs,), bool)
        model[binNos[log_bayes >= log_bayes_thr]] = True

        if connection:
                utils.flatten_and_send(connection, model, max_array_size=max_array_size)
                if joint_log_bayes:
                        connection.send( log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model) )
        else:
                if joint_log_bayes:
                        return model, log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model)
                else:
                        return model

###
def log_bayes_cut_mp(log_bayes_thr, posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, num_proc=1, max_proc=1, max_array_size=100, joint_log_bayes=False):
        """ 
        keeps only those frequencies with bayes factors larger than the specified threshold
        returns freq_truth respresenting the model and the associated log_bayes
	        parallelization achieved through delegation to log_bayes_cut
        """
        if num_proc == 1:
                return log_bayes_cut(log_bayes_thr, posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, joint_log_bayes=joint_log_bayes)

        n_pix, thetas, phis, psis = posterior.check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

        nbins_per_proc = int(np.ceil(1.0*np.sum(freq_truth)/num_proc))

        models = np.zeros((num_proc,n_freqs), bool)
        for modelNo in xrange(num_proc):
                models[modelNo][binNos[modelNo*nbins_per_proc:(modelNo+1)*nbins_per_proc]] = True

        model = np.zeros((n_freqs,),bool)

        procs = []
        for iproc in xrange(num_proc):
                if len(procs):
                        if len(procs) >= max_proc: ### reap old processes
                                p, i, con1 = procs.pop(0)
                                model += utils.recv_and_reshape(con1, (n_freqs), max_array_size=max_array_size, dtype=bool)

                ### launch new process
                con1, con2 = mp.Pipe()
                p = mp.Process(target=log_bayes_cut, args=(log_bayes_thr, posterior, thetas, phis, log_posterior_elements, n_pol_eff, models[iproc], con2, max_array_size, False))
                p.start()
                con2.close()
                procs.append( (p, iproc, con1) )

        while len(procs):
                p, i, con1 = procs.pop(0)
                model += utils.recv_and_reshape(con1, (n_freqs), max_array_size=max_array_size, dtype=bool)

	model = model.astype(bool)

        if joint_log_bayes:
                return model, log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model)
        else:
                return model

###
def fixed_bandwidth(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, n_bins=1, connection=False, max_array_size=100):
	"""
	basic model selection by sliding a window of fixed width (n_bins) throughout the spectrum defined by freq_truth
        returns a boolean array for the best model and that model's log_bayes
	"""
        n_pix, thetas, phis, psis = posterior.check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

        n_models = np.sum(freq_truth)-n_bins+1
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
def fixed_bandwidth_mp(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, n_bins=1, num_proc=1, max_proc=1, max_array_size=100):
	"""
	basic model selection by sliding a window of fixed width (n_bins) throughout the spectrum defined by freq_truth
	returns a boolean array for the best model and that model's log_bayes
		parallelization achieved through delegation to fixed_bandwidth
	"""
	if num_proc == 1:
		return fixed_bandwidth(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, n_bins=n_bins)

	n_pix, thetas, phis, psis = posterior.check_theta_phi_psi(thetas, phis, 0.0)
	n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.check_log_posterior_elements(log_posterior_elements, n_pix)

	### define the sliding frequency ranges for each possible model
	binNos = np.arange(n_freqs)[freq_truth]

	nBins = np.sum(freq_truth) ### the number of accessible bins

	if n_bins > nBins-num_proc:
		raise ValueError, "n_bins > delegated frequency ranges. try reducing num_proc"
                ### we may want to handle this more gracefully than throwing an error
                ### launch remaining jobs separately?
                ### need to think through this logic carefully

	model_sets = np.zeros((num_proc,n_freqs), bool)
	for iproc in xrange(num_proc):
		model_sets[iproc][binNos[iproc:nBins+(iproc-num_proc)]] = True

	log_bayes = np.empty((num_proc,),float)
	models = np.empty((num_proc,n_freqs),bool)

	procs = []
	for iproc in xrange(num_proc):
		if len(procs):
			if len(procs) >= max_proc: ### reap old process
				p, i, con1 = procs.pop(0)
				models[i] = utils.recv_and_reshape(con1, (n_freqs), max_array_size=max_array_size)
				log_bayes[i] = con1.recv()			

		### launch new process
                con1, con2 = mp.Pipe()
                p = mp.Process(target=fixed_bandwidth, args=(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model_sets[iproc], n_bins, con2, max_array_size))
                p.start()
                con2.close()
                procs.append( (p, iproc, con1) )

        while len(procs):
                p, i, con1 = procs.pop(0)
		models[i] = utils.recv_and_reshape(con1, (n_freqs), max_array_size=max_array_size)
		log_bayes[i] = con1.recv()

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

        n_pix, thetas, phis, psis = posterior.check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

	### build models
	sum_freq_truth = np.sum(freq_truth)
	models = []
	for n_bins in np.arange(min_n_bins, max_n_bins+dn_bins, dn_bins, int):
		n_models = sum_freq_truth-n_bins + 1
		if n_models <= 0:
			continue

        	for modelNo in xrange(n_models):
			_model = np.zeros((n_freqs,),bool)
			_model[binNos[modelNo:modelNo+n_bins]] = True
			models.append( _model )

	if not len(models):
		raise ValueError, "len(models) <= 0\n\tnothing to do"

	models = np.array(models)

        log_bayes = np.array([log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, m) for m in models])

        ### find best model
        best_modelNo = np.argmax(log_bayes)

        if connection:
                utils.flatten_and_send(connection, models[best_modelNo], max_array_size=max_array_size)
                connection.send( log_bayes[best_modelNo] )
        else:
                return models[best_modelNo], log_bayes[best_modelNo]

###
def variable_bandwidth_mp(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=1, max_n_bins=1, dn_bins=1, num_proc=1, max_proc=1, max_array_size=100):
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

	if num_proc==1:
		return variable_bandwidth(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=min_n_bins, max_n_bins=max_n_bins, dn_bins=dn_bins)

        n_pix, thetas, phis, psis = posterior.check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

	nBins = np.sum(freq_truth) ### the number of bins available to the model

	### check that biggest bin still fits in model_sets
        if max_n_bins > nBins-num_proc:
                raise ValueError, "max_n_bins > delegated frequency ranges. try reducing num_proc"
		### we may want to handle this more gracefully than throwing an error
		### launch variable_bandwidth_mp on the remaining jobs with different num_proc?
		### need to think through this logic carefully

        model_sets = np.zeros((num_proc,n_freqs), bool)
        for iproc in xrange(num_proc):
		istop = iproc-num_proc+1
		if istop == 0: ### last index is the last element of the array
	                model_sets[iproc][binNos[iproc:]] = True
		else:
			model_sets[iproc][binNos[iproc:istop]] = True

        log_bayes = np.empty((num_proc,),float)
        models = np.empty((num_proc,n_freqs),bool)

        procs = []
        for iproc in xrange(num_proc):
                if len(procs):
                        if len(procs) >= max_proc: ### reap old process
                                p, i, con1 = procs.pop(0)
                                models[i] = utils.recv_and_reshape(con1, (n_freqs), max_array_size=max_array_size)
                                log_bayes[i] = con1.recv()

                ### launch new process
                con1, con2 = mp.Pipe()
                p = mp.Process(target=variable_bandwidth, args=(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model_sets[iproc], min_n_bins, max_n_bins, dn_bins, con2, max_array_size))
                p.start()
                con2.close()
                procs.append( (p, iproc, con1) )

        while len(procs):
                p, i, con1 = procs.pop(0)
                models[i] = utils.recv_and_reshape(con1, (n_freqs), max_array_size=max_array_size)
                log_bayes[i] = con1.recv()

        ### find best model
        best_modelNo = np.argmax(log_bayes)

        return models[best_modelNo], log_bayes[best_modelNo]

###
def waterfill(posterior, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, connection=None, max_array_size=100):
	"""
	continues to add bins from the top down until the marginal bayes factor becomes negative
	
	this method may be fragile. If no individual bin has enough evidence, then no bins will be returned even if there is a detectable signal when we include several bins
	"""
        n_pix, thetas, phis, psis = posterior.check_theta_phi_psi(thetas, phis, 0.0)
        n_pix, n_gaus, n_freqs, log_posterior_elements = posterior.check_log_posterior_elements(log_posterior_elements, n_pix)

        ### define the sliding frequency ranges for each possible model
        binNos = np.arange(n_freqs)[freq_truth]

	nBins = np.sum(freq_truth)

        n_models = np.sum(freq_truth)
        models = np.zeros((n_models, n_freqs), bool)
        for modelNo in xrange(n_models):
                models[modelNo][binNos[modelNo]] = True

	### compute bayes factors for bins individually
        log_bayes = np.array([log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, m) for m in models])

	### need to iteratively find the maximum and expand around it.
	model = np.zeros((n_freqs,), bool) ### set up the model

	old_model = np.zeros((n_freqs,), bool)
	old_logB = log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, old_model)

	while np.sum(old_model) < nBins:
		binNo = log_bayes.argmax() ### bin with max individual log_bayes
		log_bayes[binNo] = -np.infty ### so we don't pick it up again
		model[binNo] = True ### add this bin
		logB = log_posterior_elements_to_log_bayes(posterior, thetas, phis, log_posterior_elements, n_pol_eff, model)
		if logB >= old_logB: ### add this bin, and tend toward including marginal bins
			old_model[binNo] = True
			old_logB = logB
		else: ### marginal bin is not helpful
			break

	return old_model, old_logB
