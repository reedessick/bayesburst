bayesburst
==========

packages for non-parametric bayesian sky localization

R. Lynch and I think that Gaussian process priors may be useful here to constrain the "degrees of freedom" in the signal with some sort of smoothness constraint. In addition, Lynch thinks using some basis (like a grid of sine-Gaussians a la q-transform) could be useful without adding any additional computational complexity. We are likely going to have to invert large matricies often for marginalization over the Gaussian process prior, which Lynch claims can be done efficiently via kalman filters

  - Ornstein-Uhlenbeck Gaussian process kernel (related to random walks/markov chains)
  - Forward-Backward algorithm (probabilistic graph theory)
  - Sum-Product algorithm 
    - the specialized form for Gaussians is called Gaussian belief propagation
