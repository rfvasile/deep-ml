import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
	d1 = sigma_q / sigma_p
	d2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
	d3 = 1/2
	return np.log10(d1 + d2 - d3)
