import tensorflow as tf
import numpy as np
from tf_util.stat_util import approx_equal
from tf_util.families import MultivariateNormal

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DTYPE = tf.float64
EPS = 1e-16

class multivariate_normal:
	def __init__(self, dim):
		self.dim = dim

	def compute_suff_stats(self, z):
		first_moment = z;
		second_moment = np.zeros((int(self.dim*(self.dim+1)//2)))
		ind = 0
		for i in range(self.dim):
			for j in range(i, self.dim):
				second_moment[ind] = z[i]*z[j]
				ind += 1
		T_z = np.concatenate((first_moment, second_moment), axis=0)
		return T_z

	def compute_log_base_measure(self, Z):
		return - (self.dim / 2) * np.log(2 * np.pi)


K = 20
n = 100
Ds = [2, 4]

def test_multivariate_normal():
	for dim in Ds:
		# test initialization
		family = MultivariateNormal(dim)
		assert(family.name == 'MultivariateNormal')
		assert(family.D_Z == dim)
		assert(family.num_suff_stats == dim+dim*(dim+1)/2)
		assert(family.has_log_p)
		assert(family.support_mapping == None)
		assert(family.eta_dist['family'] == "iso_mvn_and_iso_iw")
		assert(approx_equal(family.eta_dist['mu'], np.zeros((dim,)), EPS))
		assert(family.eta_dist['scale'] == 0.1)
		assert(family.eta_dist['df_fac'] == 5)

		# test methods
		true_family = multivariate_normal(dim)

		Z = tf.placeholder(dtype=DTYPE, shape=(None, None, dim))
		T_z = family.compute_suff_stats(Z, [], [])
		log_base_measure = family.compute_log_base_measure(Z)


		_Z = np.random.normal(0.0, 1.0, ((K,n,dim)))

		T_z_true = np.zeros((K,n,family.num_suff_stats))
		log_base_measure_true = np.zeros((K,n))
		for k in range(K):
			for i in range(n):
				T_z_true[k,i,:] = true_family.compute_suff_stats(_Z[k,i,:])
				log_base_measure_true[k,i] = true_family.compute_log_base_measure(Z)


		with tf.Session() as sess:
			_T_z, _log_base_measure = sess.run([T_z, log_base_measure], {Z:_Z})
		assert(approx_equal(_T_z, T_z_true, EPS))
		assert(approx_equal(_log_base_measure, log_base_measure_true, EPS))


		# compute mu
		_, _, _, params = family.draw_etas(K)
		for k in range(K):
			mu = params[k]['mu']
			Sigma = params[k]['Sigma']
			mu_true = np.zeros((dim+(dim*(dim+1)//2),))
			mu_true[:dim] = mu
			ind = dim
			for i in range(dim):
				for j in range(i,dim):
					mu_true[ind] = mu[i]*mu[j] + Sigma[i,j]
					ind += 1
			mu_fam = family.compute_mu(params[k])
			assert(approx_equal(mu_true, mu_fam, EPS))



	return None

if __name__ == "__main__":
	test_multivariate_normal()
