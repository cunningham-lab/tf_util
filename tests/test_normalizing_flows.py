import tensorflow as tf
import numpy as np
from tf_util.stat_util import approx_equal
from tf_util.normalizing_flows import AffineFlow, \
									  CholProdFlow, \
                                      ElemMultFlow, \
                                      ExpFlow, \
                                      IntervalFlow, \
									  PlanarFlow, \
									  RadialFlow, \
									  ShiftFlow, \
									  SimplexBijectionFlow, \
									  SoftPlusFlow, \
									  StructuredSpinnerFlow, \
									  StructuredSpinnerTanhFlow, \
									  TanhFlow

from tf_util.normalizing_flows import get_num_flow_params, \
                                      get_flow_param_inits, \
                                      get_flow_class

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

dtype = tf.float64
EPS = 1e-10

# write the ground truth functions for the normalizing flows

# Affine flows
def affine_flow(z, params):
	"""Affine flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.

	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(AffineFlow, D))

	A = np.reshape(params[:D**2], (D,D))
	b = params[D**2:]

	# compute output
	out = np.dot(A, np.expand_dims(z, 1))[:,0] + b

	# compute log abs det jacobian
	log_det_jac = np.linalg.det(A)

	return out, log_det_jac

# Cholesky product flows
def chol_prod_flow(z, params):
	"""Cholesky product flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(CholProdFlow, D))

	raise NotImplementedError()

# Elementwise multiplication flows
def elem_mult_flow(z, params):
	"""Elementwise multiplication flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(ElemMultFlow, D))

	a = params

	# compute output
	out = np.multiply(a, z)

	# compute log abs det jacobian
	log_det_jac = np.sum(np.log(np.abs(a)))

	return out, log_det_jac


# Interval flows
def interval_flow(z, params):
	"""Interval flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(IntervalFlow, D))

	raise NotImplementedError()

# Planar flows
def planar_flow(z, params):
	"""Planar flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(PlanarFlow, D))

	_u = params[:D]
	w = params[D:(2*D)]
	b = params[2*D]

	# enforce w^\topu >= -1
	wdotu = np.dot(w, _u)
	m_wdotu = -1.0 + np.log(1.0 + np.exp(wdotu))
	u = _u + (m_wdotu - wdotu) * w / np.dot(w,w)

	# compute output
	out = z + u*np.tanh(np.dot(w, z) + b)

	# compute log det jacobian
	phi = (1.0 - np.square(np.tanh(np.dot(w, z) + b))) * w
	log_det_jac = np.log(np.abs(1.0 + np.dot(u, phi)))

	return out, log_det_jac

# Radial flows
def radial_flow(z, params):
	"""Radial flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(RadialFlow, D))

	raise NotImplementedError()

# Shift flows
def shift_flow(z, params):
	"""Shift flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(ShiftFlow, D))

	b = params

	# compute output
	out = z + b

	# compute the log abs det jacobian
	log_det_jac = 0.0
	return out, log_det_jac

# Simplex bijection flows
def simplex_bijection_flow(z, params):
	"""Simplex bijection flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(SimplexBijectionFlow, D))

	raise NotImplementedError()

# Softplus flows
def softplus_flow(z, params):
	"""Softplus flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(SoftPlusFlow, D))

	# compute output
	out = np.log(1.0 + np.exp(z))

	# compute log abs det jacobian
	jac_diag = np.exp(z) / (1.0 + np.exp(z))
	log_det_jac = np.sum(np.log(np.abs(jac_diag)))

	return out, log_det_jac


# Structured spinner flows
def structured_spinner_flow(z, params):
	"""Structured spinner flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(StructuredSpinnerFlow, D))

	raise NotImplementedError()

# Structured spinner tanh flows
def structured_spinner_tanh_flow(z, params):
	"""Structured spinner tanh flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(StructuredSpinnerTanhFlow, D))

	raise NotImplementedError()

# Tanh flows
def tanh_flow(z, params):
	"""Tanh flow operation and log abs det jac.

	[Insert tex of operation]

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_num_flow_params(TanhFlow, D))

	raise NotImplementedError()



def eval_flow_at_dim(flow_class, true_flow, dim, K, n):
	num_params = get_num_flow_params(flow_class, dim)

	params1 = tf.placeholder(dtype=dtype, shape=(None, num_params))
	inputs1 = tf.placeholder(dtype=dtype, shape=(None, None, dim))

	_params = np.random.normal(0.0, 1.0, (K,num_params))
	_inputs = np.random.normal(0.0, 1.0, (K,n,dim))

	# compute ground truth
	out_true = np.zeros((K,n,dim))
	log_det_jac_true = np.zeros((K,n))
	for k in range(K):
		_params_k = _params[k,:]
		for j in range(n):
			out_true[k,j,:], log_det_jac_true[k,j] = true_flow(_inputs[k,j,:], _params_k)

	flow1 = flow_class(params1, inputs1)
	out1, log_det_jac1 = flow1.forward_and_jacobian()

	feed_dict = {params1:_params, inputs1:_inputs}
	with tf.Session() as sess:
		_out1, _log_det_jac1 = sess.run([out1, log_det_jac1], feed_dict)

		if (flow1.name == 'PlanarFlow'):
			wdotus = wdotu = tf.matmul(tf.expand_dims(flow1.w, 1), tf.expand_dims(flow1.u, 2))
			_wdotus = sess.run(wdotus, feed_dict)


	assert(approx_equal(_out1, out_true, 1e-16))
	assert(approx_equal(_log_det_jac1, log_det_jac_true, 1e-16))

	if (flow1.name == 'PlanarFlow'):
		num_inv_viols = np.sum(_wdotus < -(1+EPS))
		assert(num_inv_viols == 0)

	#print(flow1.name + ' passed at dim=%d.' % dim)
	return None


def test_get_flow_class():
	assert(get_flow_class('AffineFlow') == AffineFlow)
	assert(get_flow_class('CholProdFlow') == CholProdFlow)
	assert(get_flow_class('ElemMultFlow') == ElemMultFlow)
	assert(get_flow_class('ExpFlow') == ExpFlow)
	assert(get_flow_class('IntervalFlow') == IntervalFlow)
	assert(get_flow_class('PlanarFlow') == PlanarFlow)
	assert(get_flow_class('RadialFlow') == RadialFlow)
	assert(get_flow_class('ShiftFlow') == ShiftFlow)
	assert(get_flow_class('SimplexBijectionFlow') == SimplexBijectionFlow)
	assert(get_flow_class('SoftPlusFlow') == SoftPlusFlow)
	assert(get_flow_class('StructuredSpinnerFlow') == StructuredSpinnerFlow)
	assert(get_flow_class('StructuredSpinnerTanhFlow') == StructuredSpinnerTanhFlow)
	assert(get_flow_class('TanhFlow') == TanhFlow)
	

	print('Get flow class passed.')
	return None


def test_get_num_flow_params():
	assert(get_num_flow_params(AffineFlow, 1) == 2)
	assert(get_num_flow_params(AffineFlow, 2) == 6)
	assert(get_num_flow_params(AffineFlow, 4) == 20)
	assert(get_num_flow_params(AffineFlow, 20) == 420)
	assert(get_num_flow_params(AffineFlow, 100) == 10100)
	assert(get_num_flow_params(AffineFlow, 1000) == 1001000)

	"""
	assert(get_num_flow_params(CholProdFlow, 1) == 0)
	assert(get_num_flow_params(CholProdFlow, 2) == 0)
	assert(get_num_flow_params(CholProdFlow, 4) == 0)
	assert(get_num_flow_params(CholProdFlow, 20) == 0)
	assert(get_num_flow_params(CholProdFlow, 100) == 0)
	assert(get_num_flow_params(CholProdFlow, 1000) == 0)
	"""

	assert(get_num_flow_params(ElemMultFlow, 1) == 1)
	assert(get_num_flow_params(ElemMultFlow, 2) == 2)
	assert(get_num_flow_params(ElemMultFlow, 4) == 4)
	assert(get_num_flow_params(ElemMultFlow, 20) == 20)
	assert(get_num_flow_params(ElemMultFlow, 100) == 100)
	assert(get_num_flow_params(ElemMultFlow, 1000) == 1000)

	assert(get_num_flow_params(IntervalFlow, 1) == 2)
	assert(get_num_flow_params(IntervalFlow, 2) == 2)
	assert(get_num_flow_params(IntervalFlow, 4) == 2)
	assert(get_num_flow_params(IntervalFlow, 20) == 2)
	assert(get_num_flow_params(IntervalFlow, 100) == 2)
	assert(get_num_flow_params(IntervalFlow, 1000) == 2)

	assert(get_num_flow_params(PlanarFlow, 1) == 3)
	assert(get_num_flow_params(PlanarFlow, 2) == 5)
	assert(get_num_flow_params(PlanarFlow, 4) == 9)
	assert(get_num_flow_params(PlanarFlow, 20) == 41)
	assert(get_num_flow_params(PlanarFlow, 100) == 201)
	assert(get_num_flow_params(PlanarFlow, 1000) == 2001)

	assert(get_num_flow_params(RadialFlow, 1) == 3)
	assert(get_num_flow_params(RadialFlow, 2) == 4)
	assert(get_num_flow_params(RadialFlow, 4) == 6)
	assert(get_num_flow_params(RadialFlow, 20) == 22)
	assert(get_num_flow_params(RadialFlow, 100) == 102)
	assert(get_num_flow_params(RadialFlow, 1000) == 1002)

	assert(get_num_flow_params(ShiftFlow, 1) == 1)
	assert(get_num_flow_params(ShiftFlow, 2) == 2)
	assert(get_num_flow_params(ShiftFlow, 4) == 4)
	assert(get_num_flow_params(ShiftFlow, 20) == 20)
	assert(get_num_flow_params(ShiftFlow, 100) == 100)
	assert(get_num_flow_params(ShiftFlow, 1000) == 1000)

	assert(get_num_flow_params(SimplexBijectionFlow, 1) == 0)
	assert(get_num_flow_params(SimplexBijectionFlow, 2) == 0)
	assert(get_num_flow_params(SimplexBijectionFlow, 4) == 0)
	assert(get_num_flow_params(SimplexBijectionFlow, 20) == 0)
	assert(get_num_flow_params(SimplexBijectionFlow, 100) == 0)
	assert(get_num_flow_params(SimplexBijectionFlow, 1000) == 0)

	assert(get_num_flow_params(SoftPlusFlow, 1) == 0)
	assert(get_num_flow_params(SoftPlusFlow, 2) == 0)
	assert(get_num_flow_params(SoftPlusFlow, 4) == 0)
	assert(get_num_flow_params(SoftPlusFlow, 20) == 0)
	assert(get_num_flow_params(SoftPlusFlow, 100) == 0)
	assert(get_num_flow_params(SoftPlusFlow, 1000) == 0)

	"""
	assert(get_num_flow_params(StructuredSpinnerFlow, 1) == 0)
	assert(get_num_flow_params(StructuredSpinnerFlow, 2) == 0)
	assert(get_num_flow_params(StructuredSpinnerFlow, 4) == 0)
	assert(get_num_flow_params(StructuredSpinnerFlow, 20) == 0)
	assert(get_num_flow_params(StructuredSpinnerFlow, 100) == 0)
	assert(get_num_flow_params(StructuredSpinnerFlow, 1000) == 0)
	"""

	"""
	assert(get_num_flow_params(StructuredSpinnerTanhFlow, 1) == 0)
	assert(get_num_flow_params(StructuredSpinnerTanhFlow, 2) == 0)
	assert(get_num_flow_params(StructuredSpinnerTanhFlow, 4) == 0)
	assert(get_num_flow_params(StructuredSpinnerTanhFlow, 20) == 0)
	assert(get_num_flow_params(StructuredSpinnerTanhFlow, 100) == 0)
	assert(get_num_flow_params(StructuredSpinnerTanhFlow, 1000) == 0)
	"""

	assert(get_num_flow_params(TanhFlow, 1) == 0)
	assert(get_num_flow_params(TanhFlow, 2) == 0)
	assert(get_num_flow_params(TanhFlow, 4) == 0)
	assert(get_num_flow_params(TanhFlow, 20) == 0)
	assert(get_num_flow_params(TanhFlow, 100) == 0)
	assert(get_num_flow_params(TanhFlow, 1000) == 0)

	print('Get number of flow parameters passed.')
	return None


def test_flow_param_initialization():
	Ds = [1,2,4,20,100,1000]
	all_glorot_uniform_flows = [AffineFlow, ElemMultFlow, ShiftFlow]
	all_no_param_flows = [ExpFlow, SimplexBijectionFlow, SoftPlusFlow, TanhFlow]
	with tf.Session() as sess:
		for D in Ds:
			for flow in all_glorot_uniform_flows:
				inits, dims = get_flow_param_inits(flow, D)
				assert(len(inits)==1)
				#assert(isinstance(inits[0], tf.glorot_uniform_initializer))
				assert(sum(dims) == get_num_flow_params(flow, D))

			for flow in all_no_param_flows:
				inits, dims = get_flow_param_inits(flow, D)
				assert(len(inits) == 1)
				assert(inits[0] == None)
				assert(len(dims) == 1)
				assert(dims[0] == 0)

			"""
			inits, dims = get_flow_param_inits(CholProdFlow, D)
			"""

			inits, dims = get_flow_param_inits(PlanarFlow, D)
			assert(approx_equal(sess.run(inits[0]), np.zeros(D), EPS))
			#assert(isinstance(inits[1], tf.glorot_uniform_initializer))
			assert(approx_equal(sess.run(inits[2]), 0.0, EPS))
			assert(dims == [D, D, 1])

			"""
			inits, dims = get_flow_param_inits(RadialFlow, D)
			"""

			"""
			inits, dims = get_flow_param_inits(StructuredSpinnerFlow, D)
			"""

			"""
			inits, dims = get_flow_param_inits(StructuredSpinnerTanhFlow, D)
			"""

	print('Flow initializations passed.')
	return None


def test_affine_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(AffineFlow, affine_flow, 1, K, n)
	eval_flow_at_dim(AffineFlow, affine_flow, 2, K, n)
	eval_flow_at_dim(AffineFlow, affine_flow, 4, K, n)
	eval_flow_at_dim(AffineFlow, affine_flow, 20, K, n)
	eval_flow_at_dim(AffineFlow, affine_flow, 100, K, n)
	eval_flow_at_dim(AffineFlow, affine_flow, 1000, K, n)
	print('Affine flows passed.')
	return None

"""
def test_chol_prod_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(CholProdFlow, chol_prod_flow, 1, K, n)
	eval_flow_at_dim(CholProdFlow, chol_prod_flow, 2, K, n)
	eval_flow_at_dim(CholProdFlow, chol_prod_flow, 4, K, n)
	eval_flow_at_dim(CholProdFlow, chol_prod_flow, 20, K, n)
	eval_flow_at_dim(CholProdFlow, chol_prod_flow, 100, K, n)
	eval_flow_at_dim(CholProdFlow, chol_prod_flow, 1000, K, n)
	print('Cholesky product flows passed.')
	return None
"""


def test_elem_mult_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(ElemMultFlow, elem_mult_flow, 1, K, n)
	eval_flow_at_dim(ElemMultFlow, elem_mult_flow, 2, K, n)
	eval_flow_at_dim(ElemMultFlow, elem_mult_flow, 4, K, n)
	eval_flow_at_dim(ElemMultFlow, elem_mult_flow, 20, K, n)
	eval_flow_at_dim(ElemMultFlow, elem_mult_flow, 100, K, n)
	eval_flow_at_dim(ElemMultFlow, elem_mult_flow, 1000, K, n)
	print('Elementwise multiplication flows passed.')
	return None

"""
def test_exp_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(ExpFlow, exp_flow, 1, K, n)
	eval_flow_at_dim(ExpFlow, exp_flow, 2, K, n)
	eval_flow_at_dim(ExpFlow, exp_flow, 4, K, n)
	eval_flow_at_dim(ExpFlow, exp_flow, 20, K, n)
	eval_flow_at_dim(ExpFlow, exp_flow, 100, K, n)
	eval_flow_at_dim(ExpFlow, exp_flow, 1000, K, n)
	print('Exp flows passed.')
	return None
"""

"""
def test_interval_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(IntervalFlow, interval_flow, 1, K, n)
	eval_flow_at_dim(IntervalFlow, interval_flow, 2, K, n)
	eval_flow_at_dim(IntervalFlow, interval_flow, 4, K, n)
	eval_flow_at_dim(IntervalFlow, interval_flow, 20, K, n)
	eval_flow_at_dim(IntervalFlow, interval_flow, 100, K, n)
	eval_flow_at_dim(IntervalFlow, interval_flow, 1000, K, n)
	print('Interval flows passed.')
	return None
"""

def test_planar_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	np.random.seed(0)
	eval_flow_at_dim(PlanarFlow, planar_flow, 1, K, n)
	eval_flow_at_dim(PlanarFlow, planar_flow, 2, K, n)
	eval_flow_at_dim(PlanarFlow, planar_flow, 4, K, n)
	eval_flow_at_dim(PlanarFlow, planar_flow, 20, K, n)
	eval_flow_at_dim(PlanarFlow, planar_flow, 100, K, n)
	eval_flow_at_dim(PlanarFlow, planar_flow, 1000, K, n)
	print('Planar flows passed.')
	return None

"""
def test_radial_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(RadialFlow, radial_flow, 1, K, n)
	eval_flow_at_dim(RadialFlow, radial_flow, 2, K, n)
	eval_flow_at_dim(RadialFlow, radial_flow, 4, K, n)
	eval_flow_at_dim(RadialFlow, radial_flow, 20, K, n)
	eval_flow_at_dim(RadialFlow, radial_flow, 100, K, n)
	eval_flow_at_dim(RadialFlow, radial_flow, 1000, K, n)
	print('Radial flows passed.')
	return None
"""

def test_shift_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(ShiftFlow, shift_flow, 1, K, n)
	eval_flow_at_dim(ShiftFlow, shift_flow, 2, K, n)
	eval_flow_at_dim(ShiftFlow, shift_flow, 4, K, n)
	eval_flow_at_dim(ShiftFlow, shift_flow, 20, K, n)
	eval_flow_at_dim(ShiftFlow, shift_flow, 100, K, n)
	eval_flow_at_dim(ShiftFlow, shift_flow, 1000, K, n)
	print('Shift flows passed.')
	return None

"""
def test_simplex_bijection_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 1, K, n)
	eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 2, K, n)
	eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 4, K, n)
	eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 20, K, n)
	eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 100, K, n)
	eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 1000, K, n)
	print('Simplex bijection flows passed.')
	return None
"""


def test_softplus_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(SoftPlusFlow, softplus_flow, 1, K, n)
	eval_flow_at_dim(SoftPlusFlow, softplus_flow, 2, K, n)
	eval_flow_at_dim(SoftPlusFlow, softplus_flow, 4, K, n)
	eval_flow_at_dim(SoftPlusFlow, softplus_flow, 20, K, n)
	eval_flow_at_dim(SoftPlusFlow, softplus_flow, 100, K, n)
	eval_flow_at_dim(SoftPlusFlow, softplus_flow, 1000, K, n)
	print('SoftPlus flows passed.')
	return None

"""
def test_structured_spinner_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(StructuredSpinnerFlow, structured_spinner_flow, 1, K, n)
	eval_flow_at_dim(StructuredSpinnerFlow, structured_spinner_flow, 2, K, n)
	eval_flow_at_dim(StructuredSpinnerFlow, structured_spinner_flow, 4, K, n)
	eval_flow_at_dim(StructuredSpinnerFlow, structured_spinner_flow, 20, K, n)
	eval_flow_at_dim(StructuredSpinnerFlow, structured_spinner_flow, 100, K, n)
	eval_flow_at_dim(StructuredSpinnerFlow, structured_spinner_flow, 1000, K, n)
	print('Structured spinner flows passed.')
	return None
"""

"""
def test_structured_spinner_tanh_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(StructuredSpinnerTanhFlow, structured_spinner_tanh_flow, 1, K, n)
	eval_flow_at_dim(StructuredSpinnerTanhFlow, structured_spinner_tanh_flow, 2, K, n)
	eval_flow_at_dim(StructuredSpinnerTanhFlow, structured_spinner_tanh_flow, 4, K, n)
	eval_flow_at_dim(StructuredSpinnerTanhFlow, structured_spinner_tanh_flow, 20, K, n)
	eval_flow_at_dim(StructuredSpinnerTanhFlow, structured_spinner_tanh_flow, 100, K, n)
	eval_flow_at_dim(StructuredSpinnerTanhFlow, structured_spinner_tanh_flow, 1000, K, n)
	print('Structured spinner tanh flows passed.')
	return None
"""

"""
def test_tanh_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(TanhFlow, tanh_flow, 1, K, n)
	eval_flow_at_dim(TanhFlow, tanh_flow, 2, K, n)
	eval_flow_at_dim(TanhFlow, tanh_flow, 4, K, n)
	eval_flow_at_dim(TanhFlow, tanh_flow, 20, K, n)
	eval_flow_at_dim(TanhFlow, tanh_flow, 100, K, n)
	eval_flow_at_dim(TanhFlow, tanh_flow, 1000, K, n)
	print('Tanh flows passed.')
	return None
"""


if __name__ == "__main__":

	test_get_flow_class()
	test_get_num_flow_params()
	test_flow_param_initialization()
	#test_affine_flows()
	test_elem_mult_flows()
	test_planar_flows()
	test_shift_flows()
	test_softplus_flows()

