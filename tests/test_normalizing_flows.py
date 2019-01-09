import tensorflow as tf
import numpy as np
from tf_util.normalizing_flows import PlanarFlowLayer, ElemMultLayer, get_flow_num_params
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

dtype = tf.float64

# write the ground truth functions for the normalizing flows

# planar flows
def planar_flow(z, params):
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_flow_num_params(PlanarFlowLayer, D))

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

# elementwise multiplication flows
def elem_mult_flow(z, params):
	D = z.shape[0]
	num_params = params.shape[0]
	assert(num_params == get_flow_num_params(ElemMultLayer, D))

	a = params

	# compute output
	out = np.multiply(a, z)

	# compute log abs det jacobian
	log_det_jac = np.sum(np.log(np.abs(a)))

	return out, log_det_jac


def eval_flow_at_dim(flow_class, true_flow, dim, K, n):
	num_params = get_flow_num_params(flow_class, dim)

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

	layer1 = flow_class(params1, inputs1)
	out1, log_det_jac1 = layer1.forward_and_jacobian()

	feed_dict = {params1:_params, inputs1:_inputs}
	with tf.Session() as sess:
		_out1, _log_det_jac1 = sess.run([out1, log_det_jac1], feed_dict)

		if (layer1.name == 'PlanarFlow'):
			wdotus = wdotu = tf.matmul(tf.expand_dims(layer1.w, 1), tf.expand_dims(layer1.u, 2))
			_wdotus = sess.run(wdotus, feed_dict)


	avg_out_error = np.sum(np.square(_out1 - out_true)) / (K*n)
	avg_log_det_jac_error = np.sum(np.square(_log_det_jac1 - log_det_jac_true)) / (K*n)
	assert(avg_out_error < 1e-16)
	assert(avg_log_det_jac_error < 1e-16)

	if (layer1.name == 'PlanarFlow'):
		inv_tol = 1e-10
		num_inv_viols = np.sum(_wdotus < -(1+inv_tol))
		assert(num_inv_viols == 0)

	#print(layer1.name + ' passed at dim=%d.' % dim)

def test_planar_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	np.random.seed(0)
	eval_flow_at_dim(PlanarFlowLayer, planar_flow, 1, K, n)
	eval_flow_at_dim(PlanarFlowLayer, planar_flow, 2, K, n)
	eval_flow_at_dim(PlanarFlowLayer, planar_flow, 4, K, n)
	eval_flow_at_dim(PlanarFlowLayer, planar_flow, 20, K, n)
	eval_flow_at_dim(PlanarFlowLayer, planar_flow, 100, K, n)
	eval_flow_at_dim(PlanarFlowLayer, planar_flow, 1000, K, n)
	#eval_flow_at_dim(PlanarFlowLayer, planar_flow, 100000, K, n) # passes but takes long
	print('Planar flows passed testing.')
	return None

def test_elem_mult_flows():
	# num parameterizations
	K = 20
	# number of inputs tested per parameterization
	n = 100

	eval_flow_at_dim(ElemMultLayer, elem_mult_flow, 1, K, n)
	eval_flow_at_dim(ElemMultLayer, elem_mult_flow, 2, K, n)
	eval_flow_at_dim(ElemMultLayer, elem_mult_flow, 4, K, n)
	eval_flow_at_dim(ElemMultLayer, elem_mult_flow, 20, K, n)
	eval_flow_at_dim(ElemMultLayer, elem_mult_flow, 100, K, n)
	eval_flow_at_dim(ElemMultLayer, elem_mult_flow, 1000, K, n)
	#eval_flow_at_dim(ElemMultLayer, elem_mult_flow, 100000, K, n) # passes but takes long
	print('Elementwise multiplication flows passed testing.')
	return None


if __name__ == "__main__":
	test_planar_flows()
	test_elem_mult_flows()
