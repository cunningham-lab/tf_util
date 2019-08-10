import tensorflow as tf
import numpy as np
from tf_util.stat_util import approx_equal
from tf_util.normalizing_flows import (
    AffineFlow,
    CholProdFlow,
    ElemMultFlow,
    ExpFlow,
    IntervalFlow,
    PlanarFlow,
    PermutationFlow,
    RadialFlow,
    ShiftFlow,
    SimplexBijectionFlow,
    SoftPlusFlow,
    StructuredSpinnerFlow,
    StructuredSpinnerTanhFlow,
    TanhFlow,
    RealNVP)

from tf_util.normalizing_flows import (
    get_num_flow_params,
    get_flow_out_dim,
    get_flow_param_inits,
    get_flow_class,
    get_real_nvp_num_params)

import os
from tf_util.normalizing_flows import get_real_nvp_mask, \
                            get_real_nvp_mask_list, \
                            get_real_nvp_num_params, \
                            nvp_neural_network_np


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

DTYPE = tf.float64
EPS = 1e-10

# write the ground truth functions for the normalizing flows

# Affine flows
def affine_flow(z, params):
    """Affine flow operation and log abs det jac.

	z = Az + b
	log_det_jac = log(|det(A)|)

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.

	"""
    D = z.shape[0]
    num_params = params.shape[0]
    assert num_params == get_num_flow_params(AffineFlow, D)

    A = np.reshape(params[: D ** 2], (D, D))
    b = params[D ** 2 :]

    # compute output
    out = np.dot(A, np.expand_dims(z, 1))[:, 0] + b

    # compute log abs det jacobian
    log_det_jac = np.log(np.abs(np.linalg.det(A)))

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
    assert num_params == get_num_flow_params(CholProdFlow, D)

    # I can't find the numpy implementation of the fill triangular method!!!

    raise NotImplementedError()


# Elementwise multiplication flows
def elem_mult_flow(z, params):
    """Elementwise multiplication flow operation and log abs det jac.

	z = a * z
    log_det_jac = \sum_i log(abs(a_i))

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
    D = z.shape[0]
    num_params = params.shape[0]
    assert num_params == get_num_flow_params(ElemMultFlow, D)

    a = params

    # compute output
    out = np.multiply(a, z)

    # compute log abs det jacobian
    log_det_jac = np.sum(np.log(np.abs(a)))

    return out, log_det_jac


# Elementwise multiplication flows
def exp_flow(z, params):
    """Exponential flow operation and log abs det jac.

	z = exp(z)
    log_det_jac = \sum_i z_i

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
    D = z.shape[0]
    num_params = params.shape[0]
    assert num_params == get_num_flow_params(ExpFlow, D)

    # compute output
    out = np.exp(z)

    # compute log abs det jacobian
    log_det_jac = np.sum(z)

    return out, log_det_jac


# Interval flows
def interval_flow(z, params, a, b):
    """Interval flow operation and log abs det jac.

	z = 

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
    D = z.shape[0]
    num_params = params.shape[0]
    assert num_params == get_num_flow_params(IntervalFlow, D)

    m = (b - a) / 2.0
    c = (a + b) / 2.0

    out = m * np.tanh(z) + c
    log_det_jac = np.sum(np.log(m) + np.log(1.0 - np.square(np.tanh(z))))

    return out, log_det_jac


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
    assert num_params == get_num_flow_params(PlanarFlow, D)

    _u = params[:D]
    w = params[D : (2 * D)]
    b = params[2 * D]

    # enforce w^\topu >= -1
    wdotu = np.dot(w, _u)
    m_wdotu = -1.0 + np.log(1.0 + np.exp(wdotu))
    u = _u + (m_wdotu - wdotu) * w / np.dot(w, w)

    # compute output
    out = z + u * np.tanh(np.dot(w, z) + b)

    # compute log det jacobian
    phi = (1.0 - np.square(np.tanh(np.dot(w, z) + b))) * w
    log_det_jac = np.log(np.abs(1.0 + np.dot(u, phi)))

    return out, log_det_jac

def permutation_flow(z, params):
    z_out = z[params]
    return z_out, 0.0


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
    assert num_params == get_num_flow_params(RadialFlow, D)

    alpha = np.exp(params[0])
    _beta = params[1]
    z0 = params[2:]

    # enforce invertibility
    m_beta = np.log(1.0 + np.exp(_beta))
    beta = -alpha + m_beta

    r = np.linalg.norm(z)
    h = 1.0 / (alpha + r)
    hprime = -1.0 / np.square(alpha + r)

    out = z + beta * h * (z - z0)
    log_det_jac = (D - 1.0) * np.log(1.0 + beta * h) + np.log(
        1.0 + beta * h + beta * hprime * r
    )

    return out, log_det_jac

def real_nvp(z, params, num_masks, nlayers, upl):
    """Real NVP operation and log abs det jac.

    The first num_masks masks of the following pattern are used.

    mask 1: [++++++++--------] (first D/2) f=1+
    mask 2: [--------++++++++] (last D/2)  f=1-
    mask 3: [+-+-+-+-+-+-+-+-] (every other) f=D/2+
    mask 4: [-+-+-+-+-+-+-+-+] (every other shift) f=D/2-
    mask 5: [++++----++++----] (first D/2) f=2+
    mask 6: [----++++----++++] (last D/2)  f=2-
    mask 7: [++--++--++--++--] (every other) f=D/4+
    mask 8: [--++--++--++--++] (every other shift) f=D/4-
    ...

    # Arguments
        z (np.array): [D,] Input vector.
        params (np.array): [num_param,] Total parameter vector
        num_masks (int): number of masking layers
        nlayers (int): number of neural network layers per mask
        upl (int): number of units per layer

    # Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
    """
    D = z.shape[0]
    num_params = params.shape[0]
    opt_params = {'num_masks':num_masks, 'nlayers':nlayers, 'upl':upl}
    assert num_params == get_num_flow_params(RealNVP, D, opt_params)

    # get list of masks
    masks = get_real_nvp_mask_list(D, num_masks)

    # construct functions for each mask
    param_ind = 0
    z_i = z
    sum_log_det_jac = 0.0
    for i in range(num_masks):
        mask_i = masks[i]

        s, param_ind = nvp_neural_network_np(z_i, params, mask_i, nlayers, upl, param_ind)
        t, param_ind = nvp_neural_network_np(z_i, params, mask_i, nlayers, upl, param_ind)

        z_i = (mask_i)*z_i + (1-mask_i)*(z_i*np.exp(s) + t)

        log_det_jac = np.sum((1-mask_i)*s)
        sum_log_det_jac += log_det_jac

    return z_i, sum_log_det_jac


    

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
    assert num_params == get_num_flow_params(ShiftFlow, D)

    b = params

    # compute output
    out = z + b

    # compute the log abs det jacobian
    log_det_jac = 0.0
    return out, log_det_jac


# Simplex bijection flows
def simplex_bijection_flow(z, params):
    """Simplex bijection flow operation and log abs det jac.

	out = (e^z1 / (sum i e^zi + 1), .., e^z_d-1 / (sum i e^zi + 1), 1 / (sum i e^zi + 1))
	log_det_jac = log(1 - (sum i e^zi / (sum i e^zi+1)) - D log(sum i e^zi + 1) + sum i zi

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
    D = z.shape[0]
    num_params = params.shape[0]
    assert num_params == get_num_flow_params(SimplexBijectionFlow, D)

    exp_z = np.exp(z)
    den = np.sum(exp_z) + 1

    out = np.concatenate((exp_z / den, np.array([1.0 / den])), axis=0)

    log_det_jac = (
        np.log(1 - (np.sum(exp_z) / den)) - D * np.log(np.sum(exp_z) + 1) + np.sum(z)
    )

    return out, log_det_jac


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
    assert num_params == get_num_flow_params(SoftPlusFlow, D)

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
    assert num_params == get_num_flow_params(StructuredSpinnerFlow, D)

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
    assert num_params == get_num_flow_params(StructuredSpinnerTanhFlow, D)

    raise NotImplementedError()


# Tanh flows
def tanh_flow(z, params):
    """Tanh flow operation and log abs det jac.

	z = tanh(z)
    log_det_jac = sum_i log(abs(1 - sec^2(z_i)))

	# Arguments
	    z (np.array): [D,] Input vector.
		params (np.array): [num_param,] Total parameter vector

	# Returns 
        out (np.array): [D,] Output of affine flow operation.
        log_det_jacobian (np.float): Log abs det jac.
        
	"""
    D = z.shape[0]
    num_params = params.shape[0]
    assert num_params == get_num_flow_params(TanhFlow, D)

    out = np.tanh(z)
    log_det_jac = np.sum(np.log(1 - np.square(np.tanh(z))))

    return out, log_det_jac


def eval_flow_at_dim(flow_class, true_flow, dim, K, n):
    if (flow_class == RealNVP):
        num_masks = 4
        nlayers = 4
        upl = 20
        opt_params = {'num_masks':num_masks, 'nlayers':nlayers, 'upl':upl}
        eps = 1e-5
    else:
        opt_params = {}
        eps = EPS
    num_params = get_num_flow_params(flow_class, dim, opt_params)
    out_dim = get_flow_out_dim(flow_class, dim)

    if (flow_class == PermutationFlow):
        params1 = np.random.choice(dim, dim, replace=False)
    else:
        params1 = tf.placeholder(dtype=DTYPE, shape=(None, num_params))
    inputs1 = tf.placeholder(dtype=DTYPE, shape=(None, None, dim))

    if (flow_class == RealNVP):
        flow1 = flow_class(params1, inputs1, num_masks, nlayers, upl)
    else:
        flow1 = flow_class(params1, inputs1)
    out1, log_det_jac1 = flow1.forward_and_jacobian()

    _params = np.random.normal(0.0, 1.0, (K, num_params))
    _inputs = np.random.normal(0.0, 1.0, (K, n, dim))

        # compute ground truth
    out_true = np.zeros((K, n, out_dim))
    log_det_jac_true = np.zeros((K, n))
    for k in range(K):
        if (flow_class == PermutationFlow):
            _params_k = params1
        else:
            _params_k = _params[k, :]
        for j in range(n):
            if (flow_class == RealNVP):
                out_true[k, j, :], log_det_jac_true[k, j] = true_flow(
                    _inputs[k, j, :], _params_k, num_masks, nlayers, upl,
                )
            else:
                out_true[k, j, :], log_det_jac_true[k, j] = true_flow(
                    _inputs[k, j, :], _params_k
                )

    if (flow_class == PermutationFlow):
        feed_dict = {inputs1: _inputs}
    else:
        feed_dict = {params1: _params, inputs1: _inputs}
    with tf.Session() as sess:
        _out1, _log_det_jac1 = sess.run([out1, log_det_jac1], feed_dict)

        # Ensure invertibility
        if flow1.name == "PlanarFlow":
            wdotus = tf.matmul(tf.expand_dims(flow1.w, 1), tf.expand_dims(flow1.u, 2))
            _wdotus = sess.run(wdotus, feed_dict)
            num_inv_viols = np.sum(_wdotus < -(1 + eps))
            assert num_inv_viols == 0

        # Ensure invertibility
        if flow1.name == "RadialFlow":
            alpha, beta = sess.run([flow1.alpha, flow1.beta], feed_dict)
            for k in range(K):
                assert -alpha[k, 0] <= beta[k, 0]

        # Should check known inverse
        if flow1.name in ["RealNVP", "ElemMultFlow", "PermutationFlow", "ShiftFlow", "SoftPlusFlow"]:
            print('testing inverse', flow1.name)
            f_inv_z = flow1.inverse(out1)
            _f_inv_z = sess.run(f_inv_z, feed_dict)
            if (flow1.name == "RealNVP"):
                # Machine precision issues force this to be a bit lax for param dist.
                assert(approx_equal(_f_inv_z, _inputs, 1e-2))
            else: 
                assert(approx_equal(_f_inv_z, _inputs, 1e-16))

    assert approx_equal(_out1, out_true, eps)
    assert approx_equal(_log_det_jac1, log_det_jac_true, eps)
        
    return None


def eval_interval_flow_at_dim(dim, K, n):
    num_params = get_num_flow_params(IntervalFlow, dim)
    out_dim = get_flow_out_dim(IntervalFlow, dim)

    params1 = tf.placeholder(dtype=DTYPE, shape=(None, num_params))
    inputs1 = tf.placeholder(dtype=DTYPE, shape=(None, None, dim))

    _params = np.random.normal(0.0, 1.0, (K, num_params))
    _inputs = np.random.normal(0.0, 1.0, (K, n, dim))

    _a = np.random.normal(0.0, 1.0, (K, dim))
    _b = _a + np.abs(np.random.normal(0.0, 1.0, (K, dim))) + 0.001

    # compute ground truth
    out_true = np.zeros((K, n, out_dim))
    log_det_jac_true = np.zeros((K, n))
    for k in range(K):
        _params_k = _params[k, :]
        _a_k = _a[k, :]
        _b_k = _b[k, :]
        for j in range(n):
            out_true[k, j, :], log_det_jac_true[k, j] = interval_flow(
                _inputs[k, j, :], _params_k, _a_k, _b_k
            )

    _out1 = np.zeros((K, n, out_dim))
    _f_inv_z1 = np.zeros((K, n, dim))
    _log_det_jac1 = np.zeros((K, n))
    with tf.Session() as sess:
        for k in range(K):
            _a_k = _a[k, :]
            _b_k = _b[k, :]
            flow1 = IntervalFlow(params1, inputs1, _a_k, _b_k)
            out1, log_det_jac1 = flow1.forward_and_jacobian()
            f_inv_z = flow1.inverse(out1)
            _params_k = np.expand_dims(_params[k, :], 0)
            _inputs_k = np.expand_dims(_inputs[k, :, :], 0)
            feed_dict = {params1: _params_k, inputs1: _inputs_k}
            _out1_k, _log_det_jac1_k, _f_inv_z = sess.run([out1, log_det_jac1, f_inv_z], feed_dict)
            _out1[k, :, :] = _out1_k[0]
            _f_inv_z1[k,:,:] = _f_inv_z[0]
            _log_det_jac1[k, :] = _log_det_jac1_k[0]

    assert approx_equal(_out1, out_true, EPS)
    assert approx_equal(_f_inv_z1, _inputs, EPS)
    assert approx_equal(_log_det_jac1, log_det_jac_true, EPS)
    return None


def test_get_flow_class():
    assert get_flow_class("AffineFlow") == AffineFlow
    assert get_flow_class("CholProdFlow") == CholProdFlow
    assert get_flow_class("ElemMultFlow") == ElemMultFlow
    assert get_flow_class("ExpFlow") == ExpFlow
    assert get_flow_class("IntervalFlow") == IntervalFlow
    assert get_flow_class("PlanarFlow") == PlanarFlow
    assert get_flow_class("RadialFlow") == RadialFlow
    assert get_flow_class("ShiftFlow") == ShiftFlow
    assert get_flow_class("SimplexBijectionFlow") == SimplexBijectionFlow
    assert get_flow_class("SoftPlusFlow") == SoftPlusFlow
    assert get_flow_class("StructuredSpinnerFlow") == StructuredSpinnerFlow
    assert get_flow_class("StructuredSpinnerTanhFlow") == StructuredSpinnerTanhFlow
    assert get_flow_class("TanhFlow") == TanhFlow
    assert get_flow_class("RealNVP") == RealNVP

    return None


def test_get_num_flow_params():
    assert get_num_flow_params(AffineFlow, 1) == 2
    assert get_num_flow_params(AffineFlow, 2) == 6
    assert get_num_flow_params(AffineFlow, 4) == 20
    assert get_num_flow_params(AffineFlow, 20) == 420
    assert get_num_flow_params(AffineFlow, 100) == 10100
    assert get_num_flow_params(AffineFlow, 1000) == 1001000

    assert(get_num_flow_params(CholProdFlow, 1) == 0)
    assert(get_num_flow_params(CholProdFlow, 2) == 0)
    assert(get_num_flow_params(CholProdFlow, 4) == 0)

    assert get_num_flow_params(ElemMultFlow, 1) == 1
    assert get_num_flow_params(ElemMultFlow, 2) == 2
    assert get_num_flow_params(ElemMultFlow, 4) == 4
    assert get_num_flow_params(ElemMultFlow, 20) == 20
    assert get_num_flow_params(ElemMultFlow, 100) == 100
    assert get_num_flow_params(ElemMultFlow, 1000) == 1000

    assert get_num_flow_params(IntervalFlow, 1) == 2
    assert get_num_flow_params(IntervalFlow, 2) == 2
    assert get_num_flow_params(IntervalFlow, 4) == 2
    assert get_num_flow_params(IntervalFlow, 20) == 2
    assert get_num_flow_params(IntervalFlow, 100) == 2
    assert get_num_flow_params(IntervalFlow, 1000) == 2

    assert get_num_flow_params(PlanarFlow, 1) == 3
    assert get_num_flow_params(PlanarFlow, 2) == 5
    assert get_num_flow_params(PlanarFlow, 4) == 9
    assert get_num_flow_params(PlanarFlow, 20) == 41
    assert get_num_flow_params(PlanarFlow, 100) == 201
    assert get_num_flow_params(PlanarFlow, 1000) == 2001

    assert get_num_flow_params(RadialFlow, 1) == 3
    assert get_num_flow_params(RadialFlow, 2) == 4
    assert get_num_flow_params(RadialFlow, 4) == 6
    assert get_num_flow_params(RadialFlow, 20) == 22
    assert get_num_flow_params(RadialFlow, 100) == 102
    assert get_num_flow_params(RadialFlow, 1000) == 1002

    assert get_num_flow_params(ShiftFlow, 1) == 1
    assert get_num_flow_params(ShiftFlow, 2) == 2
    assert get_num_flow_params(ShiftFlow, 4) == 4
    assert get_num_flow_params(ShiftFlow, 20) == 20
    assert get_num_flow_params(ShiftFlow, 100) == 100
    assert get_num_flow_params(ShiftFlow, 1000) == 1000

    assert get_num_flow_params(SimplexBijectionFlow, 1) == 0
    assert get_num_flow_params(SimplexBijectionFlow, 2) == 0
    assert get_num_flow_params(SimplexBijectionFlow, 4) == 0
    assert get_num_flow_params(SimplexBijectionFlow, 20) == 0
    assert get_num_flow_params(SimplexBijectionFlow, 100) == 0
    assert get_num_flow_params(SimplexBijectionFlow, 1000) == 0

    assert get_num_flow_params(SoftPlusFlow, 1) == 0
    assert get_num_flow_params(SoftPlusFlow, 2) == 0
    assert get_num_flow_params(SoftPlusFlow, 4) == 0
    assert get_num_flow_params(SoftPlusFlow, 20) == 0
    assert get_num_flow_params(SoftPlusFlow, 100) == 0
    assert get_num_flow_params(SoftPlusFlow, 1000) == 0

    assert get_num_flow_params(TanhFlow, 1) == 0
    assert get_num_flow_params(TanhFlow, 2) == 0
    assert get_num_flow_params(TanhFlow, 4) == 0
    assert get_num_flow_params(TanhFlow, 20) == 0
    assert get_num_flow_params(TanhFlow, 100) == 0
    assert get_num_flow_params(TanhFlow, 1000) == 0

    # num_masks,nlayers,upl==1,1,1
    assert get_num_flow_params(RealNVP, 1) == 8
    assert get_num_flow_params(RealNVP, 2) == 14
    assert get_num_flow_params(RealNVP, 4) == 26
    assert get_num_flow_params(RealNVP, 20) == 122
    assert get_num_flow_params(RealNVP, 100) == 602
    assert get_num_flow_params(RealNVP, 1000) == 6002

    return None


def test_flow_param_initialization():
    Ds = [1, 2, 4, 20, 100, 1000]
    all_glorot_uniform_flows = [AffineFlow, ElemMultFlow, ShiftFlow, RealNVP]
    all_no_param_flows = [ExpFlow, SimplexBijectionFlow, SoftPlusFlow, TanhFlow]
    with tf.Session() as sess:
        for D in Ds:
            for flow in all_glorot_uniform_flows:
                if (flow == RealNVP):
                    opt_params = {'num_masks':1, 'nlayers':1, 'upl':1}
                    inits, dims = get_flow_param_inits(flow, D, opt_params)
                    assert sum(dims) == get_num_flow_params(flow, D, opt_params)
                else:
                    inits, dims = get_flow_param_inits(flow, D)
                    assert sum(dims) == get_num_flow_params(flow, D)
                assert len(inits) == 1
                # assert(isinstance(inits[0], tf.glorot_uniform_initializer))

            for flow in all_no_param_flows:
                inits, dims = get_flow_param_inits(flow, D)
                assert len(inits) == 1
                assert inits[0] == None
                assert len(dims) == 1
                assert dims[0] == 0

            """
			inits, dims = get_flow_param_inits(CholProdFlow, D)
			"""

            inits, dims = get_flow_param_inits(PlanarFlow, D)
            assert approx_equal(sess.run(inits[0]), np.zeros(D), EPS)
            # assert(isinstance(inits[1], tf.glorot_uniform_initializer))
            assert approx_equal(sess.run(inits[2]), 0.0, EPS)
            assert dims == [D, D, 1]

            """
			inits, dims = get_flow_param_inits(RadialFlow, D)
			"""

            """
			inits, dims = get_flow_param_inits(StructuredSpinnerFlow, D)
			"""

            """
			inits, dims = get_flow_param_inits(StructuredSpinnerTanhFlow, D)
			"""

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
    return None


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
    return None


def test_exp_flows():
    # num parameterizations
    K = 1
    # number of inputs tested per parameterization
    n = 100

    eval_flow_at_dim(ExpFlow, exp_flow, 1, K, n)
    eval_flow_at_dim(ExpFlow, exp_flow, 2, K, n)
    eval_flow_at_dim(ExpFlow, exp_flow, 4, K, n)
    eval_flow_at_dim(ExpFlow, exp_flow, 20, K, n)
    eval_flow_at_dim(ExpFlow, exp_flow, 100, K, n)
    eval_flow_at_dim(ExpFlow, exp_flow, 1000, K, n)
    return None


def test_interval_flows():
    # num parameterizations
    K = 20
    # number of inputs tested per parameterization
    n = 100

    eval_interval_flow_at_dim(1, K, n)
    eval_interval_flow_at_dim(2, K, n)
    eval_interval_flow_at_dim(4, K, n)
    eval_interval_flow_at_dim(20, K, n)
    eval_interval_flow_at_dim(100, K, n)
    eval_interval_flow_at_dim(1000, K, n)
    return None


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
    return None

def test_permutation_flows():
    # num parameterizations
    K = 20
    # number of inputs tested per parameterization
    n = 100

    np.random.seed(0)
    eval_flow_at_dim(PermutationFlow, permutation_flow, 2, K, n)
    eval_flow_at_dim(PermutationFlow, permutation_flow, 4, K, n)
    eval_flow_at_dim(PermutationFlow, permutation_flow, 20, K, n)
    eval_flow_at_dim(PermutationFlow, permutation_flow, 100, K, n)
    return None

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
    return None


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
    return None


def test_simplex_bijection_flows():
    # num parameterizations
    K = 1
    # number of inputs tested per parameterization
    n = 100

    eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 1, K, n)
    eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 2, K, n)
    eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 4, K, n)
    eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 20, K, n)
    eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 100, K, n)
    eval_flow_at_dim(SimplexBijectionFlow, simplex_bijection_flow, 1000, K, n)
    return None


def test_softplus_flows():
    # num parameterizations
    K = 1
    # number of inputs tested per parameterization
    n = 100

    eval_flow_at_dim(SoftPlusFlow, softplus_flow, 1, K, n)
    eval_flow_at_dim(SoftPlusFlow, softplus_flow, 2, K, n)
    eval_flow_at_dim(SoftPlusFlow, softplus_flow, 4, K, n)
    eval_flow_at_dim(SoftPlusFlow, softplus_flow, 20, K, n)
    eval_flow_at_dim(SoftPlusFlow, softplus_flow, 100, K, n)
    eval_flow_at_dim(SoftPlusFlow, softplus_flow, 1000, K, n)
    return None


def test_tanh_flows():
    # num parameterizations
    K = 1
    # number of inputs tested per parameterization
    n = 100

    eval_flow_at_dim(TanhFlow, tanh_flow, 1, K, n)
    eval_flow_at_dim(TanhFlow, tanh_flow, 2, K, n)
    eval_flow_at_dim(TanhFlow, tanh_flow, 4, K, n)
    eval_flow_at_dim(TanhFlow, tanh_flow, 20, K, n)
    eval_flow_at_dim(TanhFlow, tanh_flow, 100, K, n)
    eval_flow_at_dim(TanhFlow, tanh_flow, 1000, K, n)
    return None

def test_real_nvp():
    np.random.seed(0)
    # num parameterizations
    K = 1
    # number of inputs tested per parameterization
    n = 100
    eval_flow_at_dim(RealNVP, real_nvp, 8, K, n)
    eval_flow_at_dim(RealNVP, real_nvp, 50, K, n)
    return None

def test_get_real_nvp_mask():
    Ds = [8, 8, 8, 8, 8, 8, 8, 8, \
          2, 2, \
          3, 3, \
          17, 17]
    fs = [1, 1, 2, 2, 3, 3, 4, 4, \
          1, 1, \
          1, 1, \
          1, 8]
    firstOns = [True, False, True, False, True, False, True, False, \
                True, False, \
                True, False, \
                True, True]
    true_masks = [np.array([1, 1 ,1, 1, 0, 0, 0, 0]),
                  np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                  np.array([1, 1, 0, 0, 1, 1, 0, 0]),
                  np.array([0, 0, 1, 1, 0, 0, 1, 1]),
                  np.array([1, 0, 1, 0, 1, 0, 1, 0]),
                  np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                  np.array([1, 0, 1, 0, 1, 0, 1, 0]),
                  np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                  np.array([1, 0]),
                  np.array([0, 1]),
                  np.array([1, 1, 0]),
                  np.array([0, 0, 1]),
                  np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                  np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])]
    
    num_tests = len(Ds)
    for i in range(num_tests):
        D = Ds[i]
        f = fs[i]
        firstOn = firstOns[i]
        mask = get_real_nvp_mask(D, f, firstOn)
        true_mask = true_masks[i]
        assert(approx_equal(mask, true_mask, EPS))

    return None

def test_get_real_nvp_mask_list():
    mask_list = get_real_nvp_mask_list(5, 4)
    mask_list_true = [np.array([1, 1 ,1, 0, 0]),
                      np.array([0, 0, 0, 1, 1]),
                      np.array([1, 0, 1, 0, 1]),
                      np.array([0, 1, 0, 1, 0])]
    for i in range(4):
        approx_equal(mask_list[i], mask_list_true[i], EPS)


    mask_list = get_real_nvp_mask_list(8, 6)
    mask_list_true = [np.array([1, 1 ,1, 1, 0, 0, 0, 0]),
                      np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                      np.array([1, 0, 1, 0, 1, 0, 1, 0]),
                      np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                      np.array([1, 1, 0, 0, 1, 1, 0, 0]),
                      np.array([0, 0, 1, 1, 0, 0, 1, 1])]
    for i in range(6):
        approx_equal(mask_list[i], mask_list_true[i], EPS)


    mask_list = get_real_nvp_mask_list(16, 8)
    mask_list_true = [np.array([1, 1 ,1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                      np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
                      np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
                      np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                      np.array([1, 1 ,1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
                      np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]),
                      np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
                      np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])]
    for i in range(8):
        approx_equal(mask_list[i], mask_list_true[i], EPS)

    return None

def test_get_real_nvp_num_params():
    np.random.seed(0)
    Ds = [2, 2, 10, 10, 50]
    num_masks = [1, 4, 2, 4, 4]
    nlayers = [1, 2, 1, 4, 4]
    upls = [10, 100, 10, 100, 100]
    num_params_true = [104, 84816, 880, 259280, 323600]
    for i in range(len(Ds)):
        num_params_i = get_real_nvp_num_params(Ds[i], num_masks[i], nlayers[i], upls[i])
        assert(num_params_i == num_params_true[i])

    return None

if __name__ == "__main__":
    test_get_flow_class()
    test_get_num_flow_params()
    test_flow_param_initialization()

    test_affine_flows()
    test_elem_mult_flows()
    test_exp_flows()
    test_interval_flows()
    test_planar_flows()
    test_permutation_flows()
    test_radial_flows()
    test_shift_flows()
    test_simplex_bijection_flows()
    test_softplus_flows()
    test_tanh_flows()

    test_get_real_nvp_mask()
    test_get_real_nvp_mask_list()
    test_get_real_nvp_num_params()
    test_real_nvp()
