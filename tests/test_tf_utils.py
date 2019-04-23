import tensorflow as tf
import numpy as np
from tf_util.stat_util import approx_equal
from tf_util.tf_util import AL_cost, log_grads, max_barrier, min_barrier, quartic_roots

DTYPE = tf.float64
EPS = 1e-16

n = 100


def aug_lag_cost(log_q_z, T_x_mu_centered, Lambda, c, all_params, num_suff_stats, entropy=True, I_x = None):
    num_params = len(all_params)

    neg_H = tf.reduce_mean(log_q_z)
    dneg_H_dtheta = tf.gradients(neg_H, all_params)
    T_x_mean = tf.reduce_mean(T_x_mu_centered, axis=[0, 1])
    T_x_mean_1 = tf.reduce_mean(T_x_mu_centered[:, : (n // 2), :], axis=[0, 1])
    T_x_mean_2 = tf.reduce_mean(T_x_mu_centered[:, (n // 2) :, :], axis=[0, 1])

    lambda_term = tf.tensordot(Lambda, T_x_mean, [[0], [0]])
    dlambdaterm_dtheta = tf.gradients(lambda_term, all_params)

    T_x_mean_1_grads = []
    for i in range(num_suff_stats):
        T_x_mean_1_grads.append(tf.gradients(T_x_mean_1[i], all_params))

    dcterm_dtheta = []
    for i in range(num_params):
        dcterm_dtheta_i = 0.0
        for j in range(num_suff_stats):
            dcterm_dtheta_i += T_x_mean_1_grads[j][i] * T_x_mean_2[j]
        dcterm_dtheta.append(c * dcterm_dtheta_i)

    if (I_x is not None):
        dIterm_dtheta = tf.gradients(tf.reduce_mean(I_x, [0,1]), all_params)


    grads = []
    for i in range(num_params):
        sum_grad = dlambdaterm_dtheta[i] + dcterm_dtheta[i]
        if entropy:
            sum_grad += dneg_H_dtheta[i] 
        if (I_x is not None):
            sum_grad += dIterm_dtheta[i] 
        grads.append(sum_grad)

    return 0.0, grads, -neg_H

def _max_barrier(u, alpha, t):
    return -(1.0/t)*np.log(-u + alpha)

def _min_barrier(u, alpha, t):
    return -(1.0/t)*np.log(u - alpha)

def test_AL_cost():
    D = 10
    num_rand_draws = 10

    W = tf.placeholder(dtype=DTYPE, shape=(1, n, D))
    A1 = tf.get_variable("A1", shape=(1,D,D), dtype=DTYPE)
    b1 = tf.get_variable("b1", shape=(1,D,1), dtype=DTYPE)
    all_params1 = tf.trainable_variables()
    A2 = tf.get_variable("A2", shape=(1,D,D), dtype=DTYPE)
    b2 = tf.get_variable("b2", shape=(1,D,1), dtype=DTYPE)
    all_params2 = tf.trainable_variables()

    all_params = tf.trainable_variables()

    z1 = tf.matmul(A1, tf.transpose(W, [0, 2, 1])) + b1
    z2 = tf.matmul(A2, z1) + b2

    # Not the actual log_q_z, just an arbitrary function for testing.
    log_q_z1 = tf.log(tf.linalg.det(A1))*np.ones((1,n)) + tf.reduce_sum(b1)
    log_q_z2 = tf.log(tf.linalg.det(A2))*np.ones((1,n)) + tf.reduce_sum(b2) + log_q_z1

    mu1 = np.random.normal(0.0, 1.0, (1, 1, 2*D))
    z1 = tf.transpose(z1, [0, 2, 1]) # [1, n, |T(x)|]
    T_x1 = tf.concat((z1, tf.square(z1)), axis=2)
    T_x_mu_centered1 = T_x1 - mu1
    t = 1e2
    I_x1 = tf.stack((min_barrier(T_x1[:,:,0], -1e6, t), max_barrier(T_x1[:,:,1], 1e6, t)), axis=2)

    mu2 = np.random.normal(0.0, 1.0, (1, 1, 2*D))
    z2 = tf.transpose(z2, [0, 2, 1]) # [1, n, |T(x)|]
    T_x2 = tf.concat((z2, tf.square(z2)), axis=2)
    T_x_mu_centered2 = T_x2 - mu2
    I_x2 = tf.stack((min_barrier(T_x2[:,:,0], -1e6, t), max_barrier(T_x2[:,:,1], 1e6, t)), axis=2)

    Lambda = tf.placeholder(dtype=DTYPE, shape=(2*D,))
    c = tf.placeholder(dtype=DTYPE, shape=())

    log_q_zs = [log_q_z1, log_q_z2]
    T_x_mu_centereds = [T_x_mu_centered1, T_x_mu_centered2]
    all_params_list = [all_params1, all_params2]
    I_xs = [I_x1, I_x2]
    for i in range(len(log_q_zs)):
        log_q_z = log_q_zs[i]
        T_x_mu_centered = T_x_mu_centereds[i]
        all_params = all_params_list[i]
        I_x = I_xs[i]
        for entropy in [True, False]:
            for _I_x in [None, I_x]:
                costs_true, grads_true, H_true = aug_lag_cost(
                    log_q_z, T_x_mu_centered, Lambda, c, all_params, 2*D, entropy=entropy, I_x=_I_x
                )

                costs, grads, H = AL_cost(log_q_z, T_x_mu_centered, Lambda, c, all_params, entropy=entropy, I_x=_I_x)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    for j in range(num_rand_draws):
                        _lambda = np.random.normal(0.0, 1.0, (2*D,))
                        _c = np.random.normal(0.0, 1.0)
                        _W = np.random.normal(0.0, 1.0, (1, n, D))
                        _grads_true, _grads = sess.run(
                            [grads_true, grads], {W: _W, Lambda: _lambda, c: _c}
                        )
                        _H_true, _H = sess.run(
                            [H_true, H], {W: _W, Lambda: _lambda, c: _c}
                        )

                    for k in range(len(all_params)):
                        assert approx_equal(_grads_true[k], _grads[k], EPS)


    return None


def test_max_barrier():
    M = 1000
    u = tf.placeholder(dtype=tf.float64, shape=(1, M))
    alphas = [-1e10, 0.0, 1.0, 1e10]
    ts = [1e-10, 1.0, 1e10, 1e20]
    num_alpha = len(alphas)
    num_ts = len(ts)
    for i in range(num_alpha):
        alpha = alphas[i]
        _u = np.random.uniform(-1e20, alpha, (1,M))
        _u[0,-1] = alpha-(1.0e-4)
        for j in range(num_ts):
            t = ts[j]
            I_x = max_barrier(u, alpha, t)
            I_x_true = _max_barrier(_u, alpha, t)
            with tf.Session() as sess:
                _I_x = sess.run(I_x, {u:_u})
            assert(approx_equal(_I_x, I_x_true, EPS))
    return None

def test_min_barrier():
    M = 1000
    u = tf.placeholder(dtype=tf.float64, shape=(1, M))
    alphas = [-1e10, 0.0, 1.0, 1e10]
    ts = [1e-10, 1.0, 1e10, 1e20]
    num_alpha = len(alphas)
    num_ts = len(ts)
    for i in range(num_alpha):
        alpha = alphas[i]
        _u = np.random.uniform(alpha, 1.0e20, (1,M))
        _u[0,-1] = alpha+(1.0e-4)
        for j in range(num_ts):
            t = ts[j]
            I_x = min_barrier(u, alpha, t)
            I_x_true = _min_barrier(_u, alpha, t)
            with tf.Session() as sess:
                _I_x = sess.run(I_x, {u:_u})
            assert(approx_equal(_I_x, I_x_true, EPS))
    return None

def unroll_params(param_list):
    num_params = len(param_list)
    num_dims = 0
    dims = []
    for i in range(num_params):
        dims_i = int(np.prod(param_list[i].shape))
        dims.append(dims_i)
        num_dims += dims_i
    param_array = np.zeros((num_dims,))
    ind = 0
    for i in range(num_params):
        dims_i = dims[i]
        param_i = param_list[i]
        param_array[ind : (ind + dims_i)] = np.reshape(param_i, (dims_i,))
        ind += dims_i
    return param_array


LG_EPS = 1e-50


def test_log_grads():
    array_len = 1000
    num_vars1 = 1
    num_vars2 = 20
    cur_ind = 113

    cost_grads1 = np.zeros((array_len, num_vars1))
    cost_grads2 = np.zeros((array_len, num_vars2))

    cost_grads1[:cur_ind, :] = np.random.normal(2.0, 1.0, (cur_ind, num_vars1))
    cost_grads2[:cur_ind, :] = np.random.normal(2.0, 1.0, (cur_ind, num_vars2))

    new_grads1 = [np.array([42.0])]
    cost_grads1_1_true = cost_grads1.copy()
    cost_grads1_1_true[cur_ind, 0] = 42.0

    new_grads2_1 = [np.arange(20)]
    cost_grads2_1_true = cost_grads2.copy()
    cost_grads2_1_true[cur_ind, :] = np.arange(20)

    new_grads2_2 = [
        np.random.normal(0.0, 1.0, (2, 4)),
        np.random.normal(4.0, 1.0, (4, 3)),
    ]
    cost_grads2_2_true = cost_grads2_1_true.copy()
    cost_grads2_2_true[cur_ind + 1, :] = unroll_params(new_grads2_2)

    log_grads(new_grads1, cost_grads1, cur_ind)
    assert approx_equal(cost_grads1, cost_grads1_1_true, LG_EPS)

    log_grads(new_grads2_1, cost_grads2, cur_ind)
    assert approx_equal(cost_grads2, cost_grads2_1_true, LG_EPS)

    log_grads(new_grads2_2, cost_grads2, cur_ind + 1)
    assert approx_equal(cost_grads2, cost_grads2_2_true, LG_EPS)
    return None


def sort_quartic_roots(roots):
    """Sort roots by real part and order conjugates.

    Args:
        roots (np.array): Complex roots of quartic polynomial.

    Returns:
        sorted_roots (np.array): Sorted roots.
    """
    real_parts = np.real(roots)
    real_sort_inds = np.argsort(real_parts)
    roots = roots[real_sort_inds]
    real_parts = real_parts[real_sort_inds]

    # If all roots have the same real part order by imaginary component.
    if (real_parts[1] == real_parts[2]):
        # If 1st root also has the same real part.
        if (real_parts[0] == real_parts[1] and not(real_parts[2] == real_parts[3])):
            imag_parts = np.imag(roots[:3])
            imag_sort_inds = np.argsort(imag_parts)
            roots = np.concatenate((roots[imag_sort_inds], roots[3]), axis=0)

        # If 4th root also has the same real part.
        elif (not(real_parts[0] == real_parts[1]) and real_parts[2] == real_parts[3]):
            imag_parts = np.imag(roots[1:])
            imag_sort_inds = np.argsort(imag_parts)
            roots = np.concatenate((roots[0], roots[imag_sort_inds]), axis=0)
        
        # If all roots have the same real part.
        elif (not(real_parts[0] == real_parts[1]) and real_parts[2] == real_parts[3]):
            imag_parts = np.imag(roots)
            imag_sort_inds = np.argsort(imag_parts)
            roots = roots[imag_sort_inds]
    
        # If just the middle two have the same real part.
        else:
            if (np.imag(roots[1]) > np.imag(roots[2])):
                temp = roots[1]
                roots[1] = roots[2]
                roots[2] = temp

    else:
        # If 1st and 2nd root are complex conjuate pairs, order by imaginary part.
        if (real_parts[0] == real_parts[1]):
            if (np.imag(roots[0]) > np.imag(roots[1])):
                temp = roots[0]
                roots[0] = roots[1]
                roots[1] = temp

        # If 3rd and 4th root are complex conjuate pairs, order by imaginary part.
        if (real_parts[2] == real_parts[3]):
            if (np.imag(roots[2]) > np.imag(roots[3])):
                temp = roots[2]
                roots[2] = roots[3]
                roots[3] = temp

    return roots

def test_sort_quartic_roots():
    roots = np.array([[2.0, 1.0, 4.0, 3.0],
                      [1.0 + 1j, 1.0 - 1j, 3.0, 4.0],
                      [1.0 + 1j, 1.0 - 1j, -1.0 - 2j, -1.0 + 2j],
                      [1.0 + 1j, 1.0 - 2j, 1.0 - 1j, 1.0 + 2j]])
    sorted_roots = np.array([[1.0, 2.0, 3.0, 4.0],
                      [1.0 - 1j, 1.0 + 1j, 3.0, 4.0],
                      [-1.0 - 2j, -1.0 + 2j, 1.0 - 1j, 1.0 + 1j],
                      [1.0 - 2j, 1.0 - 1j, 1.0 + 1j, 1.0 + 2j]])

    num_roots = roots.shape[0]
    for i in range(num_roots):
        assert(approx_equal(sorted_roots[i], sort_quartic_roots(roots[i]), EPS))
    return None

    
def test_quartic_formula():
    M = 100
    a = tf.placeholder(dtype=DTYPE, shape=(M,1))
    b = tf.placeholder(dtype=DTYPE, shape=(M,1))
    c = tf.placeholder(dtype=DTYPE, shape=(M,1))
    d = tf.placeholder(dtype=DTYPE, shape=(M,1))
    e = tf.placeholder(dtype=DTYPE, shape=(M,1))

    _a = np.random.normal(0.0, 100.0, (M,1))
    _b = np.random.normal(0.0, 100.0, (M,1))
    _c = np.random.normal(0.0, 100.0, (M,1))
    _d = np.random.normal(0.0, 100.0, (M,1))
    _e = np.random.normal(0.0, 100.0, (M,1))

    roots = quartic_roots(a, b, c, d, e)
    with tf.Session() as sess:
        _roots = sess.run(roots, {a:_a, b:_b, c:_c, d:_d, e:_e})
    _roots = np.concatenate(_roots, axis=1)

    for i in range(M):
        p = np.array([_a[i,0], _b[i,0], _c[i,0], _d[i,0], _e[i,0]])
        roots_np = np.roots(p)
        sorted_roots_np = sort_quartic_roots(roots_np)
        sorted_roots_tf = sort_quartic_roots(_roots[i])
        assert(approx_equal(sorted_roots_np, sorted_roots_tf, EPS))
    return None




if __name__ == "__main__":
    np.random.seed(0)
    test_AL_cost()
    test_max_barrier()
    test_min_barrier()
    test_log_grads()
    test_sort_quartic_roots()
    test_quartic_formula()


