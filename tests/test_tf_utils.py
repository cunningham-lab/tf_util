import tensorflow as tf
import numpy as np
from tf_util.stat_util import approx_equal
from tf_util.tf_util import AL_cost, log_grads

DTYPE = tf.float64
EPS = 1e-16

n = 100


def aug_lag_cost(log_q_z, T_x_mu_centered, Lambda, c, all_params, num_suff_stats):
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

    grads = []
    for i in range(num_params):
        sum_grad = dneg_H_dtheta[i] + dlambdaterm_dtheta[i] + dcterm_dtheta[i]
        grads.append(sum_grad)

    return 0.0, grads, -neg_H


def test_AL_cost():
    D = 3
    W = tf.placeholder(dtype=DTYPE, shape=(1, n, D))
    a = tf.get_variable("a", shape=(D,), dtype=DTYPE)
    b = tf.get_variable("b", shape=(1,), dtype=DTYPE)

    all_params = tf.trainable_variables()

    log_q_z = tf.tensordot(W, a, [[2], [0]]) + b[0]
    H = -tf.reduce_mean(log_q_z)

    mu = np.array([1.0, 2.0, 3.0])
    T_x = W + tf.expand_dims(tf.expand_dims(a, 0), 0) - b
    T_x_mu_centered = T_x - np.expand_dims(np.expand_dims(mu, 0), 0)

    grad_H_a = tf.gradients(H, a)

    Lambda = tf.placeholder(dtype=DTYPE, shape=(D,))
    c = tf.placeholder(dtype=DTYPE, shape=())

    costs_true, grads_true, H_true = aug_lag_cost(
        log_q_z, T_x_mu_centered, Lambda, c, all_params, 3
    )

    costs, grads, H = AL_cost(log_q_z, T_x_mu_centered, Lambda, c, all_params)

    _lambda = np.zeros((3,))
    # _lambda = np.array([0.2, -0.1, 0.5])
    _c = 1.0
    _W = np.random.normal(0.0, 1.0, (1, n, D))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _grads_true, _grads = sess.run(
            [grads_true, grads], {W: _W, Lambda: _lambda, c: _c}
        )

    for i in range(len(all_params)):
        assert approx_equal(_grads_true[i], _grads[i], EPS)
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


if __name__ == "__main__":
    np.random.seed(0)
    test_AL_cost()
    test_log_grads()
