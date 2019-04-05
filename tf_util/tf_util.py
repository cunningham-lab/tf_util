# Copyright 2018 Sean Bittner, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import tensorflow as tf
import numpy as np

from tf_util.normalizing_flows import (
    PlanarFlow,
    ShiftFlow,
    ElemMultFlow,
    get_flow_class,
    get_density_network_inits,
    RealNVP,
)

DTYPE = tf.float64


def density_network(W, arch_dict, support_mapping=None, initdir=None):
    D = arch_dict["D"]
    if initdir is None:
        inits_by_layer, dims_by_layer = get_density_network_inits(arch_dict)
    else:
        inits_by_layer, dims_by_layer = load_nf_init(initdir, arch_dict)
        print("Loaded optimized initialization.")

    num_layers = len(inits_by_layer)
    # declare layer parameters with initializations
    params = []
    with tf.variable_scope("density_network"):
        for i in range(num_layers):
            params_i = []
            inits_i = inits_by_layer[i]
            dims_i = dims_by_layer[i]
            num_inits = len(inits_i)
            for j in range(num_inits):
                varname_ij = "theta_%d_%d" % (i + 1, j + 1)
                if isinstance(inits_i[j], tf.Tensor):
                    var_ij = tf.get_variable(
                        varname_ij, dtype=DTYPE, initializer=inits_i[j]
                    )
                else:
                    var_ij = tf.get_variable(
                        varname_ij,
                        shape=(dims_i[j],),
                        dtype=DTYPE,
                        initializer=inits_i[j],
                    )
                params_i.append(tf.expand_dims(var_ij, 0))
            params.append(tf.concat(params_i, 1))

    Z = W
    flow_layers = []
    sum_log_det_jacobians = 0.0
    ind = 0
    if arch_dict["mult_and_shift"] == "pre":
        flow_layer = ElemMultFlow(params[ind], Z)
        Z, log_det_jacobian = flow_layer.forward_and_jacobian()
        sum_log_det_jacobians += log_det_jacobian
        flow_layers.append(flow_layer)
        ind += 1

        flow_layer = ShiftFlow(params[ind], Z)
        Z, log_det_jacobian = flow_layer.forward_and_jacobian()
        sum_log_det_jacobians += log_det_jacobian
        flow_layers.append(flow_layer)
        ind += 1

    flow_class = get_flow_class(arch_dict["TIF_flow_type"])
    for i in range(arch_dict["repeats"]):
        if (flow_class == PlanarFlow):
            flow_layer = flow_class(params[ind], Z)
            Z, log_det_jacobian = flow_layer.forward_and_jacobian()
        elif (flow_class == RealNVP):
            real_nvp_arch = arch_dict['real_nvp_arch']
            num_masks = real_nvp_arch['num_masks']
            real_nvp_layers = real_nvp_arch['nlayers']
            upl = real_nvp_arch['upl']
            flow_layer = flow_class(params[ind], Z, num_masks, real_nvp_layers, upl)
            Z, log_det_jacobian = flow_layer.forward_and_jacobian()
        else:
            print('uh oh')
            raise NotImplementedError()
        sum_log_det_jacobians += log_det_jacobian
        flow_layers.append(flow_layer)
        ind += 1

    if arch_dict["mult_and_shift"] == "post":
        flow_layer = ElemMultFlow(params[ind], Z)
        Z, log_det_jacobian = flow_layer.forward_and_jacobian()
        sum_log_det_jacobians += log_det_jacobian
        flow_layers.append(flow_layer)
        ind += 1

        flow_layer = ShiftFlow(params[ind], Z)
        Z, log_det_jacobian = flow_layer.forward_and_jacobian()
        sum_log_det_jacobians += log_det_jacobian
        flow_layers.append(flow_layer)
        ind += 1

    # need to add support mapping
    if support_mapping is not None:
        final_layer = support_mapping(Z)
        Z, log_det_jacobian = final_layer.forward_and_jacobian()
        sum_log_det_jacobians += log_det_jacobian
        flow_layers.append(final_layer)

    return Z, sum_log_det_jacobians, flow_layers


def get_initdir(D, arch_dict, sigma, random_seed):
    # set file I/O stuff
    initdir = "data/inits/"
    archstring = get_archstring(arch_dict)
    initdir = initdir + "D=%d_%s_sigma=%.2f_rs=%d/" % (
        D,
        archstring,
        sigma,
        random_seed,
    )
    return initdir


def load_nf_init(initdir, arch_dict):
    initfile = np.load(initdir + "theta.npz")
    theta = initfile["theta"][()]
    scope = "density_network"
    inits_by_layer = []
    dims_by_layer = []
    layer_ind = 1

    if arch_dict["mult_and_shift"] == "pre":
        a_init = tf.constant(theta["%s/theta_1_1:0" % scope], dtype=DTYPE)
        inits_by_layer.append([a_init])
        dims_by_layer.append([a_init.shape])
        layer_ind += 1

        b_init = tf.constant(theta["%s/theta_2_1:0" % scope], dtype=DTYPE)
        inits_by_layer.append([b_init])
        dims_by_layer.append([b_init.shape])
        layer_ind += 1

    for i in range(arch_dict["repeats"]):
        if arch_dict["TIF_flow_type"] == "PlanarFlow":
            u_init = tf.constant(
                theta["%s/theta_%d_%d:0" % (scope, layer_ind, 1)], dtype=DTYPE
            )
            w_init = tf.constant(
                theta["%s/theta_%d_%d:0" % (scope, layer_ind, 2)], dtype=DTYPE
            )
            b_init = tf.constant(
                theta["%s/theta_%d_%d:0" % (scope, layer_ind, 3)], dtype=DTYPE
            )
            init_i = [u_init, w_init, b_init]
            dims_i = [u_init.shape, w_init.shape, b_init.shape]

        elif arch_dict["TIF_flow_type"] == "RealNVP":
            params_init = tf.constant(
                theta["%s/theta_%d_%d:0" % (scope, layer_ind, 1)], dtype=DTYPE
            )
            init_i = [params_init]
            dims_i = [params_init.shape]

        else:
            raise NotImplementedError()

        inits_by_layer.append(init_i)
        dims_by_layer.append(dims_i)
        layer_ind += 1

    if arch_dict["mult_and_shift"] == "post":
        a_init = tf.constant(theta["%s/theta_%d_1:0" % (scope, layer_ind)], dtype=DTYPE)
        inits_by_layer.append([a_init])
        dims_by_layer.append([a_init.shape])
        layer_ind += 1

        b_init = tf.constant(theta["%s/theta_%d_1:0" % (scope, layer_ind)], dtype=DTYPE)
        inits_by_layer.append([b_init])
        dims_by_layer.append([b_init.shape])
        layer_ind += 1

    return inits_by_layer, dims_by_layer


def construct_density_network(flow_dict, D_Z, T):
    """Generates the ordered list of instantiated density network layers.

        Args:
            flow_dict (dict): Specifies structure of approximating density network.
            D_Z (int): Dimensionality of the density network.
            T (int): Number of time points.

        Returns:
            layers (list): List of instantiated normalizing flows.
            num_theta_params (int): Total number of parameters in density network.

        """
    latent_layers = construct_latent_dynamics(flow_dict, D_Z, T)
    time_invariant_layers = construct_time_invariant_flow(flow_dict, D_Z, T)

    layers = latent_layers + time_invariant_layers
    nlayers = len(layers)

    num_theta_params = 0
    for i in range(nlayers):
        layer = layers[i]
        print(i, layer)
        num_theta_params += count_layer_params(layer)

    return layers, num_theta_params


def construct_latent_dynamics(flow_dict, D_Z, T):
    """Creates normalizing flow layer for dynamics.

        Args:
            flow_dict (dict): Specifies structure of approximating density network.
            D_Z (int): Dimensionality of the density network.
            T (int): Number of time points.

        Returns:
            layers (list): List of instantiated normalizing flows.
            num_theta_params (int): Total number of parameters in density network.

        """
    latent_dynamics = flow_dict["latent_dynamics"]

    if latent_dynamics is None:
        return []

    if not (latent_dynamics == "flatten"):
        inits = flow_dict["inits"]

    if "lock" in flow_dict:
        lock = flow_dict["lock"]
    else:
        lock = False

    if latent_dynamics == "GP":
        layer = GP_Layer("GP_Layer", dim=D_Z, inits=inits, lock=lock)
        layers = [layer]

    elif latent_dynamics == "GP_EP":
        layer = GP_EP_CondRegLayer(
            "GP_EP_CondRegLayer",
            dim=D_Z,
            T=T,
            Tps=flow_dict["Tps"],
            tau_init=inits["tau_init"],
            lock=lock,
        )
        layers = [layer]

    elif latent_dynamics == "AR":
        param_init = {
            "alpha_init": inits["alpha_init"],
            "sigma_init": inits["sigma_init"],
        }
        layer = AR_Layer(
            "AR_Layer", dim=D_Z, T=T, P=flow_dict["P"], inits=inits, lock=lock
        )
        layers = [layer]

    elif latent_dynamics == "VAR":
        param_init = {"A_init": inits["A_init"], "sigma_init": inits["sigma_init"]}
        layer = VAR_Layer(
            "VAR_Layer", dim=D_Z, T=T, P=flow_dict["P"], inits=inits, lock=lock
        )
        layers = [layer]

    elif latent_dynamics == "flatten":
        if "Flatten_flow_type" in flow_dict.keys():
            layers = []
            layer_ind = 0
            for i in range(flow_dict["flatten_repeats"]):
                layers.append(
                    PlanarFlowLayer(
                        "Flat_Planar_Layer_%d" % layer_ind, dim=int(D_Z * T)
                    )
                )
                layer_ind += 1
        else:
            layer = FullyConnectedFlowLayer("Flat_Affine_Layer", dim=int(D_Z * T))
            layers = [layer]

    else:
        raise NotImplementedError()

    return layers


def construct_time_invariant_flow(flow_dict, D_Z, T):
    """Creates list of time-invariant normalizing flow layers.

        Args:
            flow_dict (dict): Specifies structure of approximating density network.
            D_Z (int): Dimensionality of the density network.
            T (int): Number of time points.

        Returns:
            layers (list): List of instantiated normalizing flows.
            num_theta_params (int): Total number of parameters in density network.

    """

    layer_ind = 1
    layers = []
    TIF_flow_type = flow_dict["TIF_flow_type"]
    repeats = flow_dict["repeats"]
    elem_mult_flow = flow_dict["elem_mult_flow"]
    nlayers = repeats + elem_mult_flow
    if "inits" in flow_dict.keys():
        inits = flow_dict["inits"]
    else:
        inits = nlayers * [None]

    if TIF_flow_type == "ScalarFlowLayer":
        flow_class = ElemMultLayer
        name_prefix = "ScalarFlow_Layer"

    elif TIF_flow_type == "FullyConnectedFlowLayer":
        flow_class = FullyConnectedFlowLayer
        name_prefix = FullyConnectedFlow_Layer

    elif TIF_flow_type == "AffineFlowLayer":
        flow_class = AffineFlowLayer
        name_prefix = "AffineFlow_Layer"

    elif TIF_flow_type == "StructuredSpinnerLayer":
        flow_class = StructuredSpinnerLayer
        name_prefix = "StructuredSpinner_Layer"

    elif TIF_flow_type == "StructuredSpinnerTanhLayer":
        flow_class = StructuredSpinnerTanhLayer
        name_prefix = "StructuredSpinnerTanh_Layer"

    elif TIF_flow_type == "PlanarFlowLayer":
        flow_class = PlanarFlowLayer
        name_prefix = "PlanarFlow_Layer"

    elif TIF_flow_type == "RadialFlowLayer":
        flow_class = RadialFlowLayer
        name_prefix = "RadialFlow_Layer"

    elif TIF_flow_type == "TanhLayer":
        flow_class = TanhLayer
        name_prefix = "Tanh_Layer"

    else:
        raise NotImplementedError()

    if elem_mult_flow:
        layers.append(
            ElemMultLayer(
                "ScalarFlow_Layer_%d" % layer_ind, D_Z, inits=inits[layer_ind - 1]
            )
        )
        layer_ind += 1

    for i in range(repeats):
        layers.append(
            flow_class(
                "%s%d" % (name_prefix, layer_ind), D_Z, inits=inits[layer_ind - 1]
            )
        )
        layer_ind += 1

    return layers


def declare_theta(layers, inits=None):
    """Declare tensorflow variables for the density network.

        Args:
            layers (list): List of instantiated normalizing flows.

        Returns:
            theta (list): List of tensorflow variables for the density network.

    """
    L_flow = len(layers)
    theta = []
    for i in range(L_flow):
        layer = layers[i]
        layer_name, param_names, param_dims, initializers, lock = layer.get_layer_info()
        nparams = len(param_names)
        layer_i_params = []
        for j in range(nparams):
            if lock:
                param_ij = initializers[j]
            else:
                if isinstance(initializers[j], tf.Tensor):
                    param_ij = tf.get_variable(
                        layer_name + "_" + param_names[j],
                        dtype=tf.float64,
                        initializer=initializers[j],
                    )
                else:
                    param_ij = tf.get_variable(
                        layer_name + "_" + param_names[j],
                        shape=param_dims[j],
                        dtype=tf.float64,
                        initializer=initializers[j],
                    )
            layer_i_params.append(param_ij)
        theta.append(layer_i_params)
    return theta


def connect_density_network(W, layers, theta, ts=None):
    """Update parameters of normalizing flow layers to be the tensorflow theta variable,
       while pushing isotropic gaussian samples W through the network.

        This method is used for both EFN (theta is output of parameter network) and for
        NF (theta is a list of declared tensorflow variables that are optimized).

        Args:
            W (tf.Tensor): Isotropic gaussian samples.
            layers (list): List of instantiated normalizing flows.
            theta (tf.Tensor (if EFN), tf.Variable (if NF)): Density network parameters.

        Returns:
            Z (tf.Tensor): Density network samples.
            sum_log_det_jacobians (tf.Tensor): Sum of the log absolute value determinant
                                               of the jacobians of the forward 
                                               transformations.
            Z_by_layer (list): List of density network samples at each layer.

    """
    W_shape = tf.shape(W)
    K = W_shape[0]
    M = W_shape[1]
    D_Z = W_shape[2]
    T = W_shape[3]

    sum_log_det_jacobians = tf.zeros((K, M), dtype=tf.float64)
    nlayers = len(layers)
    Z_by_layer = []
    Z_by_layer.append(W)
    Z = W
    for i in range(nlayers):
        layer = layers[i]
        theta_layer = theta[i]
        layer.connect_parameter_network(theta_layer)
        if isinstance(layer, GP_Layer) or isinstance(layer, GP_EP_CondRegLayer):
            Z, sum_log_det_jacobians = layer.forward_and_jacobian(
                Z, sum_log_det_jacobians, ts
            )
        elif layer.name[:4] == "Flat":
            Z = tf.reshape(Z, [K, M, D_Z * T, 1])
            Z, sum_log_det_jacobians = layer.forward_and_jacobian(
                Z, sum_log_det_jacobians
            )
            Z = tf.reshape(Z, [K, M, D_Z, T])
        else:
            Z, sum_log_det_jacobians = layer.forward_and_jacobian(
                Z, sum_log_det_jacobians
            )
        Z_by_layer.append(Z)
    return Z, sum_log_det_jacobians, Z_by_layer


def count_layer_params(layer):
    """Count number of params in a normalizing flow layer.

        Args:
            layer (Layer): Instance of a normalizing flow.

        Returns:
            num_params (int): The total number of parameters in the layer.

    """
    num_params = 0
    name, param_names, dims, _, _ = layer.get_layer_info()
    nparams = len(dims)
    for j in range(nparams):
        num_params += np.prod(dims[j])
    return num_params


def count_params(all_params):
    """Count total parameters in the model.

        Args:
            all_params (list): List of tf.Variables.

        Returns:
            nparam_vals (int): The total number of parameters in the model.

    """
    nparams = len(all_params)
    nparam_vals = 0
    for i in range(nparams):
        param = all_params[i]
        param_shape = tuple(param.get_shape().as_list())
        nparam_vals += np.prod(param_shape)
    return nparam_vals


def log_grads(cost_grads, cost_grad_vals, ind):
    cgv_ind = 0
    nparams = len(cost_grads)
    for i in range(nparams):
        grad = cost_grads[i]
        grad_shape = grad.shape
        ngrad_vals = np.prod(grad_shape)
        grad_reshape = np.reshape(grad, (ngrad_vals,))
        for ii in range(ngrad_vals):
            cost_grad_vals[ind, cgv_ind] = grad_reshape[ii]
            cgv_ind += 1
    return None


def gradients(f, x, grad_ys=None):
    """
    An easier way of computing gradients in tensorflow. The difference from tf.gradients is
        * If f is not connected with x in the graph, it will output 0s instead of Nones. This will be more meaningful
            for computing higher-order gradients.
        * The output will have the same shape and type as x. If x is a list, it will be a list. If x is a Tensor, it
            will be a tensor as well.
    :param f: A `Tensor` or a list of tensors to be differentiated
    :param x: A `Tensor` or a list of tensors to be used for differentiation
    :param grad_ys: Optional. It is a `Tensor` or a list of tensors having exactly the same shape and type as `f` and
                    holds gradients computed for each of `f`.
    :return: A `Tensor` or a list of tensors having the same shape and type as `x`

    got this func from https://gist.github.com/yang-song/07392ed7d57a92a87968e774aef96762
    """

    if isinstance(x, list):
        grad = tf.gradients(f, x, grad_ys=grad_ys)
        for i in range(len(x)):
            if grad[i] is None:
                grad[i] = tf.zeros_like(x[i])
        return grad
    else:
        grad = tf.gradients(f, x, grad_ys=grad_ys)[0]
        if grad is None:
            return tf.zeros_like(x)
        else:
            return grad


def Lop(f, x, v):
    """
    Compute Jacobian-vector product. The result is v^T @ J_x
    :param f: A `Tensor` or a list of tensors for computing the Jacobian J_x
    :param x: A `Tensor` or a list of tensors with respect to which the Jacobian is computed.
    :param v: A `Tensor` or a list of tensors having the same shape and type as `f`
    :return: A `Tensor` or a list of tensors having the same shape and type as `x`


    got this func from https://gist.github.com/yang-song/07392ed7d57a92a87968e774aef96762
    """
    assert not isinstance(f, list) or isinstance(
        v, list
    ), "f and v should be of the same type"
    return gradients(f, x, grad_ys=v)


def AL_cost(log_q_z, T_x_mu_centered, Lambda, c, all_params):
    T_x_shape = tf.shape(T_x_mu_centered)
    M = T_x_shape[1]
    half_M = M // 2
    H = -tf.reduce_mean(log_q_z)
    R = tf.reduce_mean(T_x_mu_centered[0], 0)
    cost_terms_1 = -H + tf.tensordot(Lambda, R, axes=[0, 0])
    cost = cost_terms_1 + (c / 2.0) * tf.reduce_sum(tf.square(R))
    grad_func1 = tf.gradients(cost_terms_1, all_params)

    T_x_1 = T_x_mu_centered[0, :half_M, :]
    T_x_2 = T_x_mu_centered[0, half_M:, :]
    grad_con = Lop(T_x_1, all_params, T_x_2)
    grads = []
    nparams = len(all_params)
    for i in range(nparams):
        grads.append(grad_func1[i] + c * grad_con[i] / tf.cast(half_M, DTYPE))

    return cost, grads, H


"""def AL_cost(log_q_z, T_x_mu_centered, Lambda, c, all_params):
    T_x_shape = tf.shape(T_x_mu_centered)
    M = T_x_shape[1]
    H = -tf.reduce_mean(log_q_z)
    R = tf.reduce_mean(T_x_mu_centered[0], 0)
    cost_terms_1 = -H + tf.tensordot(Lambda, R, axes=[0, 0])
    cost = cost_terms_1 + (c / 2.0) * tf.reduce_sum(tf.square(R))
    grad_func1 = tf.gradients(cost_terms_1, all_params)

    T_x_1 = T_x_mu_centered[0, : (M // 2), :]
    T_x_2 = T_x_mu_centered[0, (M // 2) :, :]
    grad_con = Lop(T_x_1, all_params, T_x_2)
    grads = []
    nparams = len(all_params)
    for i in range(nparams):
        grads.append(grad_func1[i] + c * grad_con[i])

    return cost, grads, H"""


def load_nf_vars(initdir):
    saver = tf.train.import_meta_graph(initdir + "model.meta", import_scope="DSN")
    W = tf.get_collection("W")[0]
    phi = tf.get_collection("Z")[0]
    log_q_phi = tf.get_collection("log_q_zs")[0]
    return W, phi, log_q_phi, saver


def memory_extension(input_arrays, array_cur_len):
    """Extend numpy arrays tracking model diagnostics.

        Args:
            input_arrays (np.array): Arrays to extend.
            array_cur_len (int): Current array lengths.

        Returns:
            extended_arrays (np.array): Extended arrays.

        """
    print("Doubling memory allocation for parameter logging.")
    n = len(input_arrays)
    extended_arrays = []
    for i in range(n):
        input_array = input_arrays[i]
        extended_array = np.concatenate(
            (input_array, np.zeros((array_cur_len, input_array.shape[1]))), axis=0
        )
        extended_arrays.append(extended_array)
    return extended_arrays


def get_archstring(arch_dict):
    """Get string description of density network.

        Args:
            arch_dict (dict): Specifies structure of approximating density network.

        Returns:
            tif_str (str): String specifying time-invariant flow network architecture.

        """
    latent_dynamics = arch_dict["latent_dynamics"]
    tif_flow_type = arch_dict["TIF_flow_type"]
    repeats = arch_dict["repeats"]

    tif_str = get_TIF_string(arch_dict)

    if arch_dict["mult_and_shift"] == "pre":
        arch_str = "M_A_%d%s" % (repeats, tif_str)
    elif arch_dict["mult_and_shift"] == "post":
        arch_str = "%d%s_M_A" % (repeats, tif_str)
    else:
        arch_str = "%d%s" % (repeats, tif_str)
    if latent_dynamics is not None:
        return "%s_%s" % (latent_dynamics, arch_str)
    else:
        return arch_str

def get_TIF_string(arch_dict):
    tif_flow_type = arch_dict["TIF_flow_type"]
    if (tif_flow_type == "RealNVP"):
        real_nvp_arch = arch_dict['real_nvp_arch']
        tif_str = 'R_%dM_%dL_%dU' % (real_nvp_arch["num_masks"], \
                                     real_nvp_arch["nlayers"], \
                                     real_nvp_arch["upl"])
    else:
        tif_str = tif_flow_type[:1]
    return tif_str


