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
import os
import scipy
from tf_util.normalizing_flows import (
    PlanarFlow,
    AffineFlow,
    ShiftFlow,
    ElemMultFlow,
    get_flow_class,
    get_density_network_inits,
    get_mixture_density_network_inits,
    RealNVP,
)

DTYPE = tf.float64

def init_layer_params(inits, dims, layer_ind):
    num_inits = len(inits)
    params = []
    for j in range(num_inits):
        varname_j = "theta_%d_%d" % (layer_ind, j + 1)
        if isinstance(inits[j], tf.Tensor):
            var_j = tf.get_variable(
                varname_j, dtype=DTYPE, initializer=inits[j]
            )
        else:
            var_j = tf.get_variable(
                varname_j,
                shape=(dims[j],),
                dtype=DTYPE,
                initializer=inits[j],
            )
        params.append(tf.expand_dims(var_j,0))
    return tf.concat(params, 1)



def density_network(W, arch_dict, support_mapping=None, initdir=None, theta=None):
    connect_params = theta is not None

    if (initdir is not None) and  connect_params:
        print('Usage error: An initialization directory provided when connecting a parameter network (connect_params==True).  Either initdir must be None or connect params must be False.')
        exit()

    D = arch_dict["D"]
    if (not connect_params):
        if initdir is None:
            inits_by_layer, dims_by_layer = get_density_network_inits(arch_dict)
        else:
            inits_by_layer, dims_by_layer = load_nf_init(initdir, arch_dict)
            print("Loaded optimized initialization.")

    # declare layer parameters with initializations
    params = []
    with tf.variable_scope("DensityNetwork"):
        Z = W
        flow_layers = []
        sum_log_det_jacobians = 0.0
        ind = 0

        flow_class = get_flow_class(arch_dict["flow_type"])
        for i in range(arch_dict["repeats"]):
            with tf.variable_scope("Layer%d" % (i+1)):
                if (connect_params):
                    params = theta[ind]
                else:
                    params = init_layer_params(inits_by_layer[ind], dims_by_layer[ind], ind+1)

                if (flow_class in [PlanarFlow, AffineFlow]):
                    flow_layer = flow_class(params, Z)
                    Z, log_det_jacobian = flow_layer.forward_and_jacobian()
                elif (flow_class == RealNVP):
                    real_nvp_arch = arch_dict['real_nvp_arch']
                    num_masks = real_nvp_arch['num_masks']
                    real_nvp_layers = real_nvp_arch['nlayers']
                    upl = real_nvp_arch['upl']
                    flow_layer = flow_class(params, Z, num_masks, real_nvp_layers, upl)
                    Z, log_det_jacobian = flow_layer.forward_and_jacobian()
                else:
                    raise NotImplementedError()
                sum_log_det_jacobians += log_det_jacobian
                flow_layers.append(flow_layer)
                ind += 1

        if arch_dict["post_affine"]:
            with tf.variable_scope("PostMultLayer"):
                if connect_params:
                    params = theta[ind]
                else:
                    params = init_layer_params(inits_by_layer[ind], dims_by_layer[ind], ind+1)
                flow_layer = ElemMultFlow(params, Z)
                Z, log_det_jacobian = flow_layer.forward_and_jacobian()
                sum_log_det_jacobians += log_det_jacobian
                flow_layers.append(flow_layer)
                ind += 1

            with tf.variable_scope("PostShiftLayer"):
                if connect_params:
                    params = theta[ind]
                else:
                    params = init_layer_params(inits_by_layer[ind], dims_by_layer[ind], ind+1)
                flow_layer = ShiftFlow(params, Z)
                Z, log_det_jacobian = flow_layer.forward_and_jacobian()
                sum_log_det_jacobians += log_det_jacobian
                flow_layers.append(flow_layer)
                ind += 1

    # need to add support mapping
    if support_mapping is not None:
        with tf.variable_scope("SupportMapping"):
            final_layer = support_mapping(Z)
            Z, log_det_jacobian = final_layer.forward_and_jacobian()
            sum_log_det_jacobians += log_det_jacobian
            flow_layers.append(final_layer)

    return Z, sum_log_det_jacobians, flow_layers


def mixture_density_network(G, W, arch_dict, support_mapping=None, initdir=None):
    """
        G (int tf.tensor) : (1 x M x K) Gumble random variables
        W (tf.tensor) : (1 x M x D) isotropic noise
    """
    D = arch_dict["D"]
    K = arch_dict["K"]
    assert(K > 1)
    if initdir is None:
        gaussian_inits, inits_by_layer, dims_by_layer = get_mixture_density_network_inits(arch_dict)
        print("Got random initialization.")
    else:
        gaussian_inits, _, _ = get_mixture_density_network_inits(arch_dict)
        inits_by_layer, dims_by_layer = load_nf_init(initdir, arch_dict)
        print("MoG init with tuned density net.")

    # declare layer parameters with initializations
    params = []
    with tf.variable_scope("MixtureDensityNetwork"):

        with tf.variable_scope("MoG"):
            mixture_of_gaussians = []
            mus = []
            sigmas = []
            for k in range(K):
                mu_k_init, log_sigma_k_init = gaussian_inits[k]
                mu_k = tf.get_variable('mu_%d' % (k+1), dtype=tf.float64, initializer=mu_k_init)
                log_sigma_k = tf.get_variable('log_sigma_%d' % (k+1), dtype=tf.float64, initializer=log_sigma_k_init)
                sigma_k = tf.exp(log_sigma_k)
                mus.append(mu_k)
                sigmas.append(sigma_k)
                mixture_of_gaussians.append((mu_k, sigma_k))
            mu = tf.stack(mus, 0) # (K x D)
            sigma = tf.stack(sigmas, 0) # (K x D)

            # Gumbel Softmax Trick
            tau = 0.05
            beta_init = tf.zeros((K-1,), tf.float64)
            beta = tf.get_variable('beta', initializer=beta_init)
            exp_beta = tf.exp(beta)
            alpha = tf.concat((exp_beta, tf.ones((1,), tf.float64)), 0) \
                    / (tf.reduce_sum(exp_beta) + 1.0)
            C = gumbel_softmax_trick(G, alpha, tau) # (M x K)

            # select mu_k and sigma_k accordingly
            C_dot_mu = tf.matmul(C, mu) # (M x D)
            C_dot_sigma = tf.matmul(C, sigma) # (M x D)
            Z = C_dot_mu + tf.multiply(C_dot_sigma, W[0])
            Z = tf.expand_dims(Z, 0) # (1 x M x D)

            # write density calculation
            #log_p_c = gumbel_softmax_log_density(K, C, alpha, tau)
            #sum_log_det_jacobians = tf.reduce_sum(tf.log(tf.square(C_dot_sigma)), 1)
            #log_q_w = tf.reduce_sum((-tf.square(W) / 2.0) - np.log(np.sqrt(2.0 * np.pi)), 2)
            #q_w = tf.reduce_prod(tf.exp((-tf.square(W)) / 2.0) / np.sqrt(2.0 * np.pi), axis=2) # (1 x M)
            #mean_prod_sigmas = tf.reduce_mean(tf.reduce_prod(C_dot_sigma, 1), 0)
            #log_base_density = log_q_w - tf.log(mean_prod_sigmas)
            C_dot_mu = tf.expand_dims(C_dot_mu, 1)
            C_dot_sigma = tf.expand_dims(C_dot_sigma, 1)
            p_z0_mid_c = tf.reduce_prod(tf.divide(tf.exp(tf.divide(-tf.square(Z-C_dot_mu), 2.0*C_dot_sigma)), 
                                            np.sqrt(2.0 *np.pi)*C_dot_sigma), axis=2) # (1 x M)
            p_z0 = tf.reduce_mean(p_z0_mid_c, 0)
            log_base_density = tf.expand_dims(tf.log(p_z0), 0)

            sum_log_det_jacobians = 0.0
            
        #if (is_shared): Can't actually use K different density networks with MoG
        # If want to use different density networks, we need to use RaD or similar
        flow_layers = []
        ind = 0

        flow_class = get_flow_class(arch_dict["flow_type"])
        for i in range(arch_dict["repeats"]):
            with tf.variable_scope("Layer%d" % (i+1)):
                params = init_layer_params(inits_by_layer[ind], dims_by_layer[ind], ind+1)
                if (flow_class == PlanarFlow):
                    flow_layer = flow_class(params, Z)
                    Z, log_det_jacobian = flow_layer.forward_and_jacobian()
                elif (flow_class == RealNVP):
                    real_nvp_arch = arch_dict['real_nvp_arch']
                    num_masks = real_nvp_arch['num_masks']
                    real_nvp_layers = real_nvp_arch['nlayers']
                    upl = real_nvp_arch['upl']
                    flow_layer = flow_class(params, Z, num_masks, real_nvp_layers, upl)
                    Z, log_det_jacobian = flow_layer.forward_and_jacobian()
                else:
                    raise NotImplementedError()
                sum_log_det_jacobians += log_det_jacobian
                flow_layers.append(flow_layer)
                ind += 1

        if arch_dict["post_affine"]:
            with tf.variable_scope("PostMultLayer"):
                params = init_layer_params(inits_by_layer[ind], dims_by_layer[ind], ind+1)
                flow_layer = ElemMultFlow(params, Z)
                Z, log_det_jacobian = flow_layer.forward_and_jacobian()
                sum_log_det_jacobians += log_det_jacobian
                flow_layers.append(flow_layer)
                ind += 1

            with tf.variable_scope("PostShiftLayer"):
                params = init_layer_params(inits_by_layer[ind], dims_by_layer[ind], ind+1)
                flow_layer = ShiftFlow(params, Z)
                Z, log_det_jacobian = flow_layer.forward_and_jacobian()
                sum_log_det_jacobians += log_det_jacobian
                flow_layers.append(flow_layer)
                ind += 1

        # need to add support mapping
        if support_mapping is not None:
            with tf.variable_scope("SupportMapping"):
                final_layer = support_mapping(Z)
                Z, log_det_jacobian = final_layer.forward_and_jacobian() 
                sum_log_det_jacobians += log_det_jacobian
                flow_layers.append(final_layer)

    return Z, sum_log_det_jacobians, log_base_density, flow_layers, alpha, mu, sigma, C


def gumbel_softmax_trick(G, alpha, tau):
    """
        G (int tf.tensor) : (1 x M x K) Gumble random variables
        W (tf.tensor) : (1 x M x D) isotropic noise
    """
    alpha = tf.expand_dims(alpha, 0)
    G = G[0]
    C_unnorm = tf.exp((tf.log(alpha) + G)/tau)
    C = tf.divide(C_unnorm, tf.expand_dims(tf.reduce_sum(C_unnorm, 1), 1))
    return C
    
def gumbel_softmax_log_density(K, C, alpha, tau):
    """
        G (int tf.tensor) : (M x K) Gumble random variables
        alpha (tf.tensor) : (K,) cluster prob
        tau (scalar) : temperature of the Gumbel dist
    """
    alpha = tf.expand_dims(alpha, 0)
    log_gamma = scipy.special.loggamma(K)
    log_tau = np.log(tau)
    log_sum = tf.log(tf.reduce_sum(tf.divide(alpha, tf.pow(C, tau)), 1))
    sum_log = tf.reduce_sum(tf.log(alpha) - (tau+1)*tf.log(C), 1)
    log_p_c = log_gamma + (K-1)*log_tau - K*log_sum + sum_log
    return log_p_c

def get_initdir(system, arch_dict, sigma, random_seed, init_type="gauss"):
    # set file I/O stuff
    prefix = "data/inits/"
    archstring = get_archstring(arch_dict, init=True)

    init_mu_str = "%.2f" % system.density_network_init_mu[0]
    if system.D > 1:
        for i in range(1, system.D):
            init_mu_str += "_%.2f" % system.density_network_init_mu[i]

    sysparams = system.free_params[0]
    num_free_params = len(system.free_params)
    if num_free_params > 1:
        for i in range(1, num_free_params):
            sysparams += "_%s" % system.free_params[i]

    if (system.density_network_bounds is not None):
        if (system.name == "V1Circuit"):
            initdir = prefix + "D=%d_%s_%s_%s_mu=%s_sigma=%.2f_rs=%d/" % (
                    system.D,
                    sysparams,
                    init_type,
                    archstring,
                    init_mu_str,
                    sigma,
                    random_seed,
                    )
        else:
            initdir = prefix + "D=%d_%s_%s_%s_mu=%s_sigma=%.2f_from_%.2f_to_%.2f_rs=%d/" % (
                    system.D,
                    sysparams,
                    init_type,
                    archstring,
                    init_mu_str,
                    sigma,
                    system.density_network_bounds[0],
                    system.density_network_bounds[1],
                    random_seed,
                    )
    else:
        initdir = prefix + "D=%d_%s_%s_%s_mu=%s_sigma=%.2f_rs=%d/" % (
                    system.D,
                    sysparams,
                    init_type,
                    archstring,
                    init_mu_str, 
                    sigma,
                    random_seed,
                    )

    return initdir

def check_init(initdir):
    initfname = initdir + "theta.npz"
    resfname = initdir + "opt_info.npz"
    check_passed = False
    if os.path.exists(initfname):
        resfile = np.load(resfname, allow_pickle=True)
        # Make sure it has converged
        if not resfile["converged"]:
            print("Error: Found init file, but optimiation has not converged.")
            print("Tip: Consider adjusting approximation architecture or min_iters.")
            print("Delete the init directory if optimization was killed early.")
            exit()
        check_passed = True
    return check_passed


def load_nf_init(initdir, arch_dict):
    initfile = np.load(initdir + "theta.npz", allow_pickle=True)
    theta = initfile["theta"][()]
    scope = "DensityNetwork"
    inits_by_layer = []
    dims_by_layer = []
    layer_ind = 1

    for i in range(arch_dict["repeats"]):
        if arch_dict["flow_type"] == "PlanarFlow":
            u_init = tf.constant(
                theta["%s/Layer%d/theta_%d_%d:0" % (scope, layer_ind, layer_ind, 1)], dtype=DTYPE
            )
            w_init = tf.constant(
                theta["%s/Layer%d/theta_%d_%d:0" % (scope, layer_ind, layer_ind, 2)], dtype=DTYPE
            )
            b_init = tf.constant(
                theta["%s/Layer%d/theta_%d_%d:0" % (scope, layer_ind, layer_ind, 3)], dtype=DTYPE
            )
            init_i = [u_init, w_init, b_init]
            dims_i = [u_init.shape, w_init.shape, b_init.shape]

        elif arch_dict["flow_type"] == "RealNVP":
            params_init = tf.constant(
                theta["%s/Layer%d/theta_%d_%d:0" % (scope, layer_ind, layer_ind, 1)], dtype=DTYPE
            )
            init_i = [params_init]
            dims_i = [params_init.shape]

        else:
            raise NotImplementedError()

        inits_by_layer.append(init_i)
        dims_by_layer.append(dims_i)
        layer_ind += 1

    if arch_dict["post_affine"]:
        a_init = tf.constant(theta["%s/PostMultLayer/theta_%d_1:0" % (scope, layer_ind)], dtype=DTYPE)
        inits_by_layer.append([a_init])
        dims_by_layer.append([a_init.shape])
        layer_ind += 1

        b_init = tf.constant(theta["%s/PostShiftLayer/theta_%d_1:0" % (scope, layer_ind)], dtype=DTYPE)
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


def AL_cost(H, T_x_mu_centered, Lambda, c, all_params, entropy=True, I_x=None):
    """Computes tensorflow gradients of an augmented lagrangian cost.

        Args:
            H (tf.tensor): (1,) Entropy of batch of zs.
            T_x_mu_centered (tf.tensor): [1,M,|T|] Mean-centered suff stats.
            Lambda (tf.tensor) [|T|] Augmented Lagrangian parameters.
            c (tf.tensor) [()] Augmented Lagrangian parameter.
            all_params (list) Parameters to take gradients with respect to.
            entropy (bool) If true, optimize entropy.
            I_x (tf.tensor) [1,M,|I|] Inequality constraint function values.


        Returns:
            cost (tf.Tensor): [()] The Augmented Lagrangian cost.
            grads (list): List of gradients.
            entropy (tf.tensor): [()] The total number of parameters in the layer.

    """
    T_x_shape = tf.shape(T_x_mu_centered)
    M = T_x_shape[1]
    half_M = M // 2
    R = tf.reduce_mean(T_x_mu_centered[0], 0)
    if entropy:
        cost_terms_1 = -H + tf.tensordot(Lambda, R, axes=[0, 0])
    else:
        cost_terms_1 = tf.tensordot(Lambda, R, axes=[0, 0])
    cost = cost_terms_1 + (c / 2.0) * tf.reduce_sum(tf.square(R))
    grad_func1 = tf.gradients(cost_terms_1, all_params)

    T_x_1 = T_x_mu_centered[0, :half_M, :]
    T_x_1_mean = tf.reduce_mean(T_x_1, 0)
    T_x_2 = T_x_mu_centered[0, half_M:, :]
    T_x_2_mean = tf.reduce_mean(T_x_2, 0)
    grad_con = Lop(T_x_1_mean, all_params, T_x_2_mean)
    grads = []
    nparams = len(all_params)
    for i in range(nparams):
        grads.append(grad_func1[i] + c * grad_con[i])

    if (I_x is not None):
        I_x_mean = tf.reduce_mean(I_x[0], axis=0)
        ineq_con_grad = tf.gradients(I_x_mean, all_params)
        for i in range(nparams):
            grads[i] += ineq_con_grad[i]

    return cost, grads, R


def max_barrier(u, alpha, t):
    """Log barrier penalty for stats u, bound alpha, and parameter t.

        Enforces a maximum alpha on the statistic u.

        f(u ; alpha, t) = -(1/t)log(-u + alpha)

        Args:
            u (tf.tensor): [1,M] Batch statistics.
            alpha (float): The desired bound.
            t (float): Greater t, better approximation of indicator.

        Returns:
            f_u (tf.tensor): [1,M] f(u; alpha, t) 

    """

    return -(1.0/t)*tf.log(-u + alpha)

def min_barrier(u, alpha, t):
    """Log barrier penalty for stats u, bound alpha, and parameter t.

        Enforces a minimum alpha on the statistic u.

        f(u ; alpha, t) = -(1/t)log(u - alpha)

        Args:
            u (tf.tensor): [1,M] Batch statistics.
            alpha (float): The desired bound.
            t (float): Greater t, better approximation of indicator.

        Returns:
            f_u (tf.tensor): [1,M] f(u; alpha, t) 

    """

    return -(1.0/float(t))*tf.log(u - float(alpha))


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


def get_mep_archstring(arch_dict):
    """Get string description of latent dynamical density network.

        Args:
            arch_dict (dict): Specifies structure of approximating density network.

        Returns:
            arch_str (str): String specifying flow network architecture.

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

def get_archstring(arch_dict, init=False):
    """Get string description of density network.

        Args:
            arch_dict (dict): Specifies structure of approximating density network.

        Returns:
            archstr (str): String specifying flow network architecture.

        """
    K = arch_dict["K"] # mixture components
    flow_type = arch_dict["flow_type"]
    repeats = arch_dict["repeats"]

    flow_type_str = get_flow_type_string(arch_dict)

    if K == 1 or init:
        arch_str = ""
    elif K > 1:
        arch_str = "K=%d_s0=%.1f_" % (K, arch_dict['sigma0'])
    else:
        print('Error: K must be positive integer.')
        exit()

    arch_str += "%d%s" % (repeats, flow_type_str)

    if arch_dict["post_affine"]:
        arch_str += "_M_A"

    return arch_str

def get_flow_type_string(arch_dict):
    flow_type = arch_dict["flow_type"]
    if (flow_type == "AffineFlow"):
        flow_type_str = "Aff"
    elif (flow_type == "CholProdFlow"):
        flow_type_str = "C"
    elif (flow_type == "ElemMultFlow"):
        flow_type_str = "M"
    elif (flow_type == "ExpFlow"):
        flow_type_str = "Exp"
    elif (flow_type == "IntervalFlow"):
        flow_type_str = "I"
    elif (flow_type == "PlanarFlow"):
        flow_type_str = "P"
    elif (flow_type == "RadialFlow"):
        flow_type_str = "R"
    if (flow_type == "RealNVP"):
        real_nvp_arch = arch_dict['real_nvp_arch']
        flow_type_str = 'R_%dM_%dL_%dU' % (real_nvp_arch["num_masks"], \
                                     real_nvp_arch["nlayers"], \
                                     real_nvp_arch["upl"])
    elif (flow_type == "ShiftFlow"):
        flow_type_str = "A"
    elif (flow_type == "SimplexBijectionFlow"):
        flow_type_str = "Simp"
    elif (flow_type == "SoftPlusFlow"):
        flow_type_str = "Soft"
    elif (flow_type == "StructuredSpinnerFlow"):
        flow_type_str = "SS"
    elif (flow_type == "StructuredSpinnerTanhFlow"):
        flow_type_str = "SST"
    elif (flow_type == "TanhFlow"):
        flow_type_str = "Tanh"
    return flow_type_str 


# Functions for the quartic formula

def quartic_delta(a, b, c, d, e):
    """Computes delta for quartic formula.

    https://en.wikipedia.org/wiki/Quartic_function

    f(x) = ax^4 + bx^3 + cx^2 + dx + e = 0

    Args:
        a (tf.tensor): coefficients for x^4
        b (tf.tensor): coefficients for x^3
        c (tf.tensor): coefficients for x^2
        d (tf.tensor): coefficients for x^1
        e (tf.tensor): coefficients for x^0

    Returns:
    	delta (tf.tensor): discriminant of quartic formula
    """
    delta = 256.0*(a**3)*(e**3) \
            - 192.0*(a**2)*b*d*(e**2) \
            - 128.0*(a**2)*(c**2)*(e**2) \
            + 144.0*(a**2)*c*(d**2)*e \
            -  27.0*(a**2)*(d**4) \
            + 144.0*a*(b**2)*c*(e**2) \
            -   6.0*a*(b**2)*(d**2)*e \
            -  80.0*a*b*(c**2)*d*e \
            +  18.0*a*b*c*(d**3) \
            +  16.0*a*(c**4)*e \
            -   4.0*a*(c**3)*(d**2) \
            -  27.0*(b**4)*(e**2) \
            +  18.0*(b**3)*c*d*e \
            -   4.0*(b**3)*(d**3) \
            -   4.0*(b**2)*(c**3)*e \
            +   1.0*(b**2)*(c**2)*(d**2)
    return delta

def quartic_delta0(a, b, c, d, e):
    """Computes delta0 for quartic formula.

    https://en.wikipedia.org/wiki/Quartic_function

    f(x) = ax^4 + bx^3 + cx^2 + dx + e = 0

    Args:
        a (tf.tensor): coefficients for x^4
        b (tf.tensor): coefficients for x^3
        c (tf.tensor): coefficients for x^2
        d (tf.tensor): coefficients for x^1
        e (tf.tensor): coefficients for x^0

    Returns:
    	delta0 (tf.tensor): delta0 of quartic formula
    """
    delta0 = c**2 -3.0*b*d + 12.0*a*e
    return delta0

def quartic_delta1(a, b, c, d, e):
    """Computes p for quartic formula.

    https://en.wikipedia.org/wiki/Quartic_function

    f(x) = ax^4 + bx^3 + cx^2 + dx + e = 0

    Args:
        a (tf.tensor): coefficients for x^4
        b (tf.tensor): coefficients for x^3
        c (tf.tensor): coefficients for x^2
        d (tf.tensor): coefficients for x^1
        e (tf.tensor): coefficients for x^0

    Returns:
    	delta1 (tf.tensor): delta1 of quartic formula
    """
    delta1 = 2.0*(c**3) - 9.0*b*c*d + 27.0*(b**2)*e + 27.0*a*(d**2) - 72.0*a*c*e
    return delta1

def quartic_Q(delta, delta1):
    """Computes Q for quartic formula.

    https://en.wikipedia.org/wiki/Quartic_function

    Args:
        delta (tf.tensor): discriminant of quartic formula
        delta1 (tf.tensor): delta1 for quartic formula

    Returns:
    	Q (tf.tensor): Q of the quartic formula
    """
    Q = ((delta1 + tf.sqrt(-27.0*delta))/2.0)**(1.0 / 3.0)
    return Q

def quartic_S(a, p, Q, delta0):
    """Computes S for quartic formula.

    https://en.wikipedia.org/wiki/Quartic_function

    Args:
        a (tf.tensor): coefficients for x^4
        p (tf.tensor): p of the quartic formula
        Q (tf.tensor): Q of the quartic formula
        delta0 (tf.tensor): delta0 of the quartic formula

    Returns:
    	p (list): roots of p(x)
    """
    S = 0.5*tf.sqrt(-(2.0/3.0)*p + (1.0 / (3.0*a))*(Q + (delta0 / Q)))
    return S

def quartic_p(a, b, c, d, e):
    """Computes p for quartic formula.

    https://en.wikipedia.org/wiki/Quartic_function

    f(x) = ax^4 + bx^3 + cx^2 + dx + e = 0

    Args:
        a (tf.tensor): coefficients for x^4
        b (tf.tensor): coefficients for x^3
        c (tf.tensor): coefficients for x^2
        d (tf.tensor): coefficients for x^1
        e (tf.tensor): coefficients for x^0

    Returns:
    	p (list): roots of p(x)
    """
    p = (8.0*a*c - 3.0*(b**2)) / (8.0*(a**2))
    return p

def quartic_q(a, b, c, d, e):
    """Computes q for quartic formula.

    https://en.wikipedia.org/wiki/Quartic_function

    f(x) = ax^4 + bx^3 + cx^2 + dx + e = 0

    Args:
        a (tf.tensor): coefficients for x^4
        b (tf.tensor): coefficients for x^3
        c (tf.tensor): coefficients for x^2
        d (tf.tensor): coefficients for x^1
        e (tf.tensor): coefficients for x^0

    Returns:
    	q (list): roots of p(x)
    """
    q = ((b**3) - 4*a*b*c + 8.0*(a**2)*d) / (8.0*(a**3))
    return q

def quartic_roots(a, b, c, d, e):
    """Compute the roots of a quartic polynomial using the quartic formula.

    https://en.wikipedia.org/wiki/Quartic_function

    f(x) = ax^4 + bx^3 + cx^2 + dx + e = 0

    Args:
        a (tf.tensor): coefficients for x^4
        b (tf.tensor): coefficients for x^3
        c (tf.tensor): coefficients for x^2
        d (tf.tensor): coefficients for x^1
        e (tf.tensor): coefficients for x^0

    Returns:
    	roots (list): roots of f(x)
    """

    delta = tf.cast(quartic_delta(a, b, c, d, e), tf.complex128)
    delta0 = tf.cast(quartic_delta0(a, b, c, d, e), tf.complex128)
    delta1 = tf.cast(quartic_delta1(a, b, c, d, e), tf.complex128)
        
    p = tf.cast(quartic_p(a, b, c, d, e), tf.complex128)
    q = tf.cast(quartic_q(a, b, c, d, e), tf.complex128)

    a_tfc128 = tf.cast(a, tf.complex128)
    b_tfc128 = tf.cast(b, tf.complex128)
        
    Q = quartic_Q(delta, delta1)
    S = quartic_S(tf.cast(a_tfc128, tf.complex128), p, Q, delta0)
        
    x1 = -(b_tfc128 / (4.0*a_tfc128)) - S + 0.5*tf.sqrt(-4*(S**2) - 2*p + (q / S))
    x2 = -(b_tfc128 / (4.0*a_tfc128)) - S - 0.5*tf.sqrt(-4*(S**2) - 2*p + (q / S))
    
    x3 = -(b_tfc128 / (4.0*a_tfc128)) + S + 0.5*tf.sqrt(-4*(S**2) - 2*p - (q / S))
    x4 = -(b_tfc128 / (4.0*a_tfc128)) + S - 0.5*tf.sqrt(-4*(S**2) - 2*p - (q / S))
    roots = [x1, x2, x3, x4]
    return roots

