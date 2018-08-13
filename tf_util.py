import tensorflow as tf
import numpy as np
from flows import GP_Layer, AR_Layer, VAR_Layer

# edit 2

def construct_flow(flow_dict, D_Z, T):
    latent_layers = construct_latent_dynamics(flow_dict, D_Z, T);
    time_invariant_layers = construct_time_invariant_flow(flow_dict, D_Z, T);

    layers = latent_layers + time_invariant_layers;
    nlayers = len(layers);

    num_theta_params = 0;
    for i in range(nlayers):
        layer = layers[i];
        print(i, layer);
        num_theta_params += count_layer_params(layer);

    Z0 = tf.placeholder(tf.float64, shape=(None, None, D_Z, None), name='Z0');
    K = tf.shape(Z0)[0];
    M = tf.shape(Z0)[1];

    p0 = tf.reduce_prod(tf.exp((-tf.square(Z0))/2.0)/np.sqrt(2.0*np.pi), axis=[2,3]); 
    base_log_q_x = tf.log(p0[:,:]);
    Z_AR = Z0;
    return layers, Z0, Z_AR, base_log_q_x, num_theta_params;


def construct_latent_dynamics(flow_dict, D_Z, T):
    latent_dynamics = flow_dict['latent_dynamics'];

    if (latent_dynamics is None):
        return [];

    inits = flow_dict['inits'];
    if ('lock' in flow_dict):
        lock = flow_dict['lock'];
    else:
        lock = False;

    if (latent_dynamics == 'GP'):
        layer = GP_Layer('GP_Layer', dim=D_Z, \
                         inits=inits, lock=lock);

    elif (latent_dynamics == 'AR'):
        param_init = {'alpha_init':inits['alpha_init'], 'sigma_init':inits['sigma_init']};
        layer = AR_Layer('AR_Layer', dim=D_Z, T=T, P=flow_dict['P'], \
                      inits=inits, lock=lock);

    elif (latent_dynamics == 'VAR'):
        param_init = {'A_init':inits['A_init'], 'sigma_init':inits['sigma_init']};
        layer = VAR_Layer('VAR_Layer', dim=D_Z, T=T, P=flow_dict['P'], \
                      inits=inits, lock=lock);

    else:
        raise NotImplementedError();

    return [layer];


def construct_time_invariant_flow(flow_dict, D_Z, T):
    layer_ind = 1;
    layers = [];
    TIF_flow_type = flow_dict['TIF_flow_type'];
    repeats = flow_dict['repeats'];

    if (TIF_flow_type == 'ScalarFlowLayer'):
        flow_class = ElemMultLayer;
        name_prefix = 'ScalarFlow_Layer';

    elif (TIF_flow_type == 'FullyConnectedFlowLayer'):
        flow_class = FullyConnectedFlowLayer;
        name_prefix = FullyConnectedFlow_Layer;

    elif (TIF_flow_type == 'AffineFlowLayer'):
        flow_class = AffineFlowLayer;
        name_prefix = 'AffineFlow_Layer';

    elif (TIF_flow_type == 'StructuredSpinnerLayer'):
        flow_class = StructuredSpinnerLayer
        name_prefix = 'StructuredSpinner_Layer';

    elif (TIF_flow_type == 'StructuredSpinnerTanhLayer'):
        flow_class = StructuredSpinnerTanhLayer
        name_prefix = 'StructuredSpinnerTanh_Layer';

    elif (TIF_flow_type == 'PlanarFlowLayer'):
        flow_class = PlanarFlowLayer
        name_prefix = 'PlanarFlow_Layer';

    elif (TIF_flow_type == 'RadialFlowLayer'):
        flow_class = RadialFlowLayer
        name_prefix = 'RadialFlow_Layer';

    elif (TIF_flow_type == 'TanhLayer'):
        flow_class = TanhLayer;
        name_prefix = 'Tanh_Layer';

    else:
        raise NotImplementedError();

    for i in range(repeats):
        layers.append(flow_class('%s%d' % (name_prefix, layer_ind), D_Z));
        layer_ind += 1;
        
    return layers;


def declare_theta(flow_layers):
    L_flow = len(flow_layers);
    theta =[];
    for i in range(L_flow):
        layer = flow_layers[i];
        layer_name, param_names, param_dims, initializers, lock = layer.get_layer_info();
        nparams = len(param_names);
        layer_i_params = [];
        for j in range(nparams):
            if (lock):
                param_ij = initializers[j];
            else:
                if (isinstance(initializers[j], tf.Tensor)):
                    print('yep');
                    print(initializers[j]);
                    param_ij = tf.get_variable(layer_name+'_'+param_names[j], \
                                               dtype=tf.float64, \
                                               initializer=initializers[j]);
                else:
                    print('nope');
                    print(initializers[j]);
                    param_ij = tf.get_variable(layer_name+'_'+param_names[j], shape=param_dims[j], \
                                               dtype=tf.float64, \
                                               initializer=initializers[j]);
            layer_i_params.append(param_ij);
        theta.append(layer_i_params);
    return theta;

def connect_flow(Z, layers, theta, ts=None):
    Z_shape = tf.shape(Z);
    K = Z_shape[0];
    M = Z_shape[1];
    D_Z = Z_shape[2];
    T = Z_shape[3];

    sum_log_det_jacobians = tf.zeros((K,M), dtype=tf.float64);
    nlayers = len(layers);
    Z_by_layer = [];
    Z_by_layer.append(Z);
    print('zshapes in');
    print('connect flow');
    for i in range(nlayers):
        print(Z.shape);
        layer = layers[i];
        print(i, layer.name);
        theta_layer = theta[i];
        layer.connect_parameter_network(theta_layer);
        if (isinstance(layer, GP_Layer) or isinstance(layer, GP_EP_CondRegLayer)):
            Z, sum_log_det_jacobians = layer.forward_and_jacobian(Z, sum_log_det_jacobians, ts);
        else:
            Z, sum_log_det_jacobians = layer.forward_and_jacobian(Z, sum_log_det_jacobians);
        Z_by_layer.append(Z);
    print(Z.shape);
    return Z, sum_log_det_jacobians, Z_by_layer;

def count_layer_params(layer):
    num_params = 0;
    name, param_names, dims, _, _ = layer.get_layer_info();
    nparams = len(dims);
    for j in range(nparams):
        num_params += np.prod(dims[j]);
    return num_params;


def count_params(all_params):
    nparams = len(all_params);
    nparam_vals = 0;
    for i in range(nparams):
        param = all_params[i];
        param_shape = tuple(param.get_shape().as_list());
        nparam_vals += np.prod(param_shape);
    return nparam_vals;
