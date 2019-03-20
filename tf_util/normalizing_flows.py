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
import scipy.linalg

DTYPE = tf.float64


def get_flow_class(flow_name):
    if flow_name == "AffineFlow":
        return AffineFlow
    elif flow_name == "CholProdFlow":
        return CholProdFlow
    elif flow_name == "ElemMultFlow":
        return ElemMultFlow
    elif flow_name == "ExpFlow":
        return ExpFlow
    elif flow_name == "IntervalFlow":
        return IntervalFlow
    elif flow_name == "PlanarFlow":
        return PlanarFlow
    elif flow_name == "RadialFlow":
        return RadialFlow
    elif flow_name == "ShiftFlow":
        return ShiftFlow
    elif flow_name == "SimplexBijectionFlow":
        return SimplexBijectionFlow
    elif flow_name == "SoftPlusFlow":
        return SoftPlusFlow
    elif flow_name == "StructuredSpinnerFlow":
        return StructuredSpinnerFlow
    elif flow_name == "StructuredSpinnerTanhFlow":
        return StructuredSpinnerTanhFlow
    elif flow_name == "TanhFlow":
        return TanhFlow
    elif flow_name == "RealNVP":
        return RealNVP
    else:
        raise NotImplementedError()


def get_num_flow_params(flow_class, D, opt_params={}):
    if flow_class == AffineFlow:
        return D * (D + 1)
    elif flow_class == CholProdFlow:
        raise NotImplementedError()
    elif flow_class == ElemMultFlow:
        return D
    elif flow_class == ExpFlow:
        return 0
    elif flow_class == IntervalFlow:
        return 2
    elif flow_class == PlanarFlow:
        return 2 * D + 1
    elif flow_class == RadialFlow:
        return D + 2
    elif flow_class == ShiftFlow:
        return D
    elif flow_class == SimplexBijectionFlow:
        return 0
    elif flow_class == SoftPlusFlow:
        return 0
    elif flow_class == StructuredSpinnerFlow:
        raise NotImplementedError()
    elif flow_class == StructuredSpinnerTanhFlow:
        raise NotImplementedError()
    elif flow_class == TanhFlow:
        return 0
    elif flow_class == RealNVP:
        if ('num_masks' in opt_params.keys()):
            num_masks = opt_params['num_masks']
        else:
            num_masks = 1
        if ('nlayers' in opt_params.keys()):
            nlayers = opt_params['nlayers']
        else:
            nlayers = 1
        if ('upl' in opt_params.keys()):
            upl = opt_params['upl']
        else:
            upl = 1
        return get_real_nvp_num_params(D, num_masks, nlayers, upl)
    else:
        raise NotImplementedError()


def get_flow_out_dim(flow_class, dim):
    if flow_class == AffineFlow:
        return dim
    elif flow_class == CholProdFlow:
        raise NotImplementedError()
    elif flow_class == ElemMultFlow:
        return dim
    elif flow_class == ExpFlow:
        return dim
    elif flow_class == IntervalFlow:
        return dim
    elif flow_class == PlanarFlow:
        return dim
    elif flow_class == RadialFlow:
        return dim
    elif flow_class == ShiftFlow:
        return dim
    elif flow_class == SimplexBijectionFlow:
        return dim + 1
    elif flow_class == SoftPlusFlow:
        return dim
    elif flow_class == StructuredSpinnerFlow:
        raise NotImplementedError()
    elif flow_class == StructuredSpinnerTanhFlow:
        raise NotImplementedError()
    elif flow_class == TanhFlow:
        return dim
    elif flow_class == RealNVP:
        return dim
    else:
        raise NotImplementedError()


def get_flow_param_inits(flow_class, D, opt_params={}):
    if flow_class == AffineFlow:
        return [tf.glorot_uniform_initializer()], [D * (D + 1)]
    elif flow_class == CholProdFlow:
        raise NotImplementedError()
    elif flow_class == ElemMultFlow:
        return [tf.glorot_uniform_initializer()], [D]
    elif flow_class == ExpFlow:
        return [None], [0]
    elif flow_class == IntervalFlow:
        raise NotImplementedError()
    elif flow_class == PlanarFlow:
        inits = [
            tf.constant(np.zeros(D)),
            tf.glorot_uniform_initializer(),
            tf.constant(np.zeros(1)),
        ]
        dims = [D, D, 1]
        return inits, dims
    elif flow_class == RadialFlow:
        raise NotImplementedError()
    elif flow_class == ShiftFlow:
        return [tf.glorot_uniform_initializer()], [D]
    elif flow_class == SimplexBijectionFlow:
        return [None], [0]
    elif flow_class == SoftPlusFlow:
        return [None], [0]
    elif flow_class == StructuredSpinnerFlow:
        raise NotImplementedError()
    elif flow_class == StructuredSpinnerTanhFlow:
        raise NotImplementedError()
    elif flow_class == TanhFlow:
        return [None], [0]
    elif flow_class == RealNVP:
        num_masks = opt_params['num_masks']
        nlayers = opt_params['nlayers']
        upl = opt_params['upl']
        num_params = get_real_nvp_num_params(D, num_masks, nlayers, upl)
        return [tf.glorot_uniform_initializer()], [num_params]
    else:
        raise NotImplementedError()


def get_density_network_inits(arch_dict):
    inits_by_layer = []
    dims_by_layer = []
    D = arch_dict["D"]

    if arch_dict["latent_dynamics"] is not None:
        raise NotImplementedError()

    if arch_dict["mult_and_shift"] == "pre":
        em_inits, em_dims = get_flow_param_inits(ElemMultFlow, arch_dict["D"])
        inits_by_layer.append(em_inits)
        dims_by_layer.append(em_dims)

        shift_inits, shift_dims = get_flow_param_inits(ShiftFlow, arch_dict["D"])
        inits_by_layer.append(em_inits)
        dims_by_layer.append(em_dims)

    TIF_flow = get_flow_class(arch_dict["TIF_flow_type"])
    for i in range(arch_dict["repeats"]):
        if (TIF_flow == RealNVP):
            inits, dims = get_flow_param_inits(TIF_flow, D, arch_dict['real_nvp_arch'])
        else:
            inits, dims = get_flow_param_inits(TIF_flow, D)
        inits_by_layer.append(inits)
        dims_by_layer.append(dims)

    if arch_dict["mult_and_shift"] == "post":
        em_inits, em_dims = get_flow_param_inits(ElemMultFlow, arch_dict["D"])
        inits_by_layer.append(em_inits)
        dims_by_layer.append(em_dims)

        shift_inits, shift_dims = get_flow_param_inits(ShiftFlow, arch_dict["D"])
        inits_by_layer.append(em_inits)
        dims_by_layer.append(em_dims)

    return inits_by_layer, dims_by_layer


# Time invariant flows
class NormFlow:
    """Base class for normalizing flow layer.

    # Attributes
        self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                 K parameterizations of the layer. 
        self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
        self.dim (Dimension): Dimensionality of the normalizing flow.
        self.num_params (Dimension): Total number of paramters of flow.
    """

    def __init__(self, params, inputs):
        """Norm flow layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        self.params = params
        self.inputs = inputs
        if isinstance(inputs, tf.Tensor):
            self.dim = inputs.shape[2]
        else:
            self.dim = 0
        if isinstance(params, tf.Tensor):
            self.num_params = self.params.shape[1]
        else:
            self.num_params = 0

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        raise NotImplementedError(str(type(self)))


class AffineFlow(NormFlow):
    """Affine flow layer.

    f(z) = Az + b
    log_det_jac = log(|det(A)|)

    # Attributes
        self.A (tf.tensor): [K, self.dim, self.dim] The $$A$$ parameter.
        self.b (tf.tensor): [K, self.dim, self.dim] The $$b$$ parameter.

    """

    def __init__(self, params, inputs):
        """Affine flow layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "AffineFlow"
        self.A = params[:, : (tf.square(self.dim))]
        self.b = params[:, (tf.square(self.dim)) :]

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = tf.transpose(self.inputs, [0, 2, 1])  # make [K, dim, n]
        A_shape = tf.shape(self.A)
        K = A_shape[0]
        A = tf.reshape(self.A, [K, self.dim, self.dim])  # [K, dim, dim]
        b = tf.expand_dims(self.b, 2)  # [K, dim, 1]

        # compute the log abs det jac
        log_det_jac = tf.log(tf.abs(tf.matrix_determinant(A)))
        log_det_jac = tf.tile(
            tf.expand_dims(log_det_jac, 1), [1, tf.shape(self.inputs)[1]]
        )

        out = tf.transpose(tf.matmul(A, z) + b, [0, 2, 1])

        return out, log_det_jac


class CholProdFlow(NormFlow):
    """Cholesky product flow layer.

    f(z) = L(z)L(z)^T
    where
    L(z) = [e^z1,    0,   .., .., 0]
           [z2  , e^z3,    0, .., 0]
           [z4  ,   z5, e^z6, .., 0]
           [..  ,   ..,            ]

    log_det_jac = (See Docs)

    # Attributes
        self.A (tf.tensor): [K, self.dim, self.dim] The $$A$$ parameter.
        self.b (tf.tensor): [K, self.dim, self.dim] The $$b$$ parameter.

    """

    def __init__(self, params, inputs):
        """Affine flow layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "CholProdFlow"
        self.diag_eps = 1e-16

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs
        K = tf.shape(z)[0]
        M = tf.shape(z)[1]
        z_dim = tf.shape(z)[2]

        _L = tf.contrib.distributions.fill_triangular(z)
        z_diag_elems = tf.matrix_diag_part(_L)
        L = tf.contrib.distributions.matrix_diag_transform(_L, tf.exp)
        D = tf.shape(L)[2]
        float_D = tf.cast(D, tf.float64)
        L_diag_elems = tf.matrix_diag_part(L)

        Sigma = tf.matmul(L, tf.transpose(L, [0, 1, 3, 2]))
        # diag_boost = self.diag_eps * tf.eye(D, batch_shape=[K, M], dtype=tf.float64)
        # LLT = LLT + diag_boost
        vecSigma = tf.reshape(Sigma, [K, M, D * D])

        L_log_det = tf.reduce_sum(z_diag_elems, 2)

        var = tf.cast(tf.range(1, sqrtD + 1), tf.float64)
        diag_facs = tf.expand_dims(tf.expand_dims(floatD - var + 1.0, 0), 0)

        chol_prod_log_det = float_D * np.log(2.0) + tf.reduce_sum(
            tf.multiply(diag_facs, tf.log(L_diag_elems)), 2
        )

        log_det_jac = L_log_det + chol_prod_log_det

        return vecSigma, log_det_jac


class ElemMultFlow(NormFlow):
    """Elementwise multiplication layer.

    f(z) = a * z
    log_det_jac = sum_i log(abs(a_i))

    # Attributes
        self.a (tf.tensor): [K, self.dim] The $$a$$ parameter.

    """

    def __init__(self, params, inputs):
        """Elementwise multiplication layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "ElemMultFlow"
        self.a = params

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs
        n = tf.shape(z)[1]

        # compute the log abs det jacobian
        log_det_jac = tf.reduce_sum(tf.log(tf.abs(self.a)), 1)
        log_det_jac = tf.tile(tf.expand_dims(log_det_jac, 1), [1, n])

        # compute output
        z = tf.multiply(z, tf.expand_dims(self.a, 1))
        return z, log_det_jac


class ExpFlow(NormFlow):
    """Elementwise multiplication layer.

    f(z) = exp(z)
    log_det_jac = sum_i z_i

    """

    def __init__(self, params, inputs):
        """Exp layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "ExpFlow"

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs
        n = tf.shape(z)[1]

        # compute the log abs det jacobian
        log_det_jac = tf.reduce_sum(z, 2)

        # compute output
        z = tf.exp(z)
        return z, log_det_jac


class IntervalFlow(NormFlow):
    """Elementwise multiplication layer.

    m = (b - a)/2
    c = (a + b)/2

    f(z) = m tanh(z) + c
    log_det_jac = m (1 - tanh^2(z))

    """

    def __init__(self, params, inputs, a, b):
        """Exp layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        for i in range(a.shape[0]):
            assert a[i] < b[i]

        super().__init__(params, inputs)
        self.name = "IntervalFlow"
        self.a = a
        self.b = b

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs

        m = np.expand_dims(np.expand_dims((self.b - self.a) / 2.0, 0), 0)
        c = np.expand_dims(np.expand_dims((self.a + self.b) / 2.0, 0), 0)

        tanh_z = tf.tanh(z)

        out = tf.multiply(m, tanh_z) + c
        log_det_jac = tf.reduce_sum(np.log(m) + tf.log(1.0 - tf.square(tanh_z)), 2)
        return out, log_det_jac


class PlanarFlow(NormFlow):
    """Planar flow layer

    Implements the function [imsert some tex]

    # Attributes
        self._u (tf.tensor): [K, self.dim] u parameter from self.params.
        self.w (tf.tensor): [K, self.dim] w parameter.
        self.b (tf.tensor): [K, 1] b paramter
        self.u (tf.tensor): [K, self.dim] modified u paramter to ensure invertibility

    """

    def __init__(self, params, inputs):
        """Planar flow layer constructor.

        Sets u, w, b given params, ensuring that the flow is invertible.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "PlanarFlow"
        self._u = params[:, : self.dim]
        self.w = params[:, self.dim : (2 * self.dim)]
        self.b = params[:, 2 * self.dim :]

        # For tanh nonlinearity, enfoces w^\topz >= -1,
        # ensuring invertibility.
        wdotu = tf.matmul(tf.expand_dims(self.w, 1), tf.expand_dims(self._u, 2))[
            :, :, 0
        ]
        m_wdotu = -1 + tf.log(1 + tf.exp(wdotu))
        uhat_numer = tf.multiply(m_wdotu - wdotu, self.w)
        uhat_denom = tf.expand_dims(tf.reduce_sum(tf.square(self.w), 1), 1)

        self.u = self._u + tf.divide(uhat_numer, uhat_denom)

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs
        # derivative of tanh(x) is (1-tanh^2(x))
        wdotz = tf.matmul(z, tf.expand_dims(self.w, 2))  # [K,n,1]
        h = tf.tanh(wdotz + tf.expand_dims(self.b, 1))  # [K,n,1]
        hprime = 1.0 - tf.square(h)  # [K,n,1]
        phi = tf.matmul(hprime, tf.expand_dims(self.w, 1))  # [K,n,D]

        # compute the log abs det jacobian
        udotphi = tf.matmul(phi, tf.expand_dims(self.u, 2))  # [K,n,1]
        log_det_jacobian = tf.log(tf.abs(1.0 + udotphi))[:, :, 0]  # [K,n]

        # compute output
        f_z = z + tf.matmul(h, tf.expand_dims(self.u, 1))  # [K,n,D]

        return f_z, log_det_jacobian


class RadialFlow(NormFlow):
    """Radial flow layer.

    f(z) = z + beta h(alpha, r)(z-z0)
    log_det_jac = [1+betah(alpha,r)]^(d-1) [1+betah(alpha,r) + betah'(alpha,r)r]

    # Attributes
        self.alpha (tf.tensor): [K, 1] The $$\alpha$$ parameter.
        self.beta (tf.tensor): [K, 1] The $$\beta$$ parameter.
        self.z0 (tf.tensor): [K, self.dim] The $$z_0$$ parameter.

    """

    def __init__(self, params, inputs):
        """Radial flow layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "RadialFlow"
        self.alpha = params[:, 0:1]
        self._beta = params[:, 1:2]
        self.z0 = params[:, 2:]

        # reparameterize to ensure invertibility
        m_beta = tf.log(1.0 + tf.exp(self._beta))
        self.beta = -self.alpha + m_beta

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs
        alpha = tf.expand_dims(self.alpha, 1)
        beta = tf.expand_dims(self.beta, 1)
        z0 = tf.expand_dims(self.z0, 1)

        r = tf.expand_dims(tf.linalg.norm(z, axis=2), 2)
        h = 1.0 / (alpha + r)
        hprime = -1.0 / tf.square(alpha + r)

        out = z + beta * h * (z - z0)

        log_det_jac1 = (tf.cast(self.dim, dtype=DTYPE) - 1.0) * tf.log((1.0 + beta * h))
        log_det_jac2 = tf.log(1.0 + beta * h + beta * hprime * r)
        log_det_jac = log_det_jac1 + log_det_jac2

        return out, log_det_jac[:, :, 0]


class ShiftFlow(NormFlow):
    """Addition layer.

    Shifts D-dimensional data with a D-dimensional offset

    # Attributes
        self.b (tf.tensor): [K, self.dim] The $$b$$ parameter.

    """

    def __init__(self, params, inputs):
        """Elementwise multiplication layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "ElemMultFlow"
        self.b = params

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs
        n = tf.shape(z)[1]

        # compute the log abs det jacobian
        log_det_jac = tf.zeros_like(z[:, :, 0])

        # compute output
        z = z + tf.expand_dims(self.b, 1)
        return z, log_det_jac


class SimplexBijectionFlow(NormFlow):
    """Simplex bijection layer.

    out = (e^z1 / (sum i e^zi + 1), .., e^z_d-1 / (sum i e^zi + 1), 1 / (sum i e^zi + 1))
    log_det_jac = log(1 - (sum i e^zi / (sum i e^zi+1)) - D log(sum i e^zi + 1) + sum i zi

    """

    def __init__(self, params, inputs):
        """Simplex bijection layer layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "SimplexBijectionFlow"

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs
        ex = tf.exp(z)
        sum_ex = tf.reduce_sum(ex, 2)
        den = sum_ex + 1.0
        log_det_jac = (
            tf.log(1.0 - (sum_ex / den))
            - tf.cast(self.dim, tf.float64) * tf.log(den)
            + tf.reduce_sum(z, 2)
        )
        out = tf.concat(
            (ex / tf.expand_dims(den, 2), 1.0 / tf.expand_dims(den, 2)), axis=2
        )
        return out, log_det_jac


class SoftPlusFlow(NormFlow):
    def __init__(self, params, inputs):
        super().__init__(params, inputs)
        self.name = "SoftPlusFlow"

    def forward_and_jacobian(self,):
        z = self.inputs
        z_out = tf.log(1 + tf.exp(z))
        jacobian = tf.divide(1.0, 1.0 + tf.exp(-z))
        log_det_jac = tf.reduce_sum(tf.log(jacobian), axis=2)
        return z_out, log_det_jac


class StructuredSpinnerFlow(NormFlow):
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim
        self.param_names = ["d1", "d2", "d3", "b"]
        self.d1 = None
        self.d2 = None
        self.d3 = None
        self.b = None
        self.lock = False

    def get_layer_info(self,):
        d1_dim = (self.dim, 1)
        d2_dim = (self.dim, 1)
        d3_dim = (self.dim, 1)
        b_dim = (self.dim, 1)
        dims = [d1_dim, d2_dim, d3_dim, b_dim]
        initializers = [
            tf.glorot_uniform_initializer(),
            tf.glorot_uniform_initializer(),
            tf.glorot_uniform_initializer(),
            tf.glorot_uniform_initializer(),
        ]
        return self.name, self.param_names, dims, initializers, self.lock

    def get_params(self,):
        return self.d1, self.d2, self.d3, self.b

    def connect_parameter_network(self, theta_layer):
        self.d1, self.d2, self.d3, self.b = theta_layer
        self.param_network = len(self.d1.shape) == 3
        return None

    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        K, M, D, T = tensor4_shape(z)
        D_np = z.get_shape().as_list()[2]
        Hmat = scipy.linalg.hadamard(D_np, dtype=np.float64) / np.sqrt(D_np)
        H = tf.constant(Hmat, tf.float64)
        d1, d2, d3, b = self.get_params()
        if not self.param_network:
            D1 = tf.matrix_diag(d1[:, 0])
            D2 = tf.matrix_diag(d2[:, 0])
            D3 = tf.matrix_diag(d3[:, 0])
        else:
            H = tf.tile(tf.expand_dims(H, 0), [K, 1, 1])
            D1 = tf.matrix_diag(d1[:, :, 0])
            D2 = tf.matrix_diag(d2[:, :, 0])
            D3 = tf.matrix_diag(d3[:, :, 0])

        A = tf.matmul(H, tf.matmul(D3, tf.matmul(H, tf.matmul(D2, tf.matmul(H, D1)))))
        if not self.param_network:
            b = tf.expand_dims(tf.expand_dims(b, 0), 0)
            z = tf.tensordot(A, z, [[1], [2]])
            z = tf.transpose(z, [1, 2, 0, 3]) + b
            log_det_A = (
                tf.log(tf.abs(tf.reduce_prod(d1)))
                + tf.log(tf.abs(tf.reduce_prod(d2)))
                + tf.log(tf.abs(tf.reduce_prod(d3)))
            )
            log_det_jacobian = tf.multiply(
                log_det_A, tf.ones((K, M), dtype=tf.float64)
            ) + 0.0 * tf.reduce_sum(b)
        else:
            z_KD_MTvec = tf.reshape(tf.transpose(z, [0, 2, 1, 3]), [K, D, M * T])
            Az_KD_MTvec = tf.matmul(A, z_KD_MTvec)
            Az = tf.transpose(tf.reshape(Az_KD_MTvec, [K, D, M, T]), [0, 2, 1, 3])
            z = Az + tf.expand_dims(b, 1)
            log_det_jacobian = tf.log(tf.abs(tf.matrix_determinant(A)))
            log_det_jacobian = tf.tile(
                tf.expand_dims(log_det_jacobian, 1), [1, M]
            ) + 0.0 * tf.reduce_sum(b)
        sum_log_det_jacobians += tf.cast(T, tf.float64) * log_det_jacobian
        return z, sum_log_det_jacobians


class StructuredSpinnerTanhFlow(StructuredSpinnerFlow):
    def forward_and_jacobian(self, z, sum_log_det_jacobians):
        z, sum_log_det_jacobians = super().forward_and_jacobian(
            z, sum_log_det_jacobians
        )
        z_out = tf.tanh(z)
        log_det_jacobian = tf.reduce_sum(tf.log(1.0 - (z_out ** 2)), [2, 3])
        sum_log_det_jacobians += log_det_jacobian
        return z_out, sum_log_det_jacobians


class TanhFlow(NormFlow):
    """Tanh layer.

    f(z) = tanh(z)
    log_det_jac = sum_i log(abs(1 - tanh^2(z_i)))

    """

    def __init__(self, params, inputs):
        """Tanh layer constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "TanhFlow"

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs
        n = tf.shape(z)[1]

        tanh_z = tf.tanh(z)

        # compute the log abs det jacobian
        log_det_jac = tf.reduce_sum(tf.log(1.0 - tf.square(tanh_z)), 2)

        # compute output
        out = tanh_z

        return out, log_det_jac


class RealNVP(NormFlow):
    """Real NVP

    Implements the function [imsert some tex]

    # Attributes
        self.params (tf.tensor): [K, self.dim] u parameter from self.params.

    """

    def __init__(self, params, inputs, num_masks, nlayers, upl):
        """Real NVP constructor.

        # Arguments 
            self.params (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer. 
            self.inputs (tf.tensor): [K, batch_size, self.dim] layer input.
    
        """
        super().__init__(params, inputs)
        self.name = "RealNVP"
        self.num_masks = num_masks
        self.nlayers = nlayers
        self.upl = upl
        self.s_param_ind_by_mask = []
        self.t_param_ind_by_mask = []

        # get list of masks
        masks = get_real_nvp_mask_list(self.dim, self.num_masks)
        # add two broadcasting dimensions to the front
        for i in range(self.num_masks):
            masks[i] = np.expand_dims(np.expand_dims(masks[i], 0), 0)
        self.masks = masks

    def forward_and_jacobian(self,):
        """Perform the flow operation and compute the log-abs-det-jac.

        # Returns 
            f_z (tf.tensor): [K, batch_size, self.dim] Result of operation. 
            log_det_jacobian (tf.tensor): [K, batch_size] Log absolute
                value of the determinant of the jacobian of the mappings.
    
        """
        z = self.inputs

        # construct functions for each mask
        param_ind = 0
        z_i = z
        log_det_jacs = []
        for i in range(self.num_masks):
            mask_i = self.masks[i]

            self.s_param_ind_by_mask.append(param_ind)
            s, param_ind = nvp_neural_network_tf(z_i, self.params, mask_i, \
                                                 self.nlayers, self.upl, \
                                                 param_ind)

            self.t_param_ind_by_mask.append(param_ind)
            t, param_ind = nvp_neural_network_tf(z_i, self.params, mask_i, \
                                                 self.nlayers, self.upl, \
                                                 param_ind)

            z_i = (mask_i)*z_i + (1-mask_i)*(z_i*tf.exp(s) + t)

            log_det_jac = tf.reduce_sum((1-mask_i)*s, 2)
            log_det_jacs.append(log_det_jac)

        f_z = z_i
        log_det_jacobian = sum(log_det_jacs)

        return f_z, log_det_jacobian

    def inverse(self, z):
        """Invert sample z to random variable w.

        # Arguments 
            z (tf.tensor): [K, self.num_params] Tensor containing 
                                     K parameterizations of the layer.
        # Returns 
            f_inv_z (tf.tensor): [K, batch_size, self.dim] Result of operation

        """ 
        if len(self.s_param_ind_by_mask)==0:
            print('Error: must call RealNVP.forward_and_jacobian() before RealNVP.inverse().')
            raise EnvironmentError

        # construct functions for each mask
        z_i = z
        for i in range(self.num_masks-1, -1, -1):
            mask_i = self.masks[i]

            s_param_ind = self.s_param_ind_by_mask[i]
            s, _ = nvp_neural_network_tf(z_i, self.params, mask_i, \
                                         self.nlayers, self.upl, \
                                         s_param_ind)

            t_param_ind = self.t_param_ind_by_mask[i]
            t, _ = nvp_neural_network_tf(z_i, self.params, mask_i, \
                                         self.nlayers, self.upl, \
                                         t_param_ind)

            z_i = (mask_i)*z_i + (1-mask_i)*(z_i - t)*tf.exp(-s)

        f_inv_z = z_i

        return f_inv_z



def get_real_nvp_mask(D, f, first_on=True):
    assert(f <= (D//2))
    wavelength = D // f
    # make sure that the wavelength is even
    if (np.mod(wavelength, 2) == 1):
        wavelength += 1
    half_wl = wavelength // 2
    mask = np.zeros((D,), np.float64)
    ind = 0
    if first_on:
        bit = 1
    else:
        bit = 0

    for i in range(f+1):
        if (ind+half_wl > D):
            mask[ind:] = bit
            break
        else:
            mask[ind:(ind+half_wl)] = bit

        if (ind+wavelength > D):
            mask[(ind+half_wl):] = 1-bit
            break
        else:
            mask[(ind+half_wl):(ind+wavelength)] = 1-bit

        ind += wavelength

    return mask

def get_real_nvp_mask_list(D, num_masks):
    masks = []
    fs = []
    max_freq_depth = num_masks//4 + 1
    f_ind = 1
    for i in range(max_freq_depth):
        fs += [f_ind, f_ind, D//(2*f_ind), D//(2*f_ind)]
        f_ind *= 2
    firstOn = True
    for i in range(num_masks):
        mask_i = get_real_nvp_mask(D, fs[i], firstOn)
        masks.append(mask_i)
        firstOn = not firstOn

    return masks

def get_real_nvp_num_params(D, num_masks, nlayers, upl):
    return 2*num_masks*(2*D*upl + D + upl + (nlayers-1)*(upl+1)*upl)


def nvp_neural_network_np(z, params, mask_i, nlayers, upl, param_ind):
    D = z.shape[0]

    A_1 = np.reshape(params[param_ind:(param_ind+D*upl)], (upl, D))
    param_ind += D*upl
    b_1 = params[param_ind:(param_ind+upl)]
    param_ind += upl
    z_l = np.tanh(np.dot(A_1,(z*mask_i)) + b_1)

    for l in range(1,nlayers):
        A_l = np.reshape(params[param_ind:(param_ind+upl*upl)], (upl, upl))
        param_ind += upl*upl
        b_l = params[param_ind:(param_ind+upl)]
        param_ind += upl
        z_l = np.tanh(np.dot(A_l,z_l) + b_l)

    A_out = np.reshape(params[param_ind:(param_ind+upl*D)], (D, upl))
    param_ind += upl*D
    b_out = params[param_ind:(param_ind+D)]
    param_ind += D
    z_out = np.dot(A_out, z_l) + b_out

    return z_out, param_ind



def nvp_neural_network_tf(z, params, mask_i, nlayers, upl, param_ind):
    K = tf.shape(params)[0]
    n = tf.shape(params)[1]
    D = tf.shape(z)[2]

    # mask and transpose
    z = tf.transpose(z*mask_i, [0, 2, 1]) # [K, D, n]

    A_1 = tf.reshape(params[:, param_ind:(param_ind+D*upl)], (K, upl, D))
    param_ind += D*upl
    b_1 = tf.expand_dims(params[:, param_ind:(param_ind+upl)], 2)
    param_ind += upl
    z_l = tf.tanh(tf.matmul(A_1, z) + b_1)

    for l in range(1,nlayers):
        A_l = tf.reshape(params[:, param_ind:(param_ind+upl*upl)], (K, upl, upl))
        param_ind += upl*upl
        b_l = tf.expand_dims(params[:, param_ind:(param_ind+upl)], 2)
        param_ind += upl
        z_l = tf.tanh(tf.matmul(A_l,z_l) + b_l)

    A_out = tf.reshape(params[:, param_ind:(param_ind+upl*D)], (K, D, upl))
    param_ind += upl*D
    b_out = tf.expand_dims(params[:, param_ind:(param_ind+D)], 2)
    param_ind += D
    z_out = tf.matmul(A_out, z_l) + b_out

    # transpose back to [K, n, D]
    z_out = tf.transpose(z_out, [0, 2, 1])

    return z_out, param_ind

