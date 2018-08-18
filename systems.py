import tensorflow as tf
import numpy as np
from tf_util.tf_util import count_layer_params
from tf_util.flows import SoftPlusLayer, IntervalFlowLayer
import scipy.stats
from scipy.special import gammaln, psi
import scipy.io as sio
from itertools import compress

def system_from_str(system_str):
	if (system_str in ['null', 'null_on_interval']):
		return null_on_interval;
	elif (system_str in ['one_con', 'one_con_on_interval']):
		return one_con_on_interval;
	elif (system_str in ['two_con', 'two_con_on_interval']):
		return two_con_on_interval;
	elif (system_str in ['linear_1D']):
		return linear_1D;
	elif (system_str in ['linear_2D']):
		return linear_2D;
	elif (system_str in ['damped_harmonic_oscillator', 'dho']):
		return damped_harmonic_oscillator;


class system:
	"""Base class for exponential families.
	
	Exponential families differ in their sufficient statistics, base measures,
	supports, ane therefore their natural parametrization.  Derivatives of this 
	class are useful tools for learning exponential family models in tensorflow.

	Attributes:
			T (int): number of time points
			dt (float): time resolution of simulation
			behavior_str (str): determines sufficient statistics that characterize system
	"""

	def __init__(self, behavior_str, T, dt):
		"""family constructor

		Args:
			T (int): number of time points
			dt (float): time resolution of simulation
			behavior_str (str): determines sufficient statistics that characterize system
	
		"""
		self.behavior_str = behavior_str;
		self.T = T;
		self.dt = dt;

	def map_to_parameter_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to parameter support."""
		return layers, num_theta_params

	def compute_suff_stats(self, phi):
		"""Compute sufficient statistics of density network samples."""
		raise NotImplementedError();

	def analytic_suff_stats(self, phi):
		"""Compute closed form sufficient statistics."""
		raise NotImplementedError();

	def simulation_suff_stats(self, phi):
		"""Compute sufficient statistics that require simulation."""
		raise NotImplementedError();

	def compute_mu(self, params):
		"""No comment yet."""
		raise NotImplementedError();

	def center_suff_stats_by_mu(self, T_x, mu):
		"""Center sufficient statistics by the mean parameters mu."""
		return T_x - np.expand_dims(np.expand_dims(mu, 0), 1);


class null_on_interval(system):
	"""Null system.  D parameters no constraints.  
	   Solution should be uniform on interval.

	Attributes:
		D (int): parametric dimensionality
		a (float): beginning of interval
		b (float): end of interval
	"""

	def __init__(self, D, a=0, b=1):
		self.name = 'null_on_interval';
		self.D = D;
		self.T = 1;
		self.dt = .001;
		self.num_suff_stats = 0;
		self.a = a;
		self.b = b;

	def compute_suff_stats(self, phi):
		"""Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		phi_shape = tf.shape(phi);
		K = phi_shape[0];
		M = phi_shape[1];
		return tf.zeros((K,M,0), dtype=tf.float64);

	def compute_mu(self, behavior):
		return np.array([], dtype=np.float64);


	def map_to_parameter_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to parameter support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = IntervalFlowLayer('IntervalFlowLayer', self.a, self.b);
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params

class one_con_on_interval(null_on_interval):
	"""System with one constraint.  D parameters dim1 == dim2.  
	   Solution should be uniform on the plane.

	Attributes:
		D (int): parametric dimensionality
		a (float): beginning of interval
		b (float): end of interval
	"""

	def __init__(self, D, a=0, b=1):
		super().__init__(D, a, b);
		if ((type(D) is not int) or (D < 2)):
			print('Error: need at least two dimensions for plane on interval');
			raise ValueError;
		self.name = 'one_con_on_interval';
		self.num_suff_stats = 2;

	def compute_suff_stats(self, phi):
		"""Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		phi_shape = tf.shape(phi);
		K = phi_shape[0];
		M = phi_shape[1];
		diff01 = phi[:,:,0] - phi[:,:,1];
		T_x = tf.concat((diff01, tf.square(diff01)), axis=2);
		return T_x;

	def compute_mu(self, behavior):
		return np.array([0.0, 0.001], dtype=np.float64);


	def map_to_parameter_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to parameter support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = IntervalFlowLayer('IntervalFlowLayer', self.a, self.b);
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params


class two_con_on_interval(null_on_interval):
	"""System with two constraints.  D parameters dim1 == dim2, dim2 == dim3  
	   Solution should be uniform on the plane.

	Attributes:
		D (int): parametric dimensionality
		a (float): beginning of interval
		b (float): end of interval
	"""

	def __init__(self, D, a=0, b=1):
		super().__init__(D, a, b);
		if ((type(D) is not int) or (D < 3)):
			print('Error: need at least three dimensions for plane on interval');
			raise ValueError;
		self.name = 'two_con_on_interval';
		self.num_suff_stats = 4;

	def compute_suff_stats(self, phi):
		"""Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		phi_shape = tf.shape(phi);
		K = phi_shape[0];
		M = phi_shape[1];
		diff01 = phi[:,:,0] - phi[:,:,1];
		diff12 = phi[:,:,1] - phi[:,:,2];
		T_x = tf.concat((diff01, diff12, tf.square(diff01), tf.square(diff12)), axis=2);
		return T_x;

	def compute_mu(self, behavior):
		return np.array([0.0, 0.0, 0.001, 0.001], dtype=np.float64);


	def map_to_parameter_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to parameter support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = IntervalFlowLayer('IntervalFlowLayer', self.a, self.b);
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params


class linear_1D(system):
	"""Linear one-dimensional systems.

	Attributes:
		D (int): parametric dimensionality
		T (int): number of time points
		dt (float): time resolution of simulation
		behavior_str (str): determines sufficient statistics that characterize system
	"""

	def __init__(self, behavior_str, T, dt, init_conds):
		super().__init__(behavior_str, T, dt);
		self.name = 'linear_1D';
		self.D = 1;
		self.init_conds = init_conds;
		self.num_suff_stats = 2;

	def simulate(self, phi):
		phi_shape = tf.shape(phi);
		K = phi_shape[0];
		M = phi_shape[1];
		X = [];
		X_t = self.init_conds[0]*tf.ones((K,M,1,1), dtype=tf.float64);
		X.append(X_t);
		for i in range(1,self.T):
			X_dot = tf.multiply(phi, X_t);
			X_next = X_t + self.dt*X_dot;
			X.append(X_next);
			X_t = X_next;
		return tf.concat(X, axis=3);

	def compute_suff_stats(self, phi):
		"""Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		if (self.behavior_str == 'steady_state'):
			T_x = self.simulation_suff_stats(phi);
		else:
			raise NotImplementedError;
		return T_x;

	def analytic_suff_stats(self, phi):
		"""Compute closed form sufficient statistics.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Analytic sufficient statistics of samples.
		"""
		if (self.behavior_str == 'steady_state'):
			ss = (self.init_conds[0]*tf.exp(self.dt*(self.T-1)*phi))[:,:,:,0];
			ss = tf.clip_by_value(ss, -1e3, 1e3);
			T_x = tf.concat((ss, tf.square(ss)), 2);
		return T_x;


	def simulation_suff_stats(self, phi):
		"""Compute sufficient statistics that require simulation.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Simulation-derived sufficient statistics of samples.
		"""
		if (self.behavior_str == 'steady_state'):
			X = self.simulate(phi);
			ss = X[:,:,:,-1];
			T_x = tf.concat((ss, tf.square(ss)), 2);
		return T_x;

	def compute_mu(self, behavior):
		mu = behavior['mu'];
		Sigma = behavior['Sigma'];
		mu_mu = mu;
		mu_Sigma = np.zeros((int(self.D*(self.D+1)/2)),);
		ind = 0;
		for i in range(self.D):
			for j in range(i,self.D):
				mu_Sigma[ind] = Sigma[i,j] + mu[i]*mu[j];
				ind += 1;

		mu = np.concatenate((mu_mu, mu_Sigma), 0);
		return mu;




class damped_harmonic_oscillator(system):
	"""Linear one-dimensional systems.

	Attributes:
		D (int): parametric dimensionality
		T (int): number of time points
		dt (float): time resolution of simulation
		behavior_str (str): determines sufficient statistics that characterize system
	"""

	def __init__(self, behavior_str, T, dt, init_conds, bounds):
		super().__init__(behavior_str, T, dt);
		self.name = 'damped_harmonic_oscillator';
		self.D = 3;
		self.init_conds = init_conds;
		self.num_suff_stats = 2*T;
		self.bounds = bounds;

	def simulate(self, phi):
		"""Compute sufficient statistics that require simulation.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Simulation-derived sufficient statistics of samples.
		"""
		phi_shape = tf.shape(phi);
		K = phi_shape[0];
		M = phi_shape[1];

		print('phi', phi.shape);
		k = phi[:,:,0,:];
		c = phi[:,:,1,:];
		m = phi[:,:,2,:];

		w_0 = tf.sqrt(tf.divide(k,m));
		zeta = tf.divide(c, 2.0*tf.sqrt(tf.multiply(m,k)));

		X = [];
		Y = [];
		X_t = self.init_conds[0]*tf.ones((K,M,1), dtype=tf.float64);
		Y_t = self.init_conds[1]*tf.ones((K,M,1), dtype=tf.float64);
		X.append(tf.expand_dims(X_t, 3));
		Y.append(tf.expand_dims(Y_t, 3));
		for i in range(1,self.T):
			X_dot = Y_t;
			Y_dot = -2.0*tf.multiply(tf.multiply(w_0, zeta), Y_t) - tf.multiply(tf.square(w_0), X_t);
			X_next = X_t + self.dt*X_dot;
			Y_next = Y_t + self.dt*Y_dot;
			X.append(tf.expand_dims(X_next, 3));
			Y.append(tf.expand_dims(Y_next, 3));
			X_t = X_next;
			Y_t = Y_next;


		X = tf.concat(X, axis=3);
		Y = tf.concat(Y, axis=3);
		return tf.concat((X,Y), axis=2);

	def compute_suff_stats(self, phi):
		"""Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		if (self.behavior_str == 'trajectory'):
			T_x = self.simulation_suff_stats(phi);
		else:
			raise NotImplementedError;
		return T_x;

	def simulation_suff_stats(self, phi):
		"""Compute sufficient statistics that require simulation.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Simulation-derived sufficient statistics of samples.
		"""
		if (self.behavior_str == 'trajectory'):
			XY = self.simulate(phi);
			X = tf.clip_by_value(XY[:,:,0,:], -1e3, 1e3);
			T_x = tf.concat((X, tf.square(X)), 2);
		return T_x;

	def compute_mu(self, behavior):
		mu = behavior['mu'];
		Sigma = behavior['Sigma'];
		mu_mu = mu;
		mu_Sigma = np.square(mu_mu) + Sigma;
		print(mu_mu.shape, mu_Sigma.shape);
		mu = np.concatenate((mu_mu, mu_Sigma), 0);
		return mu;

	def map_to_parameter_support(self, layers, num_theta_params):
		"""Augment density network with bijective mapping to parameter support.

		Args:
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		Returns:
			layers (list): layers augmented with final support mapping layer.
			num_theta_params (int): Updated count of density network parameters.
		"""
		support_layer = IntervalFlowLayer('IntervalFlowLayer', self.bounds[0], self.bounds[1]);
		num_theta_params += count_layer_params(support_layer);
		layers.append(support_layer);
		return layers, num_theta_params



class linear_2D(system):
	"""Linear two-dimensional systems.

	Attributes:
		D (int): parametric dimensionality
		T (int): number of time points
		dt (float): time resolution of simulation
		behavior_str (str): determines sufficient statistics that characterize system
	"""

	def __init__(self, behavior_str, bounds):
		self.behavior_str = behavior_str;
		self.name = 'linear_2D';
		self.D = 4;
		self.dt = .001;
		self.T = 1;
		self.num_suff_stats = 6;
		self.bounds = bounds;

	def compute_suff_stats(self, phi):
		"""Compute sufficient statistics of density network samples.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
		if (self.behavior_str == 'oscillation'):
			T_x = self.analytic_suff_stats(phi);
		else:
			raise NotImplementedError;
		return T_x;

	def analytic_suff_stats(self, phi):
		"""Compute closed form sufficient statistics.

		Args:
			phi (tf.tensor): Density network system parameter samples.

		Returns:
			T_x (tf.tensor): Analytic sufficient statistics of samples.
		"""
		phi_shape = tf.shape(phi);
		K = phi_shape[0];
		M = phi_shape[1];

		tau = 0.100; # 100ms
		print('phi', phi.shape);
		"""
		w1 = phi[:,:,0,:];
		w2 = phi[:,:,1,:];
		w3 = phi[:,:,2,:];
		w4 = phi[:,:,3,:];

		a1 = (w1-1)/tau;
		a2 = w2/tau;
		a3 = w3/tau;
		a4 = (w4-1)/tau;
		"""

		a1 = phi[:,:,0,:];
		a2 = phi[:,:,1,:];
		a3 = phi[:,:,2,:];
		a4 = phi[:,:,3,:];

		beta = tf.complex(tf.square(a1 + a4) - 4*(a1*a4 + a2*a3), np.float64(0.0));
		beta_sqrt = tf.sqrt(beta);
		real_common = tf.complex(0.5*(a1 + a4), np.float64(0.0));
		if (self.behavior_str == 'oscillation'):
			lambda_1 = real_common + beta_sqrt;
			lambda_2 = real_common - beta_sqrt;
			lambda_1_real = tf.real(lambda_1);
			lambda_2_real = tf.real(lambda_2);
			lambda_1_imag = tf.imag(lambda_1);
			moments = [lambda_1_real, tf.square(lambda_1_real), \
			           lambda_2_real, tf.square(lambda_2_real), \
			           lambda_1_imag, tf.square(lambda_1_imag)];
			T_x = tf.concat(moments, 2);
		return T_x;


	def compute_mu(self, behavior):
		mu = behavior['mu'];
		Sigma = behavior['Sigma'];
		mu_mu = mu;
		mu_Sigma = np.square(mu_mu) + Sigma;
		print(mu_mu.shape, mu_Sigma.shape);
		mu = np.array([mu_mu[0], mu_Sigma[0], \
			           mu_mu[1], mu_Sigma[1], \
			           mu_mu[2], mu_Sigma[2]]);
		return mu;


