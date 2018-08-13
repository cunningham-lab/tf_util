import numpy as np

def drawPoissonRates(D, ratelim):
    return np.random.uniform(0.1, ratelim, (D,));

def drawPoissonCounts(z, N):
    D = z.shape[0];
    x = np.zeros((D,N));
    for i in range(D):
        x[i,:] = np.random.poisson(z[i], (N,));
    return x;

def truncated_multivariate_normal_rvs(mu, Sigma):
    D = mu.shape[0];
    L = np.linalg.cholesky(Sigma);
    rejected = True;
    count = 1;
    while (rejected):
        z0 = np.random.normal(0,1,(D));
        z = np.dot(L, z0) + mu;
        rejected = 1 - np.prod((np.sign(z)+1)/2);
        count += 1;
    return z;

def get_GP_Sigma(tau, T, Ts):
    K = np.zeros((T, T));
    for i in range(T):
        for j in range(i,T):
            diff = (i-j)*Ts;
            K[i,j] = np.exp(-(np.abs(diff)**2) / (2*(tau**2)));
            if (i != j):
                K[j,i] = K[i,j];
    return K;
    