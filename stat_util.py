import numpy as np
from scipy.stats import invwishart, dirichlet

def get_sampler_func(dist, D):
    if (dist['family'] == 'uniform'):
        a = dist['a'];
        b = dist['b'];
        return lambda : np.random.uniform(a,b,(D,));

    elif (dist['family'] == 'uniform_int'):
        a = dist['a'];
        b = dist['b'];
        return lambda : np.random.randint(a,b+1,());

    elif (dist['family'] == 'multivariate_normal'):
        mu = dist['mu'];
        Sigma = dist['Sigma'];
        return lambda : np.random.multivariate_normal(mu, Sigma);

    elif (dist['family'] == 'isotropic_normal'):
        mu = dist['mu'];
        scale = dist['scale'];
        Sigma = scale*np.eye(D);
        dist = {'family':'multivariate_normal', 'mu':mu, 'Sigma':Sigma};
        return get_sampler_func(dist, D);

    elif (dist['family'] == 'truncated_normal'):
        mu = dist['mu'];
        Sigma = dist['Sigma'];
        L = np.linalg.cholesky(Sigma);
        return lambda : truncated_multivariate_normal_fast_rvs(mu, L);

    elif (dist['family'] == 'isotropic_truncated_normal'):
        mu = dist['mu'];
        scale = dist['scale'];
        Sigma = scale*np.eye(D);
        dist = {'family':'truncated_normal', 'mu':mu, 'Sigma':Sigma};
        return get_sampler_func(dist, D);

    elif (dist['family'] == 'inv_wishart'):
        df = dist['df'];
        Psi = dist['Psi'];
        iw = invwishart(df=df, scale=Psi);
        return lambda : iw.rvs(1);

    elif (dist['family'] == 'isotropic_inv_wishart'):
        df_fac = dist['df_fac'];
        df = df_fac*D;
        Psi = df*np.eye(D);
        dist = {'family':'inv_wishart', 'df':df, 'Psi':Psi};
        return get_sampler_func(dist, D);

    elif (dist['family'] == 'iso_mvn_and_iso_iw'):
        mu = dist['mu'];
        scale = dist['scale']
        dist_iso_mvn = {'family':'isotropic_normal', 'mu':mu, 'scale':scale}
        df_fac = dist['df_fac'];
        dist_iso_iw = {'family':'isotropic_inv_wishart', 'df_fac':df_fac};
        iso_mvn_sampler = get_sampler_func(dist_iso_mvn, D);
        iso_iw_sampler = get_sampler_func(dist_iso_iw, D);
        return lambda : (iso_mvn_sampler(), iso_iw_sampler());

    elif (dist['family'] == 'ui_and_iso_iw'):
        a = dist['a'];
        b = dist['b'];
        dist_ui = {'family':'uniform_int', 'a':a, 'b':b}
        df_fac = dist['df_fac'];
        dist_iso_iw = {'family':'isotropic_inv_wishart', 'df_fac':df_fac};
        ui_sampler = get_sampler_func(dist_ui, dist['ui_dim']);
        iso_iw_sampler = get_sampler_func(dist_iso_iw, dist['iw_dim']);
        return lambda : (ui_sampler(), iso_iw_sampler());

def get_dist_str(dist):
    if (dist['family'] == 'uniform'):
        a = dist['a'];
        b = dist['b'];
        return 'u_%.1fto%.1f' % (a,b);

    elif (dist['family'] == 'uniform_int'):
        a = dist['a'];
        b = dist['b'];
        return 'ui_%dto%d' % (a,b);

    elif (dist['family'] == 'multivariate_normal'):
        mu = dist['mu'];
        Sigma = dist['Sigma'];
        return 'mvn'

    elif (dist['family'] == 'isotropic_normal'):
        mu = dist['mu'];
        scale = dist['scale'];
        return 'in_s=%.2f' % scale;

    elif (dist['family'] == 'truncated_normal'):
        mu = dist['mu'];
        scale = dist['scale'];
        return 'tn';

    elif (dist['family'] == 'isotropic_truncated_normal'):
        mu = dist['mu'];
        scale = dist['scale'];
        return 'itn_s=%.2f' % scale;

    elif (dist['family'] == 'inv_wishart'):
        df = dist['df'];
        Psi = dist['Psi'];
        iw = invwishart(df=df, scale=Psi);
        return 'iw'

    elif (dist['family'] == 'isotropic_inv_wishart'):
        df_fac = dist['df_fac'];
        return 'iiw_%d' % df_fac;

    elif (dist['family'] == 'iso_mvn_and_iso_iw'):
        mu = dist['mu'];
        scale = dist['scale']
        dist_iso_mvn = {'family':'isotropic_normal', 'mu':mu, 'scale':scale}
        df_fac = dist['df_fac'];
        dist_iso_iw = {'family':'isotropic_inv_wishart', 'df_fac':df_fac};
        return '%s_%s' % (get_dist_str(dist_iso_mvn), get_dist_str(dist_iso_iw));

    elif (dist['family'] == 'ui_and_iso_iw'):
        a = dist['a'];
        b = dist['b'];
        dist_ui = {'family':'uniform_int', 'a':a, 'b':b}
        df_fac = dist['df_fac'];
        dist_iso_iw = {'family':'isotropic_inv_wishart', 'df_fac':df_fac};
        return '%s_%s' % (get_dist_str(dist_ui), get_dist_str(dist_iso_iw));

def drawPoissonRates(D, ratelim):
    return np.random.uniform(0.1, ratelim, (D,));

def drawPoissonCounts(z, N):
    D = z.shape[0];
    x = np.zeros((D,N));
    for i in range(D):
        x[i,:] = np.random.poisson(z[i], (N,));
    return x;

def truncated_multivariate_normal_fast_rvs(mu, L):
    D = mu.shape[0];
    rejected = True;
    count = 1;
    while (rejected):
        z0 = np.random.normal(0,1,(D));
        z = np.dot(L, z0) + mu;
        rejected = 1 - np.prod((np.sign(z)+1)/2);
        count += 1;
    return z;

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
    