import tensorflow as tf
import numpy as np
import scipy.stats
from tf_util.stat_util import (
    approx_equal,
    get_dist_str,
    get_sampler_func,
    get_density_func,
    sample_gumbel,
)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

EPS = 1e-6
PERC_EPS = 1e-3

def test_approx_equal():
    small_eps = 1e-16
    big_eps = 1e-2
    a = 0.0
    b = 0.001
    assert approx_equal(a, a, small_eps)
    assert approx_equal(b, b, small_eps)
    assert not approx_equal(a, b, small_eps)
    assert approx_equal(a, a, big_eps)
    assert approx_equal(b, b, big_eps)
    assert approx_equal(a, b, big_eps)

    a = np.random.normal(0.0, 1.0, (100, 100))
    b = np.random.normal(0.0, 1.0, (100, 100))
    assert approx_equal(a, a, small_eps)
    assert approx_equal(b, b, small_eps)
    assert not approx_equal(a, b, small_eps)
    return None


n = 100


def test_get_dist_str():
    assert get_dist_str(None) == ""

    a_vec = np.random.normal(0.0, 100.0, (n,))
    b_vec = a_vec + (np.abs(np.random.normal(0.0, 100.0, (n,))) + 0.001)
    scale_vec = np.abs(np.random.normal(0.0, 100.0, (n,)) + 0.01)
    df_fac_vec = np.random.randint(1, 1000, (n,))

    for i in range(n):
        a = a_vec[i]
        b = b_vec[i]
        scale = scale_vec[i]
        df_fac = df_fac_vec[i]

        dist = {"family": "delta", "a": a}
        assert get_dist_str(dist) == "_delt_%.1f" % a

        dist = {"family": "uniform", "a": a, "b": b}
        assert get_dist_str(dist) == "_u_%.1fto%.1f" % (a, b)

        dist = {"family": "uniform_int", "a": a, "b": b}
        assert get_dist_str(dist) == "_ui_%dto%d" % (a, b)

        dist = {"family": "multivariate_normal"}
        assert get_dist_str(dist) == "_mvn"

        dist = {"family": "isotropic_normal", "scale": scale}
        assert get_dist_str(dist) == "_in_s=%.3f" % scale

        dist = {"family": "truncated_normal"}
        assert get_dist_str(dist) == "_tn"

        dist = {"family": "isotropic_truncated_normal", "scale": scale}
        assert get_dist_str(dist) == "_itn_s=%.2f" % scale

        dist = {"family": "dirichlet"}
        assert get_dist_str(dist) == "_dir"

        dist = {"family": "inv_wishart"}
        assert get_dist_str(dist) == "_iw"

        dist = {"family": "isotropic_inv_wishart", "df_fac": df_fac}
        assert get_dist_str(dist) == "_iiw_%d" % df_fac
    return None


Ds = [2, 4, 10, 25]
K = 5
n = 100


def test_delta():
    for D in Ds:
        for k in range(K):
            a = np.random.normal(0.0, 100.0, (D,))
            delta_dist = {"family": "delta", "a": a}
            delta_sampler = get_sampler_func(delta_dist, D)
            delta_density = get_density_func(delta_dist, D)
            p_zs = np.zeros((n,))
            for i in range(n):
                z = delta_sampler()
                p_z = delta_density(z)
                assert approx_equal(z, a, EPS)
                assert p_z == 1.0
                p_zs[i] = p_z
            H = -np.mean(np.log(p_zs))
            assert H == 0.0
    return None


def test_uniform():
    for D in Ds:
        for k in range(K):
            a = np.random.normal(0.0, 100.0, (1,))
            b = a + np.abs(np.random.normal(0.0, 100.0, (1,))) + 0.001
            uniform_dist = {"family": "uniform", "a": a, "b": b}
            uniform_sampler = get_sampler_func(uniform_dist, D)
            uniform_density = get_density_func(uniform_dist, D)
            p_zs = np.zeros((n,))
            for i in range(n):
                z = uniform_sampler()
                p_z = uniform_density(z)
                for d in range(D):
                    assert a <= z[d]
                    assert z[d] <= b
                assert p_z == (1.0 / (b - a)) ** D
                p_zs[i] = p_z
            H = -np.mean(np.log(p_zs))
            assert approx_equal(H, -D * np.log(1.0 / (b - a)), EPS)
    return None


def test_uniform_int():
    for D in Ds:
        for k in range(K):
            a = np.random.randint(0, 100, (1,))
            b = a + np.abs(np.random.randint(0.0, 100.0, (1,))) + 0.001
            uniform_int_dist = {"family": "uniform_int", "a": a, "b": b}
            uniform_int_sampler = get_sampler_func(uniform_int_dist, D)
            uniform_int_density = get_density_func(uniform_int_dist, D)
            p_zs = np.zeros((n,))
            for i in range(n):
                z = uniform_int_sampler()
                p_z = uniform_int_density(z)
                assert a <= a
                assert z <= b
                assert p_z == 1.0 / (b - a)
                p_zs[i] = p_z
            H = -np.mean(np.log(p_zs))
            assert approx_equal(H, -np.log(1.0 / (b - a)), EPS)
    return None


# This should scale with dimensionality
MVN_H_EPS = 1.0


def test_multivariate_normal():
    for D in Ds:
        df_fac = 2 * D
        Sigma_dist = scipy.stats.invwishart(df=df_fac, scale=df_fac * np.eye(D))
        for k in range(K):
            mu = np.random.normal(0.0, 1.0, (D,))
            Sigma = Sigma_dist.rvs(1)
            mvn_dist = {"family": "multivariate_normal", "mu": mu, "Sigma": Sigma}
            mvn_sampler = get_sampler_func(mvn_dist, D)
            mvn_density = get_density_func(mvn_dist, D)
            p_zs = np.zeros((n,))
            for i in range(n):
                z = mvn_sampler()
                assert np.all(np.isfinite(z))
                assert np.all(np.isreal(z))
                p_z = mvn_density(z)
                diff = np.expand_dims(z - mu, 1)
                p_z_true = (
                    1.0
                    / np.sqrt(np.linalg.det(2 * np.pi * Sigma))
                    * np.exp(
                        -0.5
                        * np.dot(np.transpose(diff), np.dot(np.linalg.inv(Sigma), diff))
                    )
                )
                assert approx_equal(p_z, p_z_true, EPS)
                p_zs[i] = p_z
            H = -np.mean(np.log(p_zs))
            H_true = 0.5 * np.log(np.linalg.det(2 * np.pi * np.exp(1) * Sigma))
            assert approx_equal(H, H_true, MVN_H_EPS)

    return None


def test_isotropic_normal():
    for D in Ds:
        df_fac = 2 * D
        for k in range(K):
            mu = np.random.normal(0.0, 1.0, (D,))
            scale = np.abs(np.random.normal(0.0, 1.0, (1,))) + 0.001
            iso_mvn_dist = {"family": "isotropic_normal", "mu": mu, "scale": scale}
            Sigma = scale * np.eye(D)
            iso_mvn_sampler = get_sampler_func(iso_mvn_dist, D)
            iso_mvn_density = get_density_func(iso_mvn_dist, D)
            p_zs = np.zeros((n,))
            for i in range(n):
                z = iso_mvn_sampler()
                assert np.all(np.isfinite(z))
                assert np.all(np.isreal(z))
                p_z = iso_mvn_density(z)
                diff = np.expand_dims(z - mu, 1)
                p_z_true = (
                    1.0
                    / np.sqrt(np.linalg.det(2 * np.pi * Sigma))
                    * np.exp(
                        -0.5
                        * np.dot(np.transpose(diff), np.dot(np.linalg.inv(Sigma), diff))
                    )
                )
                assert approx_equal(p_z, p_z_true, EPS)
                p_zs[i] = p_z
            H = -np.mean(np.log(p_zs))
            H_true = 0.5 * np.log(np.linalg.det(2 * np.pi * np.exp(1) * Sigma))
            assert approx_equal(H, H_true, MVN_H_EPS)

    return None

"""
def test_truncated_normal():
    for D in Ds:
        df_fac = 2 * D
        Sigma_dist = scipy.stats.invwishart(df=df_fac, scale=df_fac * np.eye(D))
        for k in range(K):
            mu = np.abs(np.random.normal(0.0, 1.0, (D,))) + 0.001
            Sigma = Sigma_dist.rvs(1)
            tn_dist = {"family": "truncated_normal", "mu": mu, "Sigma": Sigma}
            tn_sampler = get_sampler_func(tn_dist, D)
            for i in range(n):
                z = tn_sampler()
                assert np.all(np.isfinite(z))
                assert np.all(np.isreal(z))
                assert np.sum(z < 0.0) == 0.0

    return None


def test_isotropic_truncated_normal():
    for D in Ds:
        df_fac = 2 * D
        for k in range(K):
            mu = np.abs(np.random.normal(0.0, 1.0, (D,))) + 0.001
            scale = np.abs(np.random.normal(0.0, 1.0, (1,))) + 0.001
            iso_tn_dist = {
                "family": "isotropic_truncated_normal",
                "mu": mu,
                "scale": scale,
            }
            iso_tn_sampler = get_sampler_func(iso_tn_dist, D)
            for i in range(n):
                z = iso_tn_sampler()
                assert np.all(np.isfinite(z))
                assert np.all(np.isreal(z))
                assert np.sum(z < 0.0) == 0.0

    return None
"""


DIR_H_EPS = 1.0


def test_dirichlet():
    for D in Ds:
        for k in range(K):
            alpha = np.random.uniform(0.5, 5.0, (D,))
            dir_dist = {"family": "dirichlet", "alpha": alpha}
            dist_true = scipy.stats.dirichlet(alpha)
            dir_sampler = get_sampler_func(dir_dist, D)
            dir_density = get_density_func(dir_dist, D)
            p_zs = np.zeros((n,))
            for i in range(n):
                z = dir_sampler()
                assert np.sum(z < 0.0) == 0.0
                assert np.all(np.isreal(z))
                assert approx_equal(np.sum(z), 1.0, EPS)
                p_z = dir_density(z)
                p_z_true = dist_true.pdf(z)
                assert approx_equal(p_z, p_z_true, EPS)
                p_zs[i] = p_z
            H = -np.mean(np.log(p_zs))
            H_true = dist_true.entropy()
            assert approx_equal(H, H_true, DIR_H_EPS)

    return None


IW_H_EPS = 1e-2


def test_inv_wishart():
    for D in Ds:
        df_fac = 2 * D
        Psi_dist = scipy.stats.invwishart(df=df_fac, scale=np.eye(D))
        for k in range(K):
            df = int(np.random.randint(D, 100 * D, ()))
            Psi = Psi_dist.rvs(1)
            iw_dist = {"family": "inv_wishart", "df": df, "Psi": Psi}
            dist_true = scipy.stats.invwishart(df=df, scale=Psi)
            iw_sampler = get_sampler_func(iw_dist, D)
            iw_density = get_density_func(iw_dist, D)
            for i in range(n):
                z = iw_sampler()
                assert np.all(np.linalg.eigvals(z) > 0.0)
                assert np.all(np.isreal(z))
                p_z = iw_density(z)
                p_z_true = dist_true.pdf(z)
                assert approx_equal(p_z, p_z_true, PERC_EPS, allow_special=True, perc=True)

    return None


def test_isotropic_inv_wishart():
    for D in Ds:
        for k in range(K):
            df_fac = int(np.random.randint(1, 100, ()))
            iso_iw_dist = {"family": "isotropic_inv_wishart", "df_fac": df_fac}
            dist_true = scipy.stats.invwishart(
                df=df_fac * D, scale=df_fac * D * np.eye(D)
            )
            iso_iw_sampler = get_sampler_func(iso_iw_dist, D)
            iso_iw_density = get_density_func(iso_iw_dist, D)
            for i in range(n):
                z = iso_iw_sampler()
                assert np.all(np.linalg.eigvals(z) > 0.0)
                assert np.all(np.isreal(z))
                p_z = iso_iw_density(z)
                p_z_true = dist_true.pdf(z)
                assert approx_equal(p_z, p_z_true, PERC_EPS, allow_special=True, perc=True)

    return None

def test_sample_gumbel():
    M = 1000
    K = 100
    G = sample_gumbel(M, K)
    euler_masch_const = 0.5772156649
    true_mean = euler_masch_const
    true_var = np.square(np.pi) / 6.0
    true_skew = np.square(np.pi) / 6.0
    true_kurtosis = 12.0 / 5
    assert(np.abs(true_mean - np.mean(G)) < 0.01)
    assert(np.abs(true_var - np.var(G)) < 0.05)
    assert(np.abs(true_kurtosis - scipy.stats.kurtosis(np.reshape(G, (M*K)))) < 0.2)

    return None


if __name__ == "__main__":
    test_sample_gumbel()
    """
    test_approx_equal()
    test_get_dist_str()
    test_delta()
    test_uniform()
    test_uniform_int()
    test_multivariate_normal()
    test_isotropic_normal()
    #test_truncated_normal()  # need to use MCMC.  Current rejection sampler 
    #test_isotropic_truncated_normal() # is intractable in higher dimensions
    test_dirichlet()
    test_inv_wishart()
    test_isotropic_inv_wishart()
    """
