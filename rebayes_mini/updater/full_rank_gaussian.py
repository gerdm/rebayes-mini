from abc import ABC, abstractmethod

class MomentMatchedLinearGaussian:
    def __init__(self, apply_fn, dynamics_covariance):
        """
        apply_fn: function
            Maps state and observation to the natural parameters
        """
        self.apply_fn = apply_fn
        self.dynamics_covariance = dynamics_covariance

    def init_bel(self, params, cov=1.0):
        self.rfn, self.link_fn, init_params = self._initialise_link_fn(self.apply_fn, params)
        self.grad_link_fn = jax.jacrev(self.link_fn)

        nparams = len(init_params)
        return GaussState(
            mean=init_params,
            cov=jnp.eye(nparams) * cov,
        )

    def _initialise_link_fn(self, apply_fn, params):
        flat_params, rfn = ravel_pytree(params)

        @jax.jit
        def link_fn(params, x):
            return apply_fn(rfn(params), x)

        return rfn, link_fn, flat_params

    @abstractmethod
    def mean(self, eta):
        ...

    @abstractmethod
    def covariance(self, eta):
        ...

    def predictive_density(self, bel, X):
        eta = self.link_fn(bel.mean, X).astype(float)
        mean = self.mean(eta)
        Rt = jnp.atleast_2d(self.covariance(eta))
        Ht = self.grad_link_fn(bel.mean, X)
        covariance = Ht @ bel.cov @ Ht.T + Rt
        mean = jnp.atleast_1d(mean)
        dist = distrax.MultivariateNormalFullCovariance(mean, covariance)
        return dist

    def log_predictive_density(self, y, X, bel):
        """
        compute the log-posterior predictive density
        of the moment-matched Gaussian
        """
        log_p_pred = self.predictive_density(bel, X).log_prob(y)
        return log_p_pred


    def update(self, bel, y, x):
        eta = self.link_fn(bel.mean, x).astype(float)
        yhat = self.mean(eta)
        err = y - yhat

        Rt = jnp.atleast_2d(self.covariance(eta))
        Ht = self.grad_link_fn(bel.mean, x)
        St = Ht @ bel.cov @ Ht.T + Rt
        Kt = jnp.linalg.solve(St, Ht @ bel.cov).T

        mean_update = bel.mean + Kt @ err
        # cov_update = bel.cov - Kt @ St @ Kt.T
        I = jnp.eye(len(bel.mean))
        cov_update = (I - Kt @ Ht) @ bel.cov @ (I - Kt @ Ht).T + Kt @ Rt @ Kt.T
        bel = bel.replace(mean=mean_update, cov=cov_update)
        return bel

    def step(self, bel, y, x, callback_fn):
        bel_update = self.update(bel, y, x)
        output = callback_fn(bel_update, bel, y, x)
        return bel_update, output

    def scan(self, bel, y, X, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, x = yX
            bel, out = self.step(bel, y, x, callback_fn)
            return bel, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist


class GaussianFilter(MomentMatchedLinearGaussian):
    def __init__(self, apply_fn, variance=1.0):
        super().__init__(apply_fn)
        self.variance = variance

    def mean(self, eta):
        return eta

    def covariance(self, eta):
        return self.variance * jnp.eye(1)


class MultinomialFilter(MomentMatchedLinearGaussian):
    def __init__(self, apply_fn, eps=0.1):
        super().__init__(apply_fn)
        self.eps = eps

    def mean(self, eta):
        return jax.nn.softmax(eta)

    def covariance(self, eta):
        mean = self.mean(eta)
        return jnp.diag(mean) - jnp.outer(mean, mean) + jnp.eye(len(eta)) * self.eps
