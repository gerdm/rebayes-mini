from rebayes_mini.auxiliary import Runlength

class ExpfamRLPR(Runlength):
    """
    Runlength prior reset (RL-PR)
    """
    def __init__(self, p_change, K, updater):
        super().__init__(p_change, K)
        self.updater = updater

    def log_predictive_density(self, y, X, bel):
        return self.updater.log_predictive_density(y, X, bel)

    def update_bel(self, y, X, bel):
        bel = self.updater.update(bel_pred, y, X)
        return bel

    def init_bel(self, mean, cov, log_joint_init=0.0):
        """
        Initialize belief state
        """
        state_updater = self.updater.init_bel(mean, cov)
        mean = state_updater.mean
        cov = state_updater.cov

        bel = states.BOCDGaussState(
            mean=einops.repeat(mean, "i -> k i", k=self.K),
            cov=einops.repeat(cov, "i j -> k i j", k=self.K),
            log_joint=(jnp.ones((self.K,)) * -jnp.inf).at[0].set(log_joint_init),
            runlength=jnp.zeros(self.K)
        )

        return bel
