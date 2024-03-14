import chex

@chex.dataclass
class GaussState:
    mean: chex.Array
    cov: chex.Array

