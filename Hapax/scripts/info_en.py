"""English names and application domains of the registry operators.

Translated from the method documentation's INFO table (build_xlsx.py), plus the
operators added since. Domains are the paper's coarse categories.
"""

ML = "Machine learning"
STAT = "Statistics"
OT = "Optimal transport"
SIG = "Signal processing"
NLA = "Numerical linear algebra"
PHYS = "Statistical physics"
PREP = "Data preprocessing"
CONS = "Constructed"

INFO = {
    "ledoit_wolf": ("Ledoit--Wolf covariance", STAT),
    "ols_hc0": ("scalar OLS HC0 covariance", STAT),
    "normalized_policy_loss": ("normalised policy loss", ML),
    "softmax": ("softmax", ML),
    "log_softmax": ("log-softmax", ML),
    "logsumexp": ("log-sum-exp", ML),
    "cross_entropy": ("cross-entropy (on logits)", ML),
    "soft_cross_entropy": ("soft-label cross-entropy", ML),
    "label_smoothing": ("label-smoothing cross-entropy", ML),
    "softmax_entropy": ("entropy of a softmax", ML),
    "attention": ("scaled dot-product attention", ML),
    "attention_bias": ("attention with an additive bias", ML),
    "variance": ("variance", STAT),
    "layernorm": ("LayerNorm", ML),
    "covariance": ("covariance", STAT),
    "moment3": ("3rd central moment", STAT),
    "moment4": ("4th central moment", STAT),
    "moment5": ("5th central moment", STAT),
    "layernorm_backward": ("LayerNorm backward", ML),
    "zscore_max": ("largest z-score", STAT),
    "max_abs_deviation": ("largest absolute deviation", STAT),
    "peak_power": ("peak power after mean removal", SIG),
    "papr": ("peak-to-average power ratio", SIG),
    "peak_to_geomean": ("max over geometric mean", CONS),
    "monotone_sum": ("max of a monotone polynomial", CONS),
    "projection": ("Gram-Schmidt remainder", NLA),
    "householder": ("Householder-type norm", NLA),
    "cosine": ("cosine similarity", ML),
    "sinkhorn_half": ("log-domain Sinkhorn half step", OT),
    "sinkhorn_iteration": ("log-domain Sinkhorn iteration", OT),
    "sinkhorn_square": ("Sinkhorn iteration, equal sizes", OT),
    "weighted_variance": ("weighted variance", STAT),
    "skewness": ("skewness", STAT),
    "kurtosis": ("kurtosis", STAT),
    "pearson": ("Pearson correlation", STAT),
    "exp_centered": ("exponential sum about the mean", CONS),
    "instance_norm": ("InstanceNorm (1-D)", ML),
    "minmax_mean": ("mean after min-max scaling", PREP),
    "collision": ("collision probability of a softmax", ML),
    "ess": ("effective sample size", STAT),
    "softmax_jvp": ("softmax backward (JVP)", ML),
    "contrastive": ("symmetric contrastive loss", ML),
    "heat_capacity": ("heat capacity (energy fluctuation)", PHYS),
    # negative operators
    "logit_norm": ("logit normalisation", ML),
    "standardised_softmax": ("softmax of standardised scores", ML),
    "mean_abs_deviation": ("mean absolute deviation", STAT),
    "cov_matrix": ("covariance matrix", STAT),
    "normalised_projection": ("projection remainder, general q", NLA),
    "scaled_norm": ("scaled Euclidean norm (BLAS)", NLA),
    "min_sq_deviation": ("smallest squared deviation", CONS),
    "abs_from_max": ("sum of distances to the max", CONS),
}

# mechanism classes, by the heaviest selection step a fused operator needed
MECH_ORDER = [("hoist", "hoist"), ("parts", "parts"), ("promote", "promote"), ("supplement", "supplement")]
MECH_NAME = {
    "rebase": "rebasing only",
    "supplement": "supplement",
    "promote": "promote",
    "parts": "take a max apart",
    "hoist": "hoist",
}


def mechanism(actions: list[str]) -> str:
    acts = set(actions)
    for key, verb in MECH_ORDER:
        if verb in acts:
            return key
    return "rebase"
