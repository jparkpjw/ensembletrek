from .utils import feature_scale
import mdtraj as md
import numpy as np

def _fast_ranking(geometric_feature, counts, maximize=True, alpha=1):
    """
    FAST ranking, which sum feature scaled exploitation and exploration terms.

    Args:
        geometric_feature: The geometric feature measured for each state (array-like)
        counts: Number of times each state was observed (array-like)
        maximize: Whether to favor states that maxmize the geometric_feature (otherwise minimize)

    Returns:
        ranking: ranks of each state (array-like)
    """
    exploit_term = feature_scale(geometric_feature, maximize=maximize)
    stat_term = feature_scale(counts, maximize=False)
    return exploit_term + alpha * stat_term

def fast(n_choose, geometric_feature, counts, maximize=True, alpha=1):
    """
    Return the states selected based on the FAST ranking function.
    rank = exploit_term + alpha * stat_term
    exploit_term is feature scaled geometric term (favors large values if maximize=True)
    stat_term is feature scaled counts per state, favoring lowest counts

    Args:
        n_choose: Number of states to select
        geometric_feature: The geometric feature measured for each state (array-like)
        counts: Number of times each state was observed (array-like)
        maximize: Whether to favor states that maxmize the geometric_feature (otherwise minimize)
        alpha: The weight to put on the statistical/exploration term
    """
    ranks = _fast_ranking(geometric_feature, counts, maximize=maximize, alpha=alpha)
    return np.argsort(ranks)[-n_choose:]

def fast_spread(n_choose, geometric_feature, counts, centers, width, maximize=True, alpha=1):
    """
    Return the states selected based on the FAST ranking function with spreading.
    rank = exploit_term + alpha * stat_term
    exploit_term is feature scaled geometric term (favors large values if maximize=True)
    stat_term is feature scaled counts per state, favoring lowest counts
    Spreading adds a penalty to states that are close to previously chosen states in terms of RMSD.
    The first state is chosen based on the classic FAST ranking and then a penalty is added to all other states
    using a Gaussian centered at the first selected state. Then the second state is chosen and a new penalty is
    added to all remaining states. This is repeated until teh desired number of states is chosen.

    Args:
        n_choose: Number of states to select
        geometric_feature: The geometric feature measured for each state (array-like)
        counts: Number of times each state was observed (array-like)
        centers: An mdtraj.Trajectory object containing all the cluster center structures
        width: Sigma for the Gaussian penaly added to states near selected states
        maximize: Whether to favor states that maxmize the geometric_feature (otherwise minimize)
        alpha: The weight to put on the statistical/exploration term
    """
    choices = []
    ranks = _fast_ranking(geometric_feature, counts, maximize=maximize, alpha=alpha)
    for i in range(n_choose):
        choices.append(ranks.argmax())
        ranks[choices[-1]] = -1 * np.inf
        d = md.rmsd(centers, centers[choices[-1]])
        weights = 1 - np.exp(-(d**2)/float(2.0*(width**2)))
        ranks += weights

    choices = np.array(choices)
    
    return choices

