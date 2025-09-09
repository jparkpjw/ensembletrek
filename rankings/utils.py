
def feature_scale(arr, maximize=True):
    """
    Scale features to the range [0, 1].
    If maximize is True, the largest value gets rank 1 and the smallest value gets rank 0.
    If maximize is False, the smallest value gets rank 1 and the largest value gets rank 0.

    Args:
        arr: The array to scale (array-like)
        maximize: Whether to favor states that maxmize the input array (otherwise minimize)

    Returns:
        The scaled array (array-like)
    """
    if maximize:
        return (arr - arr.min()) / (arr.max() - arr.min())
    else:
        return (arr.max() - arr) / (arr.max() - arr.min())
    