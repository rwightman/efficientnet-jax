def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def round_features(features, multiplier=1.0, divisor=8, feat_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return features
    features *= multiplier
    return make_divisible(features, divisor, feat_min)
