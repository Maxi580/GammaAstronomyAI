def classify_m1_stats(stats, telescope_stats):
    """
    Classify based on M1 telescope statistics using population statistics.
    Returns:
        1 for gamma
        0 for proton
        -1 if uncertain
    """
    mean, std, neg_ratio, min_val, max_val, squared_mean, _, q25, q50, q75 = stats

    # Get min/max values from statistics
    m1_proton_stats = telescope_stats["m1_proton"]
    m1_gamma_stats = telescope_stats["m1_gamma"]

    # Check if it's definitely a proton (outside gamma ranges)
    if (mean < m1_gamma_stats[:, 0].min() or mean > m1_gamma_stats[:, 0].max() or
            std < m1_gamma_stats[:, 1].min() or std > m1_gamma_stats[:, 1].max() or
            neg_ratio < m1_gamma_stats[:, 2].min() or neg_ratio > m1_gamma_stats[:, 2].max() or
            min_val < m1_gamma_stats[:, 3].min() or min_val > m1_gamma_stats[:, 3].max() or
            max_val < m1_gamma_stats[:, 4].min() or max_val > m1_gamma_stats[:, 4].max() or
            squared_mean < m1_gamma_stats[:, 5].min() or squared_mean > m1_gamma_stats[:, 5].max() or
            q25 < m1_gamma_stats[:, 7].min() or q25 > m1_gamma_stats[:, 7].max() or
            q50 < m1_gamma_stats[:, 8].min() or q50 > m1_gamma_stats[:, 8].max() or
            q75 < m1_gamma_stats[:, 9].min() or q75 > m1_gamma_stats[:, 9].max()):
        return 0

    # Check if it's definitely a gamma (outside proton ranges)
    if (mean < m1_proton_stats[:, 0].min() or mean > m1_proton_stats[:, 0].max() or
            std < m1_proton_stats[:, 1].min() or std > m1_proton_stats[:, 1].max() or
            neg_ratio < m1_proton_stats[:, 2].min() or neg_ratio > m1_proton_stats[:, 2].max() or
            min_val < m1_proton_stats[:, 3].min() or min_val > m1_proton_stats[:, 3].max() or
            max_val < m1_proton_stats[:, 4].min() or max_val > m1_proton_stats[:, 4].max() or
            squared_mean < m1_proton_stats[:, 5].min() or squared_mean > m1_proton_stats[:, 5].max() or
            q25 < m1_proton_stats[:, 7].min() or q25 > m1_proton_stats[:, 7].max() or
            q50 < m1_proton_stats[:, 8].min() or q50 > m1_proton_stats[:, 8].max() or
            q75 < m1_proton_stats[:, 9].min() or q75 > m1_proton_stats[:, 9].max()):
        return 1

    return -1


def classify_m2_stats(stats, telescope_stats):
    """
    Classify based on M2 telescope statistics using population statistics.
    Returns:
        1 for gamma
        0 for proton
        -1 if uncertain
    """
    mean, std, neg_ratio, min_val, max_val, squared_mean, _, q25, q50, q75 = stats

    # Get min/max values from statistics
    m2_proton_stats = telescope_stats["m2_proton"]
    m2_gamma_stats = telescope_stats["m2_gamma"]

    # Check if it's definitely a proton (outside gamma ranges)
    if (mean < m2_gamma_stats[:, 0].min() or mean > m2_gamma_stats[:, 0].max() or
            std < m2_gamma_stats[:, 1].min() or std > m2_gamma_stats[:, 1].max() or
            neg_ratio < m2_gamma_stats[:, 2].min() or neg_ratio > m2_gamma_stats[:, 2].max() or
            min_val < m2_gamma_stats[:, 3].min() or min_val > m2_gamma_stats[:, 3].max() or
            max_val < m2_gamma_stats[:, 4].min() or max_val > m2_gamma_stats[:, 4].max() or
            squared_mean < m2_gamma_stats[:, 5].min() or squared_mean > m2_gamma_stats[:, 5].max() or
            q25 < m2_gamma_stats[:, 7].min() or q25 > m2_gamma_stats[:, 7].max() or
            q50 < m2_gamma_stats[:, 8].min() or q50 > m2_gamma_stats[:, 8].max() or
            q75 < m2_gamma_stats[:, 9].min() or q75 > m2_gamma_stats[:, 9].max()):
        return 0

    # Check if it's definitely a gamma (outside proton ranges)
    if (mean < m2_proton_stats[:, 0].min() or mean > m2_proton_stats[:, 0].max() or
            std < m2_proton_stats[:, 1].min() or std > m2_proton_stats[:, 1].max() or
            neg_ratio < m2_proton_stats[:, 2].min() or neg_ratio > m2_proton_stats[:, 2].max() or
            min_val < m2_proton_stats[:, 3].min() or min_val > m2_proton_stats[:, 3].max() or
            max_val < m2_proton_stats[:, 4].min() or max_val > m2_proton_stats[:, 4].max() or
            squared_mean < m2_proton_stats[:, 5].min() or squared_mean > m2_proton_stats[:, 5].max() or
            q25 < m2_proton_stats[:, 7].min() or q25 > m2_proton_stats[:, 7].max() or
            q50 < m2_proton_stats[:, 8].min() or q50 > m2_proton_stats[:, 8].max() or
            q75 < m2_proton_stats[:, 9].min() or q75 > m2_proton_stats[:, 9].max()):
        return 1

    return -1


def rule_based_minmax_classifier(stats_m1, stats_m2, telescope_stats):
    """
    Classify based on both M1 and M2 telescope statistics using population statistics.
    Returns:
        1 for gamma
        0 for proton
        -1 if uncertain
    """
    m1_result = classify_m1_stats(stats_m1, telescope_stats)
    m2_result = classify_m2_stats(stats_m2, telescope_stats)

    # If both telescopes agree, return that classification
    if m1_result == m2_result:
        return m1_result

    # If one telescope is certain and the other uncertain, trust the certain one
    if m1_result != -1 and m2_result == -1:
        return m1_result
    if m2_result != -1 and m1_result == -1:
        return m2_result

    # If telescopes disagree (one says proton, other says gamma), return uncertain
    return -1
