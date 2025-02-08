def is_within_std_range(value, mean, std, n_std=1):
    """Check if value is within n standard deviations of the mean"""
    lower = mean - n_std * std
    upper = mean + n_std * std
    return lower <= value <= upper


def classify_m1_stats(stats, telescope_stats):
    """
    Classify based on M1 telescope statistics using provided population statistics.
    """
    m1_proton_means = telescope_stats["m1_proton"].mean(axis=0)
    m1_proton_stds = telescope_stats["m1_proton"].std(axis=0)
    m1_gamma_means = telescope_stats["m1_gamma"].mean(axis=0)
    m1_gamma_stds = telescope_stats["m1_gamma"].std(axis=0)

    proton_matches = 0
    gamma_matches = 0

    for i in range(len(stats)):
        value = stats[i]

        if is_within_std_range(value, m1_proton_means[i], m1_proton_stds[i]):
            proton_matches += 1
        if is_within_std_range(value, m1_gamma_means[i], m1_gamma_stds[i]):
            gamma_matches += 1

    if proton_matches > gamma_matches:
        return 0
    elif gamma_matches > proton_matches:
        return 1

    return -1


def classify_m2_stats(stats, telescope_stats):
    """
    Classify based on M2 telescope statistics using provided population statistics.
    """
    m2_proton_means = telescope_stats["m2_proton"].mean(axis=0)
    m2_proton_stds = telescope_stats["m2_proton"].std(axis=0)
    m2_gamma_means = telescope_stats["m2_gamma"].mean(axis=0)
    m2_gamma_stds = telescope_stats["m2_gamma"].std(axis=0)

    proton_matches = 0
    gamma_matches = 0

    for i in range(len(stats)):
        value = stats[i]

        if is_within_std_range(value, m2_proton_means[i], m2_proton_stds[i]):
            proton_matches += 1
        if is_within_std_range(value, m2_gamma_means[i], m2_gamma_stds[i]):
            gamma_matches += 1

    if proton_matches > gamma_matches:
        return 0
    elif gamma_matches > proton_matches:
        return 1

    return -1


def rule_based_derivation_classifier(stats_m1, stats_m2, telescope_stats):
    """
    Classify based on both M1 and M2 telescope statistics using provided population statistics.
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
