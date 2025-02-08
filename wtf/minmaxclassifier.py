# Constants for M1 Proton
M1_PROTON_MEAN_MIN = 2.250409
M1_PROTON_MEAN_MAX = 655.800293
M1_PROTON_STD_MIN = 1.228611
M1_PROTON_STD_MAX = 419.023468
M1_PROTON_NEG_RATIO_MIN = 0.000000
M1_PROTON_NEG_RATIO_MAX = 0.015399
M1_PROTON_MIN_MIN = -1.246094
M1_PROTON_MIN_MAX = 41.625000
M1_PROTON_MAX_MIN = 7.812500
M1_PROTON_MAX_MAX = 1172.000000
M1_PROTON_SQUARED_MEAN_MIN = 6.743633
M1_PROTON_SQUARED_MEAN_MAX = 562412.812500
M1_PROTON_NONZERO_RATIO_MIN = 0.999038
M1_PROTON_NONZERO_RATIO_MAX = 1.000000
M1_PROTON_Q25_MIN = 1.333984
M1_PROTON_Q25_MAX = 282.500000
M1_PROTON_Q50_MIN = 2.023438
M1_PROTON_Q50_MAX = 744.000000
M1_PROTON_Q75_MIN = 2.812500
M1_PROTON_Q75_MAX = 996.000000

# Constants for M2 Proton
M2_PROTON_MEAN_MIN = 2.637800
M2_PROTON_MEAN_MAX = 792.953308
M2_PROTON_STD_MIN = 1.617415
M2_PROTON_STD_MAX = 415.691406
M2_PROTON_NEG_RATIO_MIN = 0.000000
M2_PROTON_NEG_RATIO_MAX = 0.014437
M2_PROTON_MIN_MIN = -1.316406
M2_PROTON_MIN_MAX = 102.000000
M2_PROTON_MAX_MIN = 9.593750
M2_PROTON_MAX_MAX = 1232.000000
M2_PROTON_SQUARED_MEAN_MIN = 9.862174
M2_PROTON_SQUARED_MEAN_MAX = 734487.125000
M2_PROTON_NONZERO_RATIO_MIN = 1.000000
M2_PROTON_NONZERO_RATIO_MAX = 1.000000
M2_PROTON_Q25_MIN = 1.431641
M2_PROTON_Q25_MAX = 523.000000
M2_PROTON_Q50_MIN = 2.210938
M2_PROTON_Q50_MAX = 970.000000
M2_PROTON_Q75_MIN = 3.246094
M2_PROTON_Q75_MAX = 1032.000000

# Constants for M1 Gamma
M1_GAMMA_MEAN_MIN = 1.893804
M1_GAMMA_MEAN_MAX = 323.330963
M1_GAMMA_STD_MIN = 1.027410
M1_GAMMA_STD_MAX = 340.173889  # Updated from 325.544250
M1_GAMMA_NEG_RATIO_MIN = 0.000000
M1_GAMMA_NEG_RATIO_MAX = 0.007700
M1_GAMMA_MIN_MIN = -0.667969
M1_GAMMA_MIN_MAX = 16.375000  # Updated from 15.375000
M1_GAMMA_MAX_MIN = 7.625000  # Updated from 7.750000
M1_GAMMA_MAX_MAX = 1168.000000
M1_GAMMA_SQUARED_MEAN_MIN = 4.783196
M1_GAMMA_SQUARED_MEAN_MAX = 183964.906250
M1_GAMMA_NONZERO_RATIO_MIN = 1.000000
M1_GAMMA_NONZERO_RATIO_MAX = 1.000000
M1_GAMMA_Q25_MIN = 1.113281
M1_GAMMA_Q25_MAX = 95.000000
M1_GAMMA_Q50_MIN = 1.667969
M1_GAMMA_Q50_MAX = 245.500000
M1_GAMMA_Q75_MIN = 2.335938  # Updated from 2.347656
M1_GAMMA_Q75_MAX = 492.500000

# Constants for M2 Gamma
M2_GAMMA_MEAN_MIN = 2.224912  # Updated from 2.245055
M2_GAMMA_MEAN_MAX = 561.004944
M2_GAMMA_STD_MIN = 1.151883  # Updated from 1.181835
M2_GAMMA_STD_MAX = 378.356842
M2_GAMMA_NEG_RATIO_MIN = 0.000000
M2_GAMMA_NEG_RATIO_MAX = 0.008662
M2_GAMMA_MIN_MIN = -0.886719
M2_GAMMA_MIN_MAX = 31.875000
M2_GAMMA_MAX_MIN = 8.281250
M2_GAMMA_MAX_MAX = 1228.000000
M2_GAMMA_SQUARED_MEAN_MIN = 6.275792  # Updated from 6.483352
M2_GAMMA_SQUARED_MEAN_MAX = 445398.406250
M2_GAMMA_NONZERO_RATIO_MIN = 1.000000
M2_GAMMA_NONZERO_RATIO_MAX = 1.000000
M2_GAMMA_Q25_MIN = 1.343750
M2_GAMMA_Q25_MAX = 190.500000
M2_GAMMA_Q50_MIN = 2.000000
M2_GAMMA_Q50_MAX = 540.000000
M2_GAMMA_Q75_MIN = 2.734375
M2_GAMMA_Q75_MAX = 938.000000

def classify_m1_stats(stats):
    """
    Classify based on M1 telescope statistics.
    Returns:
        1 for gamma
        0 for proton
        -1 if uncertain
    """
    mean, std, neg_ratio, min_val, max_val, squared_mean, _, q25, q50, q75 = stats

    # Check if it's definitely a proton (outside gamma ranges)
    if (mean < M1_GAMMA_MEAN_MIN or mean > M1_GAMMA_MEAN_MAX or
            std < M1_GAMMA_STD_MIN or std > M1_GAMMA_STD_MAX or
            neg_ratio < M1_GAMMA_NEG_RATIO_MIN or neg_ratio > M1_GAMMA_NEG_RATIO_MAX or
            min_val < M1_GAMMA_MIN_MIN or min_val > M1_GAMMA_MIN_MAX or
            max_val < M1_GAMMA_MAX_MIN or max_val > M1_GAMMA_MAX_MAX or
            squared_mean < M1_GAMMA_SQUARED_MEAN_MIN or squared_mean > M1_GAMMA_SQUARED_MEAN_MAX or
            q25 < M1_GAMMA_Q25_MIN or q25 > M1_GAMMA_Q25_MAX or
            q50 < M1_GAMMA_Q50_MIN or q50 > M1_GAMMA_Q50_MAX or
            q75 < M1_GAMMA_Q75_MIN or q75 > M1_GAMMA_Q75_MAX):
        return 0

    # Check if it's definitely a gamma (outside proton ranges)
    if (mean < M1_PROTON_MEAN_MIN or mean > M1_PROTON_MEAN_MAX or
            std < M1_PROTON_STD_MIN or std > M1_PROTON_STD_MAX or
            neg_ratio < M1_PROTON_NEG_RATIO_MIN or neg_ratio > M1_PROTON_NEG_RATIO_MAX or
            min_val < M1_PROTON_MIN_MIN or min_val > M1_PROTON_MIN_MAX or
            max_val < M1_PROTON_MAX_MIN or max_val > M1_PROTON_MAX_MAX or
            squared_mean < M1_PROTON_SQUARED_MEAN_MIN or squared_mean > M1_PROTON_SQUARED_MEAN_MAX or
            q25 < M1_PROTON_Q25_MIN or q25 > M1_PROTON_Q25_MAX or
            q50 < M1_PROTON_Q50_MIN or q50 > M1_PROTON_Q50_MAX or
            q75 < M1_PROTON_Q75_MIN or q75 > M1_PROTON_Q75_MAX):
        return 1

    return -1


def classify_m2_stats(stats):
    """
    Classify based on M2 telescope statistics.
    Returns:
        1 for gamma
        0 for proton
        -1 if uncertain
    """
    mean, std, neg_ratio, min_val, max_val, squared_mean, _, q25, q50, q75 = stats

    # Check if it's definitely a proton (outside gamma ranges)
    if (mean < M2_GAMMA_MEAN_MIN or mean > M2_GAMMA_MEAN_MAX or
            std < M2_GAMMA_STD_MIN or std > M2_GAMMA_STD_MAX or
            neg_ratio < M2_GAMMA_NEG_RATIO_MIN or neg_ratio > M2_GAMMA_NEG_RATIO_MAX or
            min_val < M2_GAMMA_MIN_MIN or min_val > M2_GAMMA_MIN_MAX or
            max_val < M2_GAMMA_MAX_MIN or max_val > M2_GAMMA_MAX_MAX or
            squared_mean < M2_GAMMA_SQUARED_MEAN_MIN or squared_mean > M2_GAMMA_SQUARED_MEAN_MAX or
            q25 < M2_GAMMA_Q25_MIN or q25 > M2_GAMMA_Q25_MAX or
            q50 < M2_GAMMA_Q50_MIN or q50 > M2_GAMMA_Q50_MAX or
            q75 < M2_GAMMA_Q75_MIN or q75 > M2_GAMMA_Q75_MAX):
        return 0

    # Check if it's definitely a gamma (outside proton ranges)
    if (mean < M2_PROTON_MEAN_MIN or mean > M2_PROTON_MEAN_MAX or
            std < M2_PROTON_STD_MIN or std > M2_PROTON_STD_MAX or
            neg_ratio < M2_PROTON_NEG_RATIO_MIN or neg_ratio > M2_PROTON_NEG_RATIO_MAX or
            min_val < M2_PROTON_MIN_MIN or min_val > M2_PROTON_MIN_MAX or
            max_val < M2_PROTON_MAX_MIN or max_val > M2_PROTON_MAX_MAX or
            squared_mean < M2_PROTON_SQUARED_MEAN_MIN or squared_mean > M2_PROTON_SQUARED_MEAN_MAX or
            q25 < M2_PROTON_Q25_MIN or q25 > M2_PROTON_Q25_MAX or
            q50 < M2_PROTON_Q50_MIN or q50 > M2_PROTON_Q50_MAX or
            q75 < M2_PROTON_Q75_MIN or q75 > M2_PROTON_Q75_MAX):
        return 1

    return -1


def rule_based_minmax_classifier(stats_m1, stats_m2):
    """
    Classify based on both M1 and M2 telescope statistics.
    Returns:
        1 for gamma
        0 for proton
        -1 if uncertain
    """
    m1_result = classify_m1_stats(stats_m1)
    m2_result = classify_m2_stats(stats_m2)

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
