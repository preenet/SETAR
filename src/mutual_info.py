# mutual_info.py

from math import log2
import numpy as np
import pandas as pd
from pytest import approx
from tabulate import tabulate


def calc_pmi(series):
    global px, py

    prob = series['prob']
    if prob == 0:
        prob = np.nextafter(0, 1)  # Smallest positive number after zero towards 1
    pmi = log2(prob) - log2(px[series['x']] * py[series['y']])
    return pmi


def calc_mi(series):
    mi = calc_pmi(series) * series['prob']
    return mi


def calc_joint_entropy(series):
    prob = series['prob']
    if prob == 0:
        return 0
    je = -1 * prob * log2(prob)
    return je


def calc_entropy(marginals : dict):
    """
    Given a dict of marginal probabilities calculate entropy
    :param marginals:
    :return:
    """
    h = 0
    for p in marginals:
        if marginals[p] == 0:
            continue
        h += marginals[p] * log2(marginals[p])
    h *= -1
    return h


def calc_conditional_entropy_x(df, x_value):
    vec_df = df[df['x'] == x_value].copy()
    vec_df['prob'] = vec_df['prob'] / vec_df['prob'].sum()

    h = 0
    for prob in vec_df['prob']:
        if prob != 0:
            h += prob * log2(prob)
    if h != 0:
        h *= -1
    return h


def calc_conditional_entropy_over_all_x(df, px):
    h = 0
    gs = df.groupby('x').groups
    for x in gs:
        h += px[x] * calc_conditional_entropy_x(df, x)
    return h


def calc_mutual_information_using_joint_entropy(df, hx, hy):
    df['je'] = df.apply(calc_joint_entropy, axis=1)
    je = df['je'].sum()
    mut_info = hx + hy - je
    return mut_info


def calc_mutual_information_using_cond_entropy(df, px, py):
    """
    :param df: data frame
    :param px: dictionary of marginal probabilities for x
    :param py: dictionary of marginal probabilities for y
    :return: mutual information
    """
    hy = calc_entropy(py)
    h = calc_conditional_entropy_over_all_x(df, px)
    mi = hy - h
    return mi


def calc_mutual_information_for_word(df, use_cond_entropy,
                                     cond_entropy_method=calc_mutual_information_using_cond_entropy):
    debug = False

    # Calc marginal probabilities for x
    gs = df.groupby('x').groups
    px = {}
    for g in gs:
        px[g] = df.iloc[gs[g]].sum()['prob']

    # Calc marginal probabilities for y
    gs = df.groupby('y').groups
    py = {}
    for g in gs:
        py[g] = df.iloc[gs[g]].sum()['prob']

    hx = calc_entropy(px)
    hy = calc_entropy(py)

    if debug:
        print(f"px marginals: {px}")
        print(f"py marginals: {py}")
        print(df)
        print(f"H(X): {hx}  H(Y): {hy} ")

    mi = 0
    if use_cond_entropy:
        # cond_entropy_method is either: calc_mutual_information_using_cond_entropy_decomp_y or
        #                                calc_mutual_information_using_cond_entropy
        mi = cond_entropy_method(df, px, py)
    else:
        mi = calc_mutual_information_using_joint_entropy(df, hx, hy)

    return mi


def insert_negation_rows(word_df, class_count_residuals):
    num_rows = word_df.shape[0]
    new_arr = []
    num_class_labels = len(class_count_residuals)
    # The case where not all combinations are already specified
    if num_rows != num_class_labels * 2:
        for index, row in word_df.iterrows():
            # Append row with the negative feature & probability
            new_arr.append(['not ' + row['x'], row['y'], class_count_residuals[row['y']] - row['prob']])

        new_df = pd.DataFrame(new_arr, columns=['x', 'y', 'prob'])
        word_df = pd.concat([word_df, new_df])
        word_df.reset_index(inplace=True, drop=True)
    else:
        return word_df

    return word_df


def calc_mutual_information(full_df, feat_names, class_count_residuals, use_cond_entropy=True,
                            cond_entropy_method=calc_mutual_information_using_cond_entropy):
    """
    For each feature (x) add the corresponding absence of the feature probabilities
    Resulting data frame should be 2 * # of class labels rows in length
    :param class_count_residuals:  a dictionary for each class label showing the class counts e.g.
                                 {'art' : 57, 'music' : 45 }
    :param df:
    :param use_cond_entropy:
    :return: an array of mutual information on value per feature
    """

    gs = full_df.groupby('y').groups
    num_class_labels = len(gs.keys())
    mut_infos = []

    # Order has to be same as the order of the feat_names passed to this function
    for feature in feat_names:
        # TEMP remove this
        # if feature.startswith('not '):
        #     continue

        word_df = full_df[full_df['x'] == feature].copy()
        word_df.reset_index(inplace = True, drop = True)
        num_rows = word_df.shape[0]

        if num_rows == 0:
            print(f"Feature {feature} not found in counts! Assuming zero mutual info")
            mut_infos.append(0.0)
            continue

        word_df = insert_negation_rows(word_df, class_count_residuals)

        num_rows = word_df.shape[0]
        if num_rows != num_class_labels * 2:
            raise("num_rows is not equal to num_class_labels * 2")

        word_df['prob'] = word_df['prob'] / word_df['prob'].sum()
        mi = calc_mutual_information_for_word(word_df, use_cond_entropy, cond_entropy_method)
        mut_infos.append(mi)
    return mut_infos

"""
*** Decomposition over y is analagous to decomposition over x
*** The following three functions are not used and are just for illustrative purposes
"""

def calc_conditional_entropy_y(df, y_value):
    vec_df = df[df['y'] == y_value].copy()
    vec_df['prob'] = vec_df['prob'] / vec_df['prob'].sum()

    h = 0
    for prob in vec_df['prob']:
        if prob != 0:
            h += prob * log2(prob)
    if h != 0:
        h *= -1
    return h


def calc_conditional_entropy_over_all_y(df, py):
    """
    Calculate the decomposition entropy for each class label in y
    Return the total plus a dictionary of the decomposed results
    :param df:
    :param py:
    :return:
    """
    h = 0
    res = {}
    gs = df.groupby('y').groups
    for y in gs:
        res[y] = py[y] * calc_conditional_entropy_y(df, y)
        h += res[y]
    return h, res

def calc_mutual_information_using_cond_entropy_decomp_y(df, px, py):
    """
    :param df: data frame
    :param px: dictionary of marginal probabilities for x
    :param py: dictionary of marginal probabilities for y
    :return: mutual information
    """
    hx = calc_entropy(px)
    h, decomp_entropy = calc_conditional_entropy_over_all_y(df, py)
    mi = hx - h
    print(df)
    print(f"H(x): {hx} Decomposed entropy: {decomp_entropy} Sum of decomposition: {h}")
    return mi

""" 
***
*** The previous three functions are not currently used, and are just for illustrative purposes
***
"""

def prob_y_given_x(df, x_value, px, py):
    """
    Calculation the conditional probabilities of y given x_value
    :param df: the data frame containing the JPD for x_value
    :param px: the residuals of x
    :return: a dict for keyed on y with p(y | x) as the values
    """
    py_x = {}
    for y in py:
        prob_x = df[(df['x'] == x_value) & (df['y'] == y)]['prob'].values[0]
        py_x[y] = prob_x / px[x_value]

    return py_x


def conditional_probability_for_y_given_partial_jpd(full_df, feat_names, class_count_residuals=None):


    ### Flesh out the JPD
    if not class_count_residuals:
        class_count_residuals = {}
        gs = full_df.groupby('y').groups
        for y in gs:
            class_count_residuals[y] = len(gs[y])

    gs = full_df.groupby('y').groups
    num_class_labels = len(gs.keys())
    # A dict of dict of conditional probabilities - at the top level each feature is the key
    # and at the sublevel each class label is the key the value is the probability
    cond_probs_feats = {}

    # Order has to be same as the order of the feat_names passed to this function
    for feature in feat_names:
        if feature == 'love' or feature == 'lose':
            print("Got here")

        word_df = full_df[full_df['x'] == feature].copy()
        word_df.reset_index(inplace = True, drop = True)
        num_rows = word_df.shape[0]

        if num_rows == 0:
            print(f"Feature {feature} not found in counts! Assuming zero probabilities")
            continue

        word_df = insert_negation_rows(word_df, class_count_residuals)

        num_rows = word_df.shape[0]
        if num_rows != num_class_labels * 2:
            raise("num_rows is not equal to num_class_labels * 2")

        # Normalize the probabilities
        word_df['prob'] = word_df['prob'] / word_df['prob'].sum()

        if feature == 'love':
            print("got here")

        # Calc marginal probabilities for x
        gs = word_df.groupby('x').groups
        px = {}
        for g in gs:
            px[g] = word_df.iloc[gs[g]].sum()['prob']

        # Calc marginal probabilities for y
        gs = word_df.groupby('y').groups
        py = {}
        for g in gs:
            py[g] = word_df.iloc[gs[g]].sum()['prob']

        py_x = prob_y_given_x(word_df, feature, px, py)
        cond_probs_feats[feature] = py_x

    return cond_probs_feats