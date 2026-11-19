import numpy as np

def apriori(dataset, min_support=0.5, verbose=False):
    """Implements the Apriori algorithm.

    The Apriori algorithm will iteratively generate new candidate
    k-itemsets using the frequent (k-1)-itemsets found in the previous
    iteration.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate
        candidate itemsets.

    min_support : float
        The minimum support threshold. Defaults to 0.5.

    Returns
    -------
    F : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.

    References
    ----------
    .. [1] R. Agrawal, R. Srikant, "Fast Algorithms for Mining Association
           Rules", 1994.

    """
    C1 = create_candidates(dataset)
    D = list(map(set, dataset))
    F1, support_data = get_freq(D, C1, min_support, verbose=False) # get frequent 1-itemsets
    F = [F1] # list of frequent itemsets; initialized to frequent 1-itemsets
    k = 2 # the itemset cardinality
    while (len(F[k - 2]) > 0):
        Ck = apriori_gen(F[k-2], k) # generate candidate itemsets
        Fk, supK  = get_freq(D, Ck, min_support) # get frequent itemsets
        support_data.update(supK)# update the support counts to reflect pruning
        F.append(Fk)  # add the frequent k-itemsets to the list of frequent itemsets
        k += 1

    if verbose:
        # Print a list of all the frequent itemsets.
        for kset in F:
            for item in kset:
                print(""                     + "{"                     + "".join(str(i) + ", " for i in iter(item)).rstrip(', ')                     + "}"                     + ":  sup = " + str(round(support_data[item], 3)))

    return F, support_data

def create_candidates(dataset, verbose=False):
    """Creates a list of candidate 1-itemsets from a list of transactions.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    Returns
    -------
    The list of candidate itemsets (c1) passed as a frozenset (a set that is
    immutable and hashable).
    """
    c1 = [] # list of all items in the database of transactions
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()

    if verbose:
        # Print a list of all the candidate items.
        print(""             + "{"             + "".join(str(i[0]) + ", " for i in iter(c1)).rstrip(', ')             + "}")

    # Map c1 to a frozenset because it will be the key of a dictionary.
    return list(map(frozenset, c1))

def get_freq(dataset, candidates, min_support, verbose=False):
    """

    This function separates the candidates itemsets into frequent itemset and infrequent itemsets based on the min_support,
	and returns all candidate itemsets that meet a minimum support threshold.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    candidates : frozenset
        The list of candidate itemsets.

    min_support : float
        The minimum support threshold.

    Returns
    -------
    freq_list : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.
    """

    freq_list = []
    support_data = {}
    minsup = min_support * len(dataset)

    for candidate in candidates:
        support_count = 0
        for item in dataset:

            if candidate.issubset(item):
                support_count += 1

        if support_count >= minsup:
            freq_list.append(candidate)
            support_data[candidate] = support_count

    if len(candidates) > 0:
        print("Number of length-"+str(len(candidates[0]))+ " Frequent Itemsets ")
        print(len(freq_list))
        print(freq_list)
    

    return freq_list, support_data
    

    

    

def apriori_gen(freq_sets, k):
    """Generates candidate itemsets (via the F_k-1 x F_k-1 method).

    This part generates new candidate k-itemsets based on the frequent
    (k-1)-itemsets found in the previous iteration.

    The apriori_gen function performs two operations:
    (1) Generate length k candidate itemsets from length k-1 frequent itemsets
    (2) Prune candidate itemsets containing subsets of length k-1 that are infrequent

    Parameters
    ----------
    freq_sets : list
        The list of frequent (k-1)-itemsets.

    k : integer
        The cardinality of the current itemsets being evaluated.

    Returns
    -------
    candidate_list : list
        The list of candidate itemsets.
    """
    # Fk-1 * Fk-1 method, merge if their first k-2 items are identical, start with k = 2

    
    candidate_list = []
    candidate_list_tmp = []
    F_sorted = [sorted(list(itemset)) for itemset in freq_sets]


    for i in range(len(F_sorted)):
        for j in range(i + 1, len(F_sorted)):
            if F_sorted[i][:k-2] == F_sorted[j][:k-2]:
                merged_set = frozenset(F_sorted[i]) | frozenset(F_sorted[j])
                candidate_list_tmp.append(merged_set)


    # pruning: generate fk-1 subsets of each candidates, compare them to freq_sets
    freq_set_real = set(freq_sets)
    for candidate in candidate_list_tmp:
        is_valid = True
        subsets = combinations(candidate, k - 1)

        for subset in subsets:
            frozen_subset = frozenset(subset)

            # pruning: if the subset is not in Fk-1 -> infrequent
            if frozen_subset not in freq_sets:
                is_valid = False
                break
        if is_valid:
            candidate_list.append(candidate)
    print("Number of length-" + str(k) + " Candidate Itemsets")
    print(len(candidate_list))
    print(candidate_list)
    print("----------------------------------------------------\n")
    return candidate_list

def run_apriori(dataset, min_support, verbose=False):
    F, support = apriori(dataset, min_support=min_support, verbose=verbose)
    return F, support