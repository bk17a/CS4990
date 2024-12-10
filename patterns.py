from itertools import chain, combinations


# Function to calculate the support of an itemset
def calculate_support(itemset, transactions):
    count = 0
    for transaction in transactions:
        if itemset.issubset(
            transaction
        ):  # Check if itemset is a subset of the transaction
            count += 1
    return count / len(
        transactions
    )  # Return support as the fraction of transactions containing the itemset


# Function to generate all non-empty subsets of a given itemset
def generate_subsets(itemset):
    """Generate all non-empty subsets of the itemset"""
    return list(
        chain.from_iterable(
            combinations(itemset, r) for r in range(1, len(itemset) + 1)
        )
    )


# The Apriori algorithm implementation
def apriori(transactions, min_support):
    # Step 1: Create the set of all items
    items = set()
    for transaction in transactions:
        for item in transaction:
            items.add(frozenset([item]))

    # Step 2: Generate candidate itemsets of size 1
    itemsets = list(items)
    frequent_itemsets = []

    # Step 3: Find frequent itemsets of size 1
    itemset_supports = [
        (itemset, calculate_support(itemset, transactions)) for itemset in itemsets
    ]
    frequent_itemsets.extend(
        [
            (itemset, support)
            for itemset, support in itemset_supports
            if support >= min_support
        ]
    )

    # Step 4: Generate candidate itemsets of size 2 and larger
    k = 2
    while True:
        candidate_itemsets = []
        for i in range(len(frequent_itemsets)):
            for j in range(i + 1, len(frequent_itemsets)):
                itemset1, _ = frequent_itemsets[i]
                itemset2, _ = frequent_itemsets[j]
                union_itemset = itemset1 | itemset2  # Union of the two itemsets
                if len(union_itemset) == k and union_itemset not in candidate_itemsets:
                    candidate_itemsets.append(union_itemset)

        # Calculate support for candidate itemsets
        candidate_supports = [
            (itemset, calculate_support(itemset, transactions))
            for itemset in candidate_itemsets
        ]
        frequent_itemsets_of_k = [
            (itemset, support)
            for itemset, support in candidate_supports
            if support >= min_support
        ]

        if not frequent_itemsets_of_k:
            break

        frequent_itemsets.extend(frequent_itemsets_of_k)
        k += 1

    return frequent_itemsets


# Function to generate association rules from frequent itemsets
def association_rules(
    transactions, frequent_itemsets, filter_type="all", min_confidence=0.0
):
    rules = []
    for itemset, support in frequent_itemsets:
        if len(itemset) < 2:
            continue  # Skip 1-itemsets, as they can't generate rules

        # Generate all subsets of the itemset
        subsets = generate_subsets(itemset)

        for subset in subsets:
            # Convert subset into a frozenset
            antecedent = frozenset(subset)
            consequent = itemset - antecedent  # consequent = itemset - antecedent

            if len(consequent) > 0:  # Ensure the consequent is not empty
                # Calculate the confidence of the rule
                support_antecedent = calculate_support(antecedent, transactions)
                support_consequent = calculate_support(consequent, transactions)

                if support_antecedent > 0:  # To avoid division by zero
                    confidence = support / support_antecedent

                    if confidence >= min_confidence:
                        # The `filter_type` argument is currently not used in logic.
                        rules.append((antecedent, consequent, confidence))

    return rules
