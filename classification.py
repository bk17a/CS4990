import numpy as np
import math


# These are suggested helper functions
# You can structure your code differently, but if you have
# trouble getting started, this might be a good starting point


# Create the decision tree recursively
def make_node(previous_ys, xs, ys, columns):
    # WARNING: lists are passed by reference in python
    # If you are planning to remove items, it's better
    # to create a copy first
    columns = columns[:]

    # First, check the three termination criteria:

    # If there are no rows (xs and ys are empty):
    #      Return a node that classifies as the majority class of the parent
    if not xs or not ys:
        return {"type": "class", "class": majority(previous_ys)}

    # If all ys are the same:
    #      Return a node that classifies as that class
    if same(ys):
        return {"type": "class", "class": ys[0]}

    # If there are no more columns left:
    #      Return a node that classifies as the majority class of the ys
    if not columns:
        return {"type": "class", "class": majority(ys)}

    # Otherwise:
    # Compute the entropy of the current ys
    current_entropy = entropy(ys)
    best_gain = -1
    best_column = None
    best_splits = None

    # For each column:
    for column in columns:
        splits = {}
        for x, y in zip(xs, ys):
            value = x[column]
            if value not in splits:
                splits[value] = {"xs": [], "ys": []}
            splits[value]["xs"].append(x)
            splits[value]["ys"].append(y)

        # Calculate the entropy of each of the pieces
        # Compute the overall entropy as the weighted sum
        split_entropy = sum(
            len(split["ys"]) / len(ys) * entropy(split["ys"])
            for split in splits.values()
        )
        # The gain of the column is the difference of the entropy before
        #    the split, and this new overall entropy
        gain = current_entropy - split_entropy

        # Select the column with the highest gain
        if gain > best_gain:
            best_gain = gain
            best_column = column
            best_splits = splits

    # If no gain, return a leaf node with majority class
    if best_gain <= 0:
        return {"type": "class", "class": majority(ys)}

    # Split the data along the column values and recursively call
    #    make_node for each piece
    # Create a split-node that splits on this column, and has the result
    #    of the recursive calls as children.
    node = {"type": "split", "attribute": best_column, "children": {}}
    remaining_columns = [col for col in columns if col != best_column]

    for value, split in best_splits.items():
        node["children"][value] = make_node(
            ys, split["xs"], split["ys"], remaining_columns
        )

    return node


# Determine if all values in a list are the same
# Useful for the second basecase above
def same(values):
    if not values:
        return True
    # if there are values:
    first_val = values[0]
    # pick the first, check if all other are the same
    return all(val == first_val for val in values)


# Determine how often each value shows up
# in a list; this is useful for the entropy
# but also to determine which values is the
# most common
def counts(values):
    count_dict = {}
    for value in values:
        if value in count_dict:
            count_dict[value] += 1
        else:
            count_dict[value] = 1
    return count_dict


# Return the most common value from a list
# Useful for base cases 1 and 3 above.
def majority(values):
    count_dict = counts(values)
    return max(count_dict, key=count_dict.get)


# Calculate the entropy of a set of values
# First count how often each value shows up
# When you divide this value by the total number
# of elements, you get the probability for that element
# The entropy is the negation of the sum of p*log2(p)
# for all these probabilities.
def entropy(values):
    count_dict = counts(values)
    total = len(values)
    ent = 0
    for count in count_dict.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent


# This is the main decision tree class
# DO NOT CHANGE THE FOLLOWING LINE
class DecisionTree:
    # DO NOT CHANGE THE PRECEDING LINE
    def __init__(self, tree={}):
        self.tree = tree

    # DO NOT CHANGE THE FOLLOWING LINE
    def fit(self, x, y):
        # DO NOT CHANGE THE PRECEDING LINE

        self.majority = majority(y)
        self.tree = make_node(y, x, y, list(range(len(x[0]))))

    # DO NOT CHANGE THE FOLLOWING LINE
    def predict(self, x):
        # DO NOT CHANGE THE PRECEDING LINE
        if not self.tree:
            return None

        # To classify using the tree:
        # Start with the root as the "current" node
        # As long as the current node is an interior node (type == "split"):
        #    get the value of the attribute the split is performed on
        #    select the child corresponding to that value as the new current node

        # NOTE: In some cases, your tree may not have a child for a particular value
        #       In that case, return the majority value (self.majority) from the training set

        # IMPORTANT: You have to perform this classification *for each* element in x

        predictions = []
        for instance in x:
            current_node = self.tree
            while current_node["type"] == "split":
                attribute = current_node["attribute"]
                value = instance[attribute]
                if value in current_node["children"]:
                    current_node = current_node["children"][value]
                else:
                    current_node = {"type": "class", "class": self.majority}
            predictions.append(current_node["class"])
        return predictions

    # DO NOT CHANGE THE FOLLOWING LINE
    def to_dict(self):
        # DO NOT CHANGE THE PRECEDING LINE
        # change this if you store the tree in a different format
        return self.tree
