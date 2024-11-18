import features
import math
import sys
import pickle
from dataclasses import dataclass

global MAX_TREE_DEPTH
global NUM_STUMPS

"""
data structure representing one node of the decision tree
"""
@dataclass
class TreeNode:
    attribute: int = None # what attribute/feature are we splitting on?
    value: bool = None # answer to the feature/attribute
    left: 'TreeNode' = None
    right: 'TreeNode' = None
    class_result: str = None  # class label only for leaf nodes

"""
converts the raw data to sentences in an array. in case of training data, appends the language to the end of the string
"""
def parse_data(filename, type):
    ds = []
    if type == "train":
        with open(filename) as file:
            for line in file:
                lang = line[:2]
                line = line[3:].strip()
                line += " " + lang
                ds.append(line)
    else:
        with open(filename) as file:
            for line in file:
                line = line.strip()
                ds.append(line)
    return ds

"""
creates feature set for input: the boolean answers to the features"""
def create_feature_set(data, type):
    feature_sets = []
    for seg in data:
        if type == "train":
            lang = seg[-2:]
            seg = seg[:-3]
        set = []
        set.append(features.nl_fcn_words(seg))
        set.append(features.nl_frq_substrings(seg))
        set.append(features.avg_word_length(seg))
        set.append(features.common_subs_2(seg))
        set.append(features.common_subs_3(seg))

        if type == "train":
            set.append(lang)

        feature_sets.append(set)

    return feature_sets

"""
calculates entropy
"""
def B(q):
    if q == 1 or q == 0:
        return 0
    else:
        return -1 * ( (q * math.log2(q)) + ((1-q) * math.log2(1-q)))

"""
calculates remainder
list data: the given dataset
int feature_idx: the feature to split data on
list weights: weights associated with the data
"""
def remainder(data, feature_idx, weights):
    r = 0
    A_true = []
    t_weights = []
    A_false = []
    f_weights = []

    # split data on given feature
    for i in range(len(data)):
        if data[i][feature_idx] == True:
            A_true.append(data[i])
            t_weights.append(weights[i])
        else:
            A_false.append(data[i])
            f_weights.append(weights[i])

    # then count the en/nl in each subdata
    # true subdata
    p_k = 0 # nl is the positive example
    n_k = 0
    for i in range(len(A_true)):
        if A_true[i][-1] == "nl":
            p_k += t_weights[i]
        else:
            n_k += t_weights[i]

    if p_k + n_k == 0:
        r += 0
    else:
        r +=  ((p_k + n_k ) / len(data)) * B(p_k / (p_k + n_k))


    # false subdata
    p_k = 0 # nl is the positive example
    n_k = 0
    for i in range(len(A_false)):
        if A_false[i][-1] == "nl":
            p_k += f_weights[i]
        else:
            n_k += f_weights[i]
    
    if p_k + n_k == 0:
        r += 0
    else:
        r +=  ((p_k + n_k ) / len(data)) * B(p_k / (p_k + n_k))
    
    return r                   


"""
calculate information gain of specific attribute
data: the given dataset
feature_idx: Attribute A that we are splitting on
weights: the weights associated with the data
"""
def info_gain(data, feature_idx, weights):
    p_ct = 0 
    # count positive examples
    for i in range(len(data)):
        if data[i][-1] == 'nl':
            p_ct += weights[i]

    prob_p = p_ct / len(data) # prob of positive 

    return B(prob_p) - remainder(data, feature_idx, weights)

"""
picks the attribute with the most info gain
data: the given dataset
weights: the weights associated with the data
"""
def pick_A(data, weights):
    gains = [] # info gains of all attributes
    for i in range(5):
        gains.append(info_gain(data, i, weights))
    max_gain_idx = 0
    max_el = gains[0]

    for i in range(1, len(gains)):
        if gains[i] > max_el:
            max_el = gains[i]
            max_gain_idx = i
    
    return max_gain_idx

"""
returns subdata of examples that answered 'req' to attribute at 'feature_idx'
int feature_idx: idx of the attribute being examined
bool req: the required answer to the attribute
list data: the given dataset
list weights: the weights associated with the data
"""
def make_subtree(feature_idx, req, dataset, weights):
    result = []
    new_weights = []
    for i in range(len(dataset)):
        if dataset[i][feature_idx] == req:
            result.append(dataset[i])
            new_weights.append(weights[i])

    return result, new_weights

"""
finds majority language in given set of data
"""
def find_result_class(subdata):
    nl_ct = 0
    en_ct = 0
    for d in subdata:
        if d[-1] == 'nl': 
            nl_ct += 1
        else:
            en_ct += 1
    
    if nl_ct > en_ct:
        return "nl"
    else:
        return "en"
    
"""
decision tree learning algorithm
list data: the feature sets of the input
int depth: maximum depth of the tree
list weights: weights associated with the data examples. adaboost is weighted while DTs have dummy weights of 1
"""
def build_tree(data, depth, weights):
    if depth == 1 or not data:
        class_result = find_result_class(data)
        return TreeNode(class_result=class_result)
    
    a = pick_A(data, weights)
    
    true_data, true_weights = make_subtree(a, True, data, weights)
    false_data, false_weights = make_subtree(a, False, data, weights)

    tree_node = TreeNode(
        attribute=a,
        value=None,
        left=None,
        right=None
    )
    
    tree_node.value = True
    tree_node.left = build_tree(true_data, depth - 1, true_weights)
    tree_node.value = False
    tree_node.right = build_tree(false_data, depth - 1, false_weights)

    return tree_node


"""
adaptive boosting learning algorithm that uses DTs as the learning algorithm.
examples: feature sets of some input
L: learning algorithm build_tree 
K: number of hypotheses(decision trees) to return
"""
def ada(examples, L, K):
    
    w = [1/len(examples) for i in range(len(examples))] # example weights
    h = [None for i in range(K)] # K hypotheses
    z = [None for i in range(K)] # K hypothesis weights

    for k in range(K):

        h[k] = L(examples, MAX_TREE_DEPTH, w)
        error = 0
        for j in range(len(examples)):
            if predict(h[k], examples[j][:-1]) != examples[j][-1]:
                error += w[j]

        for j in range(len(examples)):
            if predict(h[k], examples[j][:-1]) == examples[j][-1]:
                w[j] = w[j] * (error/(1-error))
        w = normalize(w)
        z[k] = (0.5) * math.log((1-error)/error)
        
    return z, h

"""
predicts the weighted majority for a single feature vector with the use of a list of hypotheses and their weights
list feature_set: single feature vector
list z: weights associated with hypotheses
list h: the ensemble fo hypotheses
"""
def weighted_maj(feature_set, z, h):
    nl_weights = 0
    en_weights = 0
    predictions = ["None" for i in range(len(h))]
    for i in range(len(h)):
        predictions[i] = predict(h[i], feature_set)
    for i in range(len(predictions)):
        if predictions[i] == "en":
            en_weights += z[i]
        else:
            nl_weights += z[i]

    if en_weights > nl_weights:
        return "en"
    else:
        return "nl"

"""
normalizes weights
list arr: array to normalize
"""
def normalize(arr):
    sum = 0
    for a in arr:
        sum += a
    x = 1 / sum

    for i in range(len(arr)):
        arr[i] *= x
    
    return arr

"""
traverses a model to find resulting class of a single feature vector
"""
def predict(model, feature_set):
    while model.value is not None:
        a = model.attribute
        ans = feature_set[a]

        if ans == True:
            model = model.left
        else:
            model = model.right
    
    return model.class_result

"""
serializes and saves a model using pickle
"""
def save_tree(filename, model):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    if sys.argv[1] == "train": # train mode
        example_file = sys.argv[2] # labeled examples
        hypothesisOut_file = sys.argv[3] # file to write model to
        learning_type = sys.argv[4]
        dataset = parse_data(example_file, "train")
        feature_set = create_feature_set(dataset, "train")
        if learning_type == "dt":
            MAX_TREE_DEPTH = 4
            model = build_tree(feature_set, MAX_TREE_DEPTH, [1 for j in range(len(feature_set))])
            save_tree(hypothesisOut_file, model)
        elif learning_type == "ada":
            MAX_TREE_DEPTH = 3
            NUM_STUMPS = 2
            weights, ensemble = ada(feature_set, build_tree, NUM_STUMPS)
            save_tree(hypothesisOut_file, (weights, ensemble))
    elif sys.argv[1] == "predict": # predict mode
        hypothesis = sys.argv[2] # trained decision tree
        file = sys.argv[3]

        test_data = parse_data(file, "test")
        feature_set_test = create_feature_set(test_data, "test")
        with open(hypothesis, 'rb') as f:
            model = pickle.load(f)
        if type(model) == tuple:
            weights, ensemble = model[0], model[1]
            for f in feature_set_test:
                p = weighted_maj(f, weights, ensemble)
                print(p)
        else:
            for f in feature_set_test:
                p = predict(model, f)
                print(p)
