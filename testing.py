import lab3 as m

def test_adaboost():
    global MAX_TREE_DEPTH
    test_data = m.parse_data("big_test.dat", "test")
    feature_set_test = m.create_feature_set(test_data, "test")

    error_data = []

    for i in range(1,51):
        for j in range(1,51):
            MAX_TREE_DEPTH = i
            NUM_STUMPS = j


            z, h = m.ada(m.feature_set, m.build_tree, NUM_STUMPS)

            predictions = []
            for f in feature_set_test:
                p = m.weighted_maj(f, z, h)
                print(p)
                predictions.append(p)

            # train_langs = ["nl", "nl", "en", "en", "en", "nl", "nl", "nl", "en", "en"]
            train_langs = []
            with open("correct_classification.dat") as file:
                for line in file:
                    train_langs.append(line.strip())

            percent_correct = error_report(predictions, train_langs)

            error_data.append((MAX_TREE_DEPTH, NUM_STUMPS, percent_correct))

    max_idx = max(range(len(error_data)), key=lambda i: error_data[i][2])
    print("The highest accuracy rate of " + str(error_data[max_idx][2]) + "was with NUM_STUMPS="+str(error_data[max_idx][1])+" and MAX_TREE_DEPTH="+str(error_data[max_idx][0]))

def test_dt():
    test_data = m.parse_data("test.dat", "test")
    feature_set_test = m.create_feature_set(test_data, "test")

    error_data = []

    for i in range(1,16):
        MAX_TREE_DEPTH = i

        model = m.build_tree(m.feature_set, MAX_TREE_DEPTH, [1 for j in range(len(m.feature_set))])

        predictions = []
        for f in feature_set_test:
            p = m.predict(model, f)
            print(p)
            predictions.append(p)

        train_langs = ["nl", "nl", "en", "en", "en", "nl", "nl", "nl", "en", "en"]

        percent_correct = error_report(predictions, train_langs)

        error_data.append((MAX_TREE_DEPTH, percent_correct))

    max_idx = max(range(len(error_data)), key=lambda i: error_data[i][1])
    print("The highest accuracy rate of " + str(error_data[max_idx][1]) + "was MAX_TREE_DEPTH="+str(error_data[max_idx][0]))

def error_report(actual, expected):
    correct = 0
    total = len(actual)
    for i in range(len(actual)):
        if actual[i] == expected[i]:
            correct += 1
    
    print("The model correctly predicted " + str(correct/total*100) + "% of sentences.")
    print("The values used for this were NUM_STUMPS="+str(m.NUM_STUMPS)+" and MAX_TREE_DEPTH="+str(MAX_TREE_DEPTH))

    return correct/total*100