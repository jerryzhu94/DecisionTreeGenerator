import pandas as pd
import util.id3
import util.model_tester
import util.visualizer


def main():

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        0: emails data sets
        1: playtennis data sets
        2: Modified playtennis data sets
        3-5: Edge cases: empty data sets
        6: Large data sets from census bureau database
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    files = {0: ("data/emails.csv", "data/emails.csv"),
             1: ("data/playtennis.csv", "data/playtennis.csv"),
             2: ("data/playtennis_training.csv", "data/playtennis_test.csv"),
             3: ("data/playtennis_empty.csv", "data/playtennis_empty.csv"),
             4: ("data/playtennis.csv", "data/playtennis_empty.csv"),
             5: ("data/playtennis_empty.csv", "data/playtennis.csv"),
             6: ("data/census_training.csv", "data/census_training_test.csv")}

    # Load file to DataFrame. Change index to test different data sets
    training_file, test_file = files[6]
    training_df = pd.read_csv(training_file)
    test_df = pd.read_csv(test_file)

    # Test model using information gain
    test_model(training_df, test_df, use_gain_ratio=False, use_pruning=False)


def test_model(training_df: pd.DataFrame, test_df: pd.DataFrame, use_gain_ratio: bool = False,
               use_pruning: bool = False) -> None:
    """
        1) Creates decision tree using training data set.
        2) Tests the decision tree using the test data set.
        3) Draws decision tree on pdf
    """
    title = ["Testing model using Information Gain", "Testing model using Gain Ratio"][use_gain_ratio]
    model_file_name = ["visualization/decision_tree_IG.gv", "visualization/decision_tree_GR.gv"][use_gain_ratio]
    pruning = ["", "with Pruning"][use_pruning]
    print("====================", title, pruning, "==================== ")
    # Generate decision tree
    attribute_dict = {attribute: training_df[attribute].unique() for attribute in training_df.columns.values[:-1]}
    decision_tree = util.ID3.generate_tree(training_df, attribute_dict, use_gain_ratio, use_pruning)

    # Prints an accuracy report
    util.ModelTester.test_accuracy(test_df, decision_tree)

    # Visualize the decision tree by generating a pdf of it
    attributes = training_df.columns.values[:-1]
    dot = util.Visualizer.draw_decision_tree_dictionary(decision_tree, attributes)
    dot.render(filename=model_file_name, view=True)


main()
