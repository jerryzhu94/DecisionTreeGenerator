import pandas as pd
from enum import Enum


class ModelTester:
    """ Public Method """

    # Tests the accuracy of the model with the test DataFrame and prints a report
    @staticmethod
    def test_accuracy(test_df: pd.DataFrame, decision_tree: {}) -> None:
        if len(test_df) == 0:
            raise ValueError("The DataFrame is empty")
        if not decision_tree:
            raise ValueError("The decision tree is empty")
        corrects, incorrects, unknowns = 0, 0, 0
        # Classify each instance in the DataFrame
        for index, instance in test_df.iterrows():
            classification = ModelTester.__classify_instance(instance, decision_tree)
            if classification == ModelTester.__Classification.CORRECT:
                corrects += 1
            elif classification == ModelTester.__Classification.INCORRECT:
                incorrects += 1
            else:
                unknowns += 1
        ModelTester.__print_report(corrects, incorrects, unknowns)

    """ Private Helper Methods """

    # Returns an enumeration that indicates whether the classification is correct, incorrect, or unknown
    @staticmethod
    def __classify_instance(instance: pd.Series, decision_tree: {}) -> Enum:
        # Traverses down the decision tree until a leaf node has been hit
        while isinstance(decision_tree, dict):
            attribute = next(iter(decision_tree))
            instance_attribute_value = instance[attribute]
            """ If the instance is at an attribute node that does not offer the attribute value of that instance, 
            the instance has hit a dead end. Hence, it can not be classified """
            if instance_attribute_value not in decision_tree[attribute]:
                return ModelTester.__Classification.UNKNOWN
            decision_tree = decision_tree[attribute][instance_attribute_value]
        # Compare the decision tree's classification with the instance's class label
        return ModelTester.__Classification.CORRECT if decision_tree == instance[-1] \
            else ModelTester.__Classification.INCORRECT

    # Prints accuracy report
    @staticmethod
    def __print_report(corrects: int, incorrects: int, unknowns: int) -> None:
        total = corrects + incorrects + unknowns
        knowns = corrects + incorrects
        if knowns == 0:
            raise ValueError("None of the instances were classified using the model")
        accuracy = corrects / knowns * 100
        print("==================== TEST STARTED ============================================ ")
        print("Number of testing examples =", total)
        print("Number of testing examples classified =", knowns)
        print("Number of testing examples not classified =", unknowns)
        print("Correct classification count =", corrects)
        print("Incorrect classification count =", incorrects)
        print("Accuracy =", accuracy, "%")
        print("===================== TEST ENDED ============================================== ")

    """ Private Field """
    __Classification = Enum("Classification", "CORRECT INCORRECT UNKNOWN")
