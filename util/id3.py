import pandas as pd
import numpy as np


class ID3:
    """ Public Method """

    # Uses the ID3 algorithm to generates the decision tree
    @staticmethod
    def generate_tree(df: pd.DataFrame, attribute_dict: {}, use_gain_ratio: bool, use_pruning: bool) -> {}:
        if len(df) == 0:
            return {}
        candidates = list(df.columns.values[:-1])
        ID3.__instance_threshold = len(df) // 100
        return ID3.__generate_tree(df, candidates, attribute_dict, use_gain_ratio, use_pruning)

    """ Private Helper Methods """

    # Helps generates the tree by recursively creating dictionaries
    @staticmethod
    def __generate_tree(df: pd.DataFrame, candidates: list, attribute_dict: {}, use_gain_ratio: bool,
                        use_pruning: bool) -> {}:
        class_label = df.columns[-1]
        # Base case #1: No more attributes to partition over. Return most common class label
        if not candidates:
            return df[class_label].value_counts().idxmax()
        # Base case #2: Entropy has hit 0. All instances in the DataFrame have the same class label
        if not ID3.__entropy(df):
            return df[class_label].iloc[0]
        # Base case #3: Pre-pruning
        if use_pruning and len(df) < ID3.__instance_threshold :
            return df[class_label].value_counts().idxmax()
        # Generate tree by finding the next optimal attribute to partition over
        tree, attribute_tree = {}, {}
        next_attribute = ID3.__next_attribute(df, candidates, use_gain_ratio)
        updated_candidates = candidates.copy()
        updated_candidates.remove(next_attribute)
        for attribute_value in attribute_dict[next_attribute]:
            partitioned_df = df[df[next_attribute] == attribute_value]
            # Recurse only if partitioned DataFrame is not empty
            if len(partitioned_df) != 0:
                attribute_tree[attribute_value] = ID3.__generate_tree(partitioned_df, updated_candidates,
                                                                      attribute_dict, use_gain_ratio, use_pruning)
            # If partitioned DataFrame is empty, return most common class label
            else:
                attribute_tree[attribute_value] = df[class_label].value_counts().idxmax()
        tree[next_attribute] = attribute_tree
        return tree

    # Get the attribute that yields the highest gain
    @staticmethod
    def __next_attribute(df: pd.DataFrame, candidates: list, use_gain_ratio: bool) -> str:
        attribute_gains = {}
        for attribute in candidates:
            if use_gain_ratio:
                attribute_gains[attribute] = ID3.__gain_ratio(df, attribute)
            else:
                attribute_gains[attribute] = ID3.__information_gain(df, attribute)
        return max(attribute_gains, key=attribute_gains.get)

    # Calculates the information gain acquired from partitioning over attribute
    @staticmethod
    def __information_gain(df: pd.DataFrame, attribute: str) -> float:
        if len(df) == 0:
            return 0
        # Calculate entropy before partitioning
        entropy_before = ID3.__entropy(df)
        # Calculate entropy after partitioning
        entropy_after = 0
        for attribute_value in df[attribute].unique():
            mask = df[attribute] == attribute_value
            partitioned_df = df[mask]
            entropy_after += len(partitioned_df) / len(df) * ID3.__entropy(partitioned_df)
        # Information gain = entropy before decision - entropy after decision
        return entropy_before - entropy_after

    # Calculates the entropy of DataFrame
    @staticmethod
    def __entropy(df: pd.DataFrame) -> float:
        # Get the class ratios by dividing the number of class instances by the total number of instances
        class_ratios = []
        class_label = df.columns[-1]
        total_instances = len(df.index)
        for class_value in df[class_label].unique():
            mask = df[class_label] == class_value
            partitioned_df = df[mask]
            partitioned_instances = len(partitioned_df.index)
            class_ratios.append(partitioned_instances / total_instances)
        # Entropy equation
        return sum([-class_ratio * np.log2(class_ratio) for class_ratio in class_ratios])

    # Calculates the gain ratio acquired from partitioning over attribute
    @staticmethod
    def __gain_ratio(df: pd.DataFrame, attribute: str) -> float:
        information_gain = ID3.__information_gain(df, attribute)
        split_information = ID3.__split_information(df, attribute)
        return information_gain / split_information

    # Calculates the split information
    @staticmethod
    def __split_information(df: pd.DataFrame, attribute: str) -> float:
        attribute_ratios = []
        total_instances = len(df)
        for attribute_value in df[attribute].unique():
            mask = df[attribute] == attribute_value
            partitioned_instances = len(df[mask])
            attribute_ratios.append(partitioned_instances / total_instances)
        return sum([-attribute_ratio * np.log2(attribute_ratio) for attribute_ratio in attribute_ratios])

    """ Private Field """
    __instance_threshold = 0
