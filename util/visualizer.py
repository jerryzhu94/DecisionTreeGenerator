from graphviz import Digraph


class Visualizer:
    """ Public Method """

    # Creates the dot graph
    @staticmethod
    def draw_decision_tree_dictionary(tree_dictionary, features_list):
        if not isinstance(tree_dictionary, dict):
            raise TypeError("Argument tree_dictionary must be of type dictionary")
        if not tree_dictionary:
            raise ValueError("Dictionary tree_dictionary is empty")
        dot = Digraph(strict=True)
        Visualizer.__draw_tree(dot, features_list, tree_dictionary, None)
        return dot

    """ Private Helper Method """

    # Constructs dot graph by recursively creating nodes
    @staticmethod
    def __draw_tree(dot, features_list, tree_dictionary, parent_node_name):
        if isinstance(tree_dictionary, dict):
            for key in tree_dictionary:
                no_spaces_key = str(key).replace(" ", "")
                dot.node(no_spaces_key, str(key), shape="ellipse")
                if parent_node_name is not None:
                    dot.edge(parent_node_name, no_spaces_key)
                Visualizer.__draw_tree(dot, features_list, tree_dictionary[key], no_spaces_key)
        else:
            val = str(tree_dictionary)
            dot.node(val, val, shape="plaintext")
            dot.edge(parent_node_name, val)
