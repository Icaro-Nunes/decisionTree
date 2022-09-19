from functools import reduce
from io import BytesIO
import math
import operator
import pandas as pd
import graphviz
import numpy
from PIL import Image

def entropy(df, param):
  return (
      reduce(
          operator.add,
          map(lambda x: (-1)*(x/len(df))*math.log2(x/len(df)), df[param].value_counts())
      )
  )

def conditional_entropy(df: pd.DataFrame, known_param, param):
  return reduce(
    operator.add,
    map(
      lambda val: (
          len(
            df[df[known_param] == val]
          )
          *
          entropy(
            df[df[known_param] == val],
            param
          )
          /
          len(df)
      ), df[known_param].unique()
    )
  )


    

def information_gain(df, known_param, class_axis):
  return (None, entropy(df, class_axis) - conditional_entropy(df, known_param, class_axis))

def calculate_distribution(x, y, value):
    set = x.join(y)
    return float(len(set[set[y.name] == value]))/len(set)


class DecisionTreeNode():
    def __init__(self, feature , children: dict):
        self.feature = feature
        self.children = children
        self.string_width = None

    def visit(self, x):
        return self.children[
            x[self.feature]
        ].visit(x)

    def print(self, value, depth=0):
        print(f"{'    '*depth}({value}){self.feature}")
        for val, child in self.children.items():
            child.print(val, depth+1)
        
    def plot(self, graph: graphviz.Graph, index=0):
        graph.node(str(index), self.feature)
        last_index = index
        child_index = last_index
        for value, child in self.children.items():
            child_index = last_index + 1
            last_index = child.plot(graph, child_index)
            graph.edge(str(index), str(child_index), str(value))
        return last_index
        

class NumericalDecisionTreeNode(DecisionTreeNode):
    def __init__(self, feature, children: dict, threshold):
        self.threshold = threshold
        super().__init__(feature, children)
    
    def visit(self, x):
        if x[self.feature] > self.threshold:
            return self.children[f"> {self.threshold}"].visit(x)
        return self.children[f"<= {self.threshold}"]

class DecisionTreeTerminalNode():
    def __init__(self, label):
        self.label = label
        self.string_width = None
    
    def visit(self, x):
        return self.label

    def print(self, value, depth=0):
        print(f"{'    '*depth}({value}){self.label}")
    
    def plot(self, graph: graphviz.Graph, index=0):
        graph.node(str(index), self.label)
        return index


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

class BinaryDecisionTree():
    def __init__(self, threshold_step=0.1):
        self.numerical_types = [int, float, numpy.int64, numpy.int32, numpy.float64, numpy.float32, numpy.float16, numpy.double]
        self.NON_CATEGORICAL_PERCENT = 0.7
        self.threshold_step = threshold_step
        self.root = None
        self.string_tree = None

    def make_threshold_sequence(self, min, max):
        return drange(min, max, self.threshold_step)


    def numerical_information_gain(self, df: pd.DataFrame, known_param, class_axis):
        column = df[known_param]
        min = column.min()
        max = column.max()
        best_information_gain = (0.0, 0.0)
        for threshold in self.make_threshold_sequence(min, max):
            case_df = df.drop(columns=[known_param])
            case_df[known_param] = column.transform(lambda it: it > threshold)
            info_gain = information_gain(case_df, known_param, class_axis)[1]
            if info_gain > best_information_gain[1]:
                best_information_gain = (threshold, info_gain)
        return best_information_gain


    def __build_tree_node(self, x_subset, y_subset, unused_features: list):
        subset = x_subset.join(y_subset)
        distribution = calculate_distribution(x_subset, y_subset, self.labels[0])

        estimated_label = self.labels[0] if distribution > 0.5 else self.labels[1]

        if len(unused_features) == 0:
            return DecisionTreeTerminalNode(
                label=estimated_label
            )

        if distribution == 1.0 or distribution == 0.0:
            return DecisionTreeTerminalNode(
                label=estimated_label
            )

        best_feature = max(
            map(
            lambda feature: (feature['name'], feature['information_gain'](subset, feature['name'], y_subset.name))
            , unused_features
            ), key=lambda it: it[1][1]
        )

        feature = best_feature[0]
        threshold = best_feature[1][0]
        info_gain = best_feature[1][1]

        feature_values = x_subset[feature].unique()
        
        unused_features_passed = [feat for feat in unused_features if feat['name'] != feature]

        if threshold != None:
            children = {
                f"> {threshold}": self.__build_tree_node(
                    x_subset[x_subset[feature] > threshold], subset[subset[feature] > threshold][y_subset.name], unused_features_passed
                ),
                f"<= {threshold}": self.__build_tree_node(
                    x_subset[x_subset[feature] <= threshold], subset[subset[feature] <= threshold][y_subset.name], unused_features_passed
                )
            }
            return NumericalDecisionTreeNode(
                feature=feature,
                children=children,
                threshold=threshold
            )

        children = {
            value: self.__build_tree_node(
                x_subset[x_subset[feature] == value], subset[subset[feature] == value][y_subset.name], unused_features_passed
            ) for value in feature_values
        }

        return DecisionTreeNode(
            feature=feature,
            children=children
        )
    
    def evaluate_categorical_and_numerical_information_gain(self, x: pd.DataFrame):
        features = []
        for feature in x:
            feat_info = {'name': feature}
            if x[feature].dtype.type in self.numerical_types and float(len(x[feature].unique()))/len(x[feature]) >= self.NON_CATEGORICAL_PERCENT:
                feat_info['information_gain'] = self.numerical_information_gain
            else:
                feat_info['information_gain'] = information_gain
            features.append(feat_info)
        return features


    def fit(self, x: pd.DataFrame, y, labels=None):
        if labels == None:
            if type(y) == pd.Series:
                self.labels = list(y.unique())
            else:
                self.labels = reduce(operator.eq, y)
        else:
            self.labels = labels
        
        features = self.evaluate_categorical_and_numerical_information_gain(x)

        self.root = self.__build_tree_node(x, y, features)
        
    
    def predict(self, x):
        return self.root.visit(x)

    def print(self):
        self.root.print('root')
    
    def plot(self):
        graph = graphviz.Graph()
        self.root.plot(graph)
        return graph

