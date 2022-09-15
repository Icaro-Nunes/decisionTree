from functools import reduce
import math
import operator
from unittest import result
import pandas as pd

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
  return entropy(df, class_axis) - conditional_entropy(df, known_param, class_axis)

def calculate_distribution(x, y, value):
    set = x.join(y)
    return float(len(set[set[y.name] == value]))/len(set)

class DecisionTreeNode():
    def __init__(self, feature , children):
        self.feature = feature
        self.children = children

    def visit(self, x):
        return self.children[
            x[self.feature]
        ].visit(x)

class DecisionTreeTerminalNode():
    def __init__(self, label):
        self.label = label
    
    def visit(self, x):
        return self.label

class BinaryDecisionTree():
    def __init__(self):
        self.root = None
        # self.current_node = None


    def __build_tree_node(self, x_subset, y_subset, unused_features: list):
        subset = x_subset.join(y_subset)
        distribution = calculate_distribution(x_subset, y_subset, 'Sim')

        estimated_label = 'Sim' if distribution > 0.5 else 'Não'

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
            lambda feature: (feature, information_gain(subset, feature, y_subset.name))
            , unused_features
            )
        )

        feature = best_feature[0]

        feature_values = x_subset[feature].unique()
        
        unused_features_passed = [feat for feat in unused_features if feat != feature]

        children = {
            value: self.__build_tree_node(
                x_subset[x_subset[feature] == value], subset[subset[feature] == value][y_subset.name], unused_features_passed
            ) for value in feature_values
        }

        return DecisionTreeNode(
            feature=feature,
            children=children
        )
    
    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.root = self.__build_tree_node(x, y, list(x))
        
    
    def predict(self, x):
        return self.root.visit(x)
