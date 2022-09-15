from functools import reduce
import math
import operator


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

class DecisionTreeNode():
  def __init__(self, feature=None, terminal=False, label = None):
    self.terminal = terminal
    self.children = dict()
    
    if terminal:
      self.label = label
      return
    
    self.feature = feature


  def is_leaf(self):
    return len(self.children) == 0


  def setChildren(self, children):
    self.children = children


  def visit(self, point):
    if self.terminal:
      return self.label
    
    return self.children[point[self.feature]].visit(point)


class DecisionTree():
  def __init__(self):
    self.root = None
    self.currentNode = None

  def fit(self, x=None, y=None):
    # if x==None:
    #   raise ValueError("Argument 'x' cannot be None")
    
    # if y==None:
    #   raise ValueError("Argument 'y' cannot be None")
    
    values = dict()
    unused_features = []

    for feature in x:
      values[feature] = x[feature].unique()
      unused_features.append(feature)

    done = False

    while not done:
      if len(unused_features) == 0:
        done = True
        break
      
      best_feature = max(
        map(
          lambda feature: (feature, information_gain(x.join(y), feature, y.name))
          , unused_features
        ), axis=1
      )

      unused_features.remove(best_feature[0])

      if self.root == None:
        if len(unused_features) == 0:
          self.root = DecisionTreeNode(terminal=True)
        else:
          self.root = DecisionTreeNode(best_feature[0])
      else:
        self.currentNode



