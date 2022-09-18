from binary_decision_tree import BinaryDecisionTree
# from decision_tree import CategoricalDecisionTree
import pandas as pd
from sklearn.preprocessing import LabelEncoder

clmns = ['Sexo', 'Idade < 26', 'Tem carro', 'É assinante?']

data = pd.DataFrame(
    [
     ('M', True, False, 'Sim'),
     ('M', True, True, 'Sim'),
     ('F', True, True, 'Não'),
     ('M', False, True, 'Não'),
     ('F', True, False, 'Não'),
     ('M', False, False, 'Não'),
     ('F', False, True, 'Não')
    ],
    columns=clmns
)

x=data[clmns[:-1]]
y=data[clmns[-1]]
lencoder = LabelEncoder()
# y = lencoder.fit(y)

tree = BinaryDecisionTree()

tree.fit(x=x, y=y)

case = (
    'M', True, False
)

case = (
    'F', True, True
)

result = tree.predict(
    {
        clmns[0]: case[0],
        clmns[1]: case[1],
        clmns[2]: case[2]
    }
)

print(result)

tree.print()

print("end")