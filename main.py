from decision_tree import DecisionTree
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
y = lencoder.fit_transform(y)

tree = DecisionTree()

tree.fit(x=data[clmns[:-1]], y=data[clmns[-1]])