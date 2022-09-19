from io import BytesIO
from binary_decision_tree import BinaryDecisionTree
# from decision_tree import CategoricalDecisionTree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image

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
tree.plot()

image = Image.open(BytesIO(tree.plot().pipe(format='png')))
image.show()

clmns = ['A', 'B', 'C', 'X']
data = pd.DataFrame(
    [
        (1, 1,  5.0, 'P'),
        (0, 1, 11.5, 'P'),
        (0, 1, 13.5, 'P'),
        (1, 0, 15.0, 'P'),
        (1, 0, 13.0, 'P'),
        (0, 0, 11.5, 'N'),
        (0, 1,  5.0, 'N'),
        (1, 0,  9.5, 'N'),
        (0, 0,  2.0, 'N'),
        (1, 0,  8.0, 'N')
    ], columns=clmns
)

x = data[clmns[:-1]]
y = data[clmns[-1]]

tree = BinaryDecisionTree()
tree.fit(x, y)
tree.print()
tree.plot()

image = Image.open(BytesIO(tree.plot().pipe(format='png')))
image.show()

print("end")