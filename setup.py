
'''
#       Install Project Requirements 
'''
from setuptools import setup

setup(name='decisionTree',
    url="https://github.com/Icaro-Nunes/decisionTree",
    description="Árvore de decisão com boa diferenciação entre variáveis categóricas e numéricas",
    packages=['decision_tree'],
    install_requires=['pandas', 'graphviz']
)