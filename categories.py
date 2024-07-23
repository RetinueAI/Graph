import json
from typing import Type

with open('categories.json', 'r') as f:
    categories = json.load(f)


def sub_cat(cat, n_i = 1):
    indentation = n_i * '\t'
    for sub in cat.keys():
        print(f'{indentation}{sub}')
        sub_cat(cat[sub], n_i=n_i+1)


for parent in categories.keys():
    print(parent)
    sub_cat(categories[parent])