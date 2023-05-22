import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules, fpmax
from decimal import Decimal
from statistics import mean, median
import numpy as np
import networkx as nx

all_data = pd.read_csv('groceries - groceries.csv')

np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem,str)] for row in np_data]
unique_items = set()
for row in np_data:
    for elem in row:
        unique_items.add(elem)

te = TransactionEncoder()
te_ary = te.fit(np_data).transform(np_data)
data = pd.DataFrame(te_ary, columns=te.columns_)
result = fpgrowth(data, min_support=0.03, use_colnames = True)
print(result)
result["itemsets_len"] = result["itemsets"].apply(lambda x: len(x))
length_itemsets = set(result["itemsets_len"])
for length in length_itemsets:
    length_result = result[(result['itemsets_len'] == length)]

result_fpmax = fpmax(data, min_support=0.03, use_colnames=True)
result_fpmax["itemsets_len"] = result_fpmax["itemsets"].apply(lambda x: len(x))
length_itemsets_fpmax = set(result["itemsets_len"])
for length in length_itemsets_fpmax:
    length_result_fpmax = result_fpmax[(result_fpmax["itemsets_len"] == length)]

result_fpmax_len1 = fpmax(data, min_support=0.03, use_colnames=True, max_len=1)
items = []
supports = []
for item in result_fpmax_len1["itemsets"]:
    items.append(*list(item))
for support in result_fpmax_len1["support"]:
    supports.append(support)
dictionary_most_common = {}
for i in range(len(items)):
    dictionary_most_common[items[i]] = supports[i]
tuple_most_common = tuple(sorted(dictionary_most_common.items(), key=lambda x: x[1], reverse=True))[0:10]

fig1, ax = plt.subplots(figsize=(18, 7))
names = [name[0] for name in tuple_most_common]
sups = [sup[1] for sup in tuple_most_common]
ax.bar(names, sups)
# plt.show()

items = ['whole milk', 'yogurt', 'soda', 'tropical fruit', 'shopping bags', 'sausage',
'whipped/sour cream', 'rolls/buns', 'other vegetables', 'root vegetables',
'pork', 'bottled water', 'pastry', 'citrus fruit', 'canned beer', 'bottled beer']
np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem,str) and elem in items] for row in np_data]

te1 = TransactionEncoder()
te_ary1 = te1.fit(np_data).transform(np_data)
data1 = pd.DataFrame(te_ary1, columns=te1.columns_)

result1 = fpgrowth(data, min_support=0.03, use_colnames = True)

result2 = fpmax(data1, min_support=0.03, use_colnames=True, max_len=1)
items = []
supports = []

for item in result2["itemsets"]:
    items.append(*list(item))
for support in result2["support"]:
    supports.append(support)
dictionary_most_common = {}
for i in range(len(items)):
    dictionary_most_common[items[i]] = supports[i]
tuple_most_common = tuple(sorted(dictionary_most_common.items(), key=lambda x: x[1], reverse=True))[0:10]

fig2, ax = plt.subplots(figsize=(12, 3))
names = [name[0] for name in tuple_most_common]
sups = [sup[1] for sup in tuple_most_common]
ax.bar(names, sups)
# plt.show()

start = Decimal(0.05)
support = []
length = []
max_len = []
while start <= 1:
    support.append(start)
    results = fpgrowth(data1, min_support=start, use_colnames=True)
    results['length'] = results['itemsets'].apply(lambda x: len(x))
    length.append(len(results))
    if len(results) > 0:
        max_len.append(max(results['length']))
    start += Decimal(0.01)

fig3, ax = plt.subplots(figsize=(6, 4))
ax.plot(support, length)
ax.scatter(support[max_len.index(1)], length[max_len.index(1)], color='green', s=40, marker='o')
ax.scatter(support[len(max_len)], length[len(max_len)], color='yellow', s=40, marker='o')
# plt.show()

np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem, str) and elem in items] for row in np_data]
np_data = [row for row in np_data if len(row) > 1]

result = fpgrowth(data, min_support=0.05, use_colnames=True)

rules = association_rules(result, min_threshold=0.3)

result = fpgrowth(data, min_support=0.04, use_colnames=True)

rules = association_rules(result, min_threshold=0.1, metric='confidence')

print(f'Среднее значение параметра confidence: {mean(rules["confidence"])}')
print(f'Медиана параметра confidence: {median(rules["confidence"])}')
print(f'СКО параметра confidence: {np.std(rules["confidence"])}')

rules = association_rules(result, min_threshold=0.01, metric='support')
print(f'Среднее значение параметра support: {mean(rules["support"])}')
print(f'Медиана параметра support: {median(rules["support"])}')
print(f'СКО параметра support: {np.std(rules["support"])}')

rules = association_rules(result, min_threshold=0.01, metric='leverage')
print(f'Среднее значение параметра leverage: {mean(rules["leverage"])}')
print(f'Медиана параметра leverage: {median(rules["leverage"])}')
print(f'СКО параметра leverage: {np.std(rules["leverage"])}')

rules = association_rules(result, min_threshold=0.01, metric='conviction')
print(f'Среднее значение параметра conviction: {mean(rules["conviction"])}')
print(f'Медиана параметра conviction: {median(rules["conviction"])}')
print(f'СКО параметра conviction: {np.std(rules["conviction"])}')

rules = association_rules(result, min_threshold=0.4, metric='confidence')

antecedents = rules["antecedents"]
consequents = rules["consequents"]
confidence = rules["confidence"]
supports = rules["support"]
edges = []
dict_labels = {}

for i in range(len(antecedents)):
    edges.append([str(list(antecedents[i])[0]), str(list(consequents[i])[0])])
    dict_labels[(str(list(antecedents[i])[0]), str(list(consequents[i])[0]))] = round(confidence[i], 2)

G = nx.DiGraph()
for i in edges:
    G.add_nodes_from(i)
    G.add_edges_from([tuple(i)])
pos = nx.spring_layout(G)
plt.figure()
for i in range(len(dict_labels.items())):
    nx.draw(
        G, pos, edge_color='black', width=round(supports[i], 2), linewidths=1,
        node_size=3000, node_color='pink', alpha=0.9, font_size=10,
        labels={node: node for node in G.nodes()})

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=dict_labels,
    font_color='red'
)
plt.axis('off')
plt.show()