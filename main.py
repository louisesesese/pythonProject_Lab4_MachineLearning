import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules, fpmax
from decimal import Decimal
from statistics import mean, median
import numpy as np
import networkx as nx

groceries = pd.read_csv('groceries - groceries.csv')

groceries_np = groceries.to_numpy()
groceries_np = [[elem for elem in row[1:] if isinstance(elem,str)] for row in groceries_np]

unique_items = set()
for row in groceries_np:
    for elem in row:
        unique_items.add(elem)

te = TransactionEncoder()
te_ary = te.fit(groceries_np).transform(groceries_np)
groceries_data = pd.DataFrame(te_ary, columns=te.columns_)

fpg_result = fpgrowth(groceries_data, min_support=0.03, use_colnames = True)
fpg_result["itemsets_len"] = fpg_result["itemsets"].apply(lambda x: len(x))
length_itemsets = set(fpg_result["itemsets_len"])
for length in length_itemsets:
    length_result = fpg_result[(fpg_result['itemsets_len'] == length)]

fpmax_result = fpmax(groceries_data, min_support=0.03, use_colnames=True)
fpmax_result["itemsets_len"] = fpmax_result["itemsets"].apply(lambda x: len(x))
length_itemsets_fpmax = set(fpmax_result["itemsets_len"])
for length in length_itemsets_fpmax:
    length_result_fpmax = fpmax_result[(fpmax_result["itemsets_len"] == length)]

fpmax_result_len1 = fpmax(groceries_data, min_support=0.03, use_colnames=True, max_len=1)
items = []
supports = []
for item in fpmax_result_len1["itemsets"]:
    items.append(*list(item))
for support in fpmax_result_len1["support"]:
    supports.append(support)
dictionary_most_common = {}
for i in range(len(items)):
    dictionary_most_common[items[i]] = supports[i]
tuple_most_common = tuple(sorted(dictionary_most_common.items(), key=lambda x: x[1], reverse=True))[0:10]

fig1, ax = plt.subplots(figsize=(18, 7))
names = [name[0] for name in tuple_most_common]
sups = [sup[1] for sup in tuple_most_common]
ax.bar(names, sups)

items = ['whole milk', 'yogurt', 'soda', 'tropical fruit', 'shopping bags', 'sausage',
'whipped/sour cream', 'rolls/buns', 'other vegetables', 'root vegetables',
'pork', 'bottled water', 'pastry', 'citrus fruit', 'canned beer', 'bottled beer']
groceries_np = groceries.to_numpy()
groceries_np = [[elem for elem in row[1:] if isinstance(elem,str) and elem in items] for row in groceries_np]

te1 = TransactionEncoder()
te_ary1 = te1.fit(groceries_np).transform(groceries_np)
groceries_data1 = pd.DataFrame(te_ary1, columns=te1.columns_)

fpg_result1 = fpgrowth(groceries_data1, min_support=0.03, use_colnames=True)

fpmax_result2 = fpmax(groceries_data1, min_support=0.03, use_colnames=True, max_len=1)
items = []
supports = []

for item in fpmax_result2["itemsets"]:
    items.append(*list(item))
for support in fpmax_result2["support"]:
    supports.append(support)
dictionary_most_common = {}
for i in range(len(items)):
    dictionary_most_common[items[i]] = supports[i]
tuple_most_common = tuple(sorted(dictionary_most_common.items(), key=lambda x: x[1], reverse=True))[0:10]
fig2, ax = plt.subplots(figsize=(12, 3))
names = [name[0] for name in tuple_most_common]
sups = [sup[1] for sup in tuple_most_common]
ax.bar(names, sups)

start = Decimal(0.05)
support = []
length = []
max_len = []
while start <= 1:
    support.append(start)
    results = fpgrowth(groceries_data1, min_support=start, use_colnames=True)
    results['length'] = results['itemsets'].apply(lambda x: len(x))
    length.append(len(results))
    if len(results) > 0:
        max_len.append(max(results['length']))
    start += Decimal(0.01)
fig3, ax = plt.subplots(figsize=(6, 4))
ax.plot(support, length)
ax.scatter(support[max_len.index(1)], length[max_len.index(1)], color='green', s=40, marker='o')
ax.scatter(support[len(max_len)], length[len(max_len)], color='yellow', s=40, marker='o')

groceries_np = groceries.to_numpy()
groceries_np = [[elem for elem in row[1:] if isinstance(elem, str) and elem in items] for row in groceries_np]
groceries_np = [row for row in groceries_np if len(row) > 1]

fpg_result = fpgrowth(groceries_data, min_support=0.05, use_colnames=True)

result = fpgrowth(fpg_result, min_support=0.04, use_colnames=True)
rules_conf = association_rules(result, min_threshold=0.1, metric='confidence')
mean_conf = mean(rules_conf["confidence"])
median_conf = median(rules_conf["confidence"])
std_conf = np.std(rules_conf["confidence"])

rules_support = association_rules(result, min_threshold=0.01, metric='support')
mean_support = mean(rules_support["support"])
median_support = median(rules_support["support"])
std_support = np.std(rules_support["support"])

rules_leverage = association_rules(result, min_threshold=0.01, metric='leverage')
mean_leverage = mean(rules_leverage["leverage"])
median_leverage = median(rules_leverage["leverage"])
std_leverage = np.std(rules_leverage["leverage"])

rules_conviction = association_rules(result, min_threshold=0.01, metric='conviction')
mean_conviction = mean(rules_conviction["conviction"])
median_conviction = median(rules_conviction["conviction"])
std_conviction = np.std(rules_conviction["conviction"])

rules_graph = association_rules(result, min_threshold=0.4, metric='confidence')
antecedents = rules_graph["antecedents"]
consequents = rules_graph["consequents"]
confidence = rules_graph["confidence"]
supports = rules_graph["support"]
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
    font_color='red')
plt.axis('off')
plt.show()

print(f'Среднее значение confidence: {mean_conf}')
print(f'Медиана confidence: {median_conf}')
print(f'СКО confidence: {std_conf}')
print(f'Среднее значение support: {mean_support}')
print(f'Медиана support: {median_support}')
print(f'СКО support: {std_support}')
print(f'Среднее значение leverage: {mean_leverage}')
print(f'Медиана leverage: {median_leverage}')
print(f'СКО leverage: {std_leverage}')
print(f'Среднее значение conviction: {mean_conviction}')
print(f'Медиана conviction: {median_conviction}')
print(f'СКО conviction: {std_conviction}')