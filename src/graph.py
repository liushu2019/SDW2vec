# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
from io import open
from time import time
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Counter
from multiprocessing import cpu_count
from itertools import permutations, chain
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterable
import copy
import math


def calculate_entropy(data):
  # Count occurrences of each unique element
  counts = Counter(data)
  total_count = sum(counts.values())
  
  # Calculate probabilities
  probabilities = [count / total_count for count in counts.values()]
  
  # Calculate entropy
  entropy = -sum(p * math.log2(p) for p in probabilities)
  return entropy

def get_entropy(f, w, b):
  mapping = f(w,b)[0]
  return calculate_entropy([mapping[x] for x in w])

def normalize_data(x, y):
  """Normalize the data to 0-1 range"""
  x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
  y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
  return x_norm, y_norm

def find_elbow_point(x, y):
  # print (x,y)
  # Normalize the data
  x_norm, y_norm = normalize_data(x, y)
  y_dense = y_norm - x_norm
  # # Find point of maximum curvature
  elbow_idx = np.argmax(y_dense)
  elbow_x = x[elbow_idx]
  elbow_y = y[elbow_idx]
  return elbow_x, elbow_y

def rescale_cut(input_list, num_quantiles):
  bins = np.linspace(min(input_list), max(input_list), num_quantiles+1) 
  return dict(zip(np.unique(input_list), np.searchsorted(bins, np.unique(input_list)) )), list(range(sum(bins<0), 0, -1)) + list(range(1, sum(bins>=0) + 1, 1))

def rescale_qcut(input_list, num_quantiles):
  quantiles = np.percentile(input_list, np.linspace(0, 100, num_quantiles+1))
  return dict(zip(np.unique(input_list), np.digitize(np.unique(input_list), np.unique(quantiles), right=True))), list(range(sum(quantiles<0), 0, -1)) + list(range(1, sum(quantiles>=0) + 1, 1))

def rescale_log_cut(input_list, num_quantiles, logfunc):
  bins = np.linspace(logfunc(min(input_list)), logfunc(max(input_list)), num_quantiles+1) 
  return dict(zip([logfunc(x) for x in np.unique(input_list)], np.searchsorted(bins, [logfunc(x) for x in np.unique(input_list)]) )), list(range(sum(bins<0), 0, -1)) + list(range(1, sum(bins>=0) + 1, 1))

def get_hist_list(input_list, max_bin):
  result_list = [0] * (max_bin + 1)
  # Fill the list with values from the dictionary
  for key, value in dict(Counter(input_list)).items():
    result_list[key] = value
  return result_list

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(dict)

  def nodes(self):
    return list(self.keys())

  def adjacency_iter(self):
    return self.items()

  def subgraph(self, nodes={}):
    subgraph = Graph()

    for n in nodes:
      if n in self:
        subgraph[n] = {x:self[n][x] for x in self[n] if x in nodes}

    return subgraph

  def make_undirected(self):
    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].update({v:self[v][other]})
    self.make_consistent()
    return self

  def make_consistent(self):
    for k in iterkeys(self):
      self[k] = dict(sorted(self[k].items()))
    return self

  def remove_self_loops(self):
    removed = 0
    for x in self:
      if x in self[x]: 
        del self[x][x]
        removed += 1
    logging.info('remove_self_loops: removed {} loops'.format(removed))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def maxDegree(self):
    return max([len(self[v]) for v in list(self.keys())])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order() 

  def gToDict(self):
    d = {}
    for k,v in self.items():
      d[k] = v
    return d

  def printAdjList(self):
    for key,value in self.items():
      print (key,":",value)

  def maxWeight(self):
    "Returns the maximum of weights"
    # print ([x[0] for x in self.items()])
    return max([w for v in self.values() for w in [x[1] for x in v.items() if x[0] != 'degree_hist']])

  def minWeight(self):
    "Returns the minimum of weights"
    return min([w for v in self.values() for w in [x[1] for x in v.items() if x[0] != 'degree_hist']])

  def rescaleWeights(self, rescalefunction, num_quantiles):
    "Rescale the weights by the given funcion. CALL ONLY ONCE!"
    list_weights = [w for v in self.values() for w in v.values()]
    if len(Counter(list_weights)) == 1:
      for v in self.keys():
        self[v]['degree_hist'] = [len(self[v])]
      return [1]
    if num_quantiles <= 0: # search for best number of bins
      if rescalefunction == 'auto':
        list_func = [rescale_cut, rescale_qcut]
      elif rescalefunction == 'cut':
        list_func = [rescale_cut]
      elif rescalefunction == 'qcut':
        list_func = [rescale_qcut]
      
      rst_all = {}
      for f in list_func:
        rst_all[f.__name__] = []
        for num in range(1, len(Counter(list_weights))):
          rst_all[f.__name__].append(get_entropy(f, list_weights, num))
      best_score = 0
      for f in list_func:
        y = rst_all[f.__name__]
        x = list(range(1,len(Counter(list_weights))))
        elbow_x, elbow_y = find_elbow_point(x, y)
        if best_score < elbow_y:
          best_score = elbow_y
          best_f = f
          best_num = elbow_x
        logging.info(f"Best func and num_bin found at: x={best_num:.2f}, y={best_score:.2f}, f={best_f.__name__}")
      mapping, bin_coefficient = best_f(list_weights, best_num)
      num_quantiles = best_num
    elif num_quantiles > 0:
      if num_quantiles > len(Counter(list_weights)):
        num_quantiles = len(Counter(list_weights))
      if rescalefunction == 'auto':
        list_func = [rescale_cut, rescale_qcut]
      elif rescalefunction == 'cut':
        list_func = [rescale_cut]
      elif rescalefunction == 'qcut':
        list_func = [rescale_qcut]
      
      rst_all = {}
      for f in list_func:
        rst_all[f.__name__] = get_entropy(f, list_weights, num_quantiles)
      best_score = 0
      for f in list_func:
        if best_score < rst_all[f.__name__]:
          best_f = f
          best_score = rst_all[f.__name__]
        logging.info(f"Best func found at: x={num_quantiles}, y={best_score}, f={best_f.__name__}")
      mapping, bin_coefficient = best_f(list_weights, num_quantiles)
      
    for v in self.keys():
      for u in self[v].keys():
        self[v][u] = mapping[self[v][u]]
      self[v]['degree_hist'] = get_hist_list(self[v].values(), num_quantiles)
    return bin_coefficient

# def clique(size):
#     return from_adjlist(permutations(range(1,size+1)))

# # http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
# def grouper(n, iterable, padvalue=None):
#     "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
#     return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

# def parse_adjacencylist(f):
#   adjlist = []
#   for l in f:
#     if l and l[0] != "#":
#       introw = [int(x) for x in l.strip().split()]
#       row = [introw[0]]
#       row.extend(set(sorted(introw[1:])))
#       adjlist.extend([row])

#   return adjlist

# def parse_adjacencylist_unchecked(f):
#   adjlist = []
#   for l in f:
#     if l and l[0] != "#":
#       adjlist.extend([[int(x) for x in l.strip().split()]])
#   return adjlist

# def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

#   if unchecked:
#     parse_func = parse_adjacencylist_unchecked
#     convert_func = from_adjlist_unchecked
#   else:
#     parse_func = parse_adjacencylist
#     convert_func = from_adjlist

#   adjlist = []

#   t0 = time()

#   with open(file_) as f:
#     with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
#       total = 0 
#       for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
#           adjlist.extend(adj_chunk)
#           total += len(adj_chunk)

#   t1 = time()

#   logging.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

#   t0 = time()
#   G = convert_func(adjlist)
#   t1 = time()

#   logging.info('Converted edges to graph in {}s'.format(t1-t0))

#   if undirected:
#     t0 = time()
#     G = G.make_undirected()
#     t1 = time()
#     logging.info('Made graph undirected in {}s'.format(t1-t0))

#   return G 


# def load_edgelist(file_, undirected=True):
#   G = Graph()
#   with open(file_) as f:
#     for line in f:
# #       import pdb; pdb.set_trace()
#       if(len(line.strip().split()[:2]) > 1):
#         x, y = line.strip().split()[:2]
#         x = int(x)
#         y = int(y)
#         G[x].append(y)
#         if undirected:
#           G[y].append(x)
#       else:
#         x = line.strip().split()[:2]
#         x = int(x[0])
#         G[x] = []  

#   G.make_consistent()
#   return G

def load_edgelist_weighted(file_, rescale, num_bins, directed=False, signed=False):
  G = [Graph()]
  if directed:
    G = G + [Graph() for x in G]
  if signed:
    G = G + [Graph() for x in G]
    
  with open(file_) as f:
    for line in f:
      if(len(line.strip().split()[:3]) > 2):
        x, y, s = line.strip().split()[:3]
        x = int(x)
        y = int(y)
        s = float(s)
        if signed and directed:
          if s > 0:
            G[0][x].update({y:s})
            G[2][y].update({x:s})
          elif s < 0:
            G[1][x].update({y:-s})
            G[3][y].update({x:-s})
        elif signed and not directed:
          if s > 0:
            G[0][x].update({y:s})
            G[0][y].update({x:s})
          elif s < 0:
            G[1][x].update({y:-s})
            G[1][y].update({x:-s})
        elif not signed and directed:
            G[0][x].update({y:s})
            G[1][y].update({x:s})
        else:
          G[0][x].update({y:s})
          G[0][y].update({x:s})
          
      elif (len(line.strip().split()[:3]) == 2):
        x, y = line.strip().split()[:2]
        x = int(x)
        y = int(y)
        print (f'[Input file] weight undefined, replaced by 1. Edge={x},{y}')
        G[0][x].update({y:1.0})
        if not directed:
          G[0][y].update({x:1.0})
        else:
          if signed:
            G[2][y].update({x:1.0})
          else:
            G[1][y].update({x:1.0})
      else:
        print ('Input file error: missed target node, line ignored.')
  G_remove_null = []
  for subg in G:
    if (len(subg)) > 0:
      G_remove_null.append(subg)
  G = G_remove_null
  
  node_list = list(set().union(*[g.nodes() for g in G]))
  G_all = Graph()
  # print (G)
  for node in node_list:
    for g in G:
      G_all[node].update(g[node])
  bin_coefficient = []
  
  for subg in G:
    subg.make_consistent()
    bin_coefficient += subg.rescaleWeights(rescale, num_bins)
  for node in node_list:
    G_all[node]['degree_hist'] = list(chain(*[g[node]['degree_hist'] for g in G]))
    
  # print (G_all)
  # print (bin_coefficient)
  return G_all, bin_coefficient

# def load_matfile(file_, variable_name="network", undirected=True):
#   mat_varables = loadmat(file_)
#   mat_matrix = mat_varables[variable_name]

#   return from_numpy(mat_matrix, undirected)


# def from_networkx(G_input, undirected=True):
#     G = Graph()

#     for idx, x in enumerate(G_input.nodes_iter()):
#         for y in iterkeys(G_input[x]):
#             G[x].append(y)

#     if undirected:
#         G.make_undirected()

#     return G


# def from_numpy(x, undirected=True):
#     G = Graph()

#     if issparse(x):
#         cx = x.tocoo()
#         for i,j,v in zip(cx.row, cx.col, cx.data):
#             G[i].append(j)
#     else:
#       raise Exception("Dense matrices not yet supported.")

#     if undirected:
#         G.make_undirected()

#     G.make_consistent()
#     return G


# def from_adjlist(adjlist):
#     G = Graph()

#     for row in adjlist:
#         node = row[0]
#         neighbors = row[1:]
#         G[node] = list(sorted(set(neighbors)))

#     return G


# def from_adjlist_unchecked(adjlist):
#     G = Graph()

#     for row in adjlist:
#         node = row[0]
#         neighbors = row[1:]
#         G[node] = neighbors

#     return G


# def from_dict(d):
#     G = Graph()
#     for k,v in d.items():
#       G[k] = v

#     return G
