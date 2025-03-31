# -*- coding: utf-8 -*-

import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from time import time
from collections import deque, defaultdict

from utils import *
from algorithm import *
from algorithm_distance import *
from scipy.spatial import KDTree

class Graph_weighted():
	def __init__(self, g, rescale, workers, untilLayer = None):

		logging.info(" - Converting graph to dict...")
		self.G = defaultdict(mylambda)
		self.G.update(g.gToDict())
		logging.info("Graph converted.")
		self.max_weight = g.maxWeight()
		self.min_weight = g.minWeight()

		self.num_vertices = g.number_of_nodes()
		self.num_edges = g.number_of_edges()
		self.rescale = rescale
		self.workers = workers
		self.calcUntilLayer = untilLayer
		logging.info('Graph - Number of vertices: {}'.format(self.num_vertices))
		logging.info('Graph - Number of edges: {}'.format(self.num_edges))
		logging.info('Graph - Max weight: {}'.format(self.max_weight))
		logging.info('Graph - Min weight: {}'.format(self.min_weight))
		logging.info('Graph - rescale function: {}'.format(self.rescale))

	def preprocess_neighbors_with_bfs(self):
		exec_bfs_weighted(self.G,self.workers,self.calcUntilLayer)
		# with ProcessPoolExecutor(max_workers=self.workers) as executor:
		# 	job = executor.submit(exec_bfs_weighted,self.G,self.workers,self.calcUntilLayer)

		# 	job.result()

		return

	def preprocess_neighbors_with_bfs_compact(self): # TODO
		# exec_bfs_compact_complex(self.G_p,self.G_n,self.workers,self.calcUntilLayer)
		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs_compact_complex,self.G_p,self.G_n,self.workers,self.calcUntilLayer)

			job.result()

		return

	def create_vectors(self):
		logging.info("Creating degree vectors...")
		degrees = {}
		degrees_sorted = set()
		G = self.G
		for v in G.keys():
			degree = len(G[v])
			degrees_sorted.add(degree)
			if(degree not in degrees):
				degrees[degree] = {}
				degrees[degree]['vertices'] = deque() 
			degrees[degree]['vertices'].append(v)
		degrees_sorted = np.array(list(degrees_sorted),dtype='int')
		degrees_sorted = np.sort(degrees_sorted)

		l = len(degrees_sorted)
		for index, degree in enumerate(degrees_sorted):
			if(index > 0):
				degrees[degree]['before'] = degrees_sorted[index - 1]
			if(index < (l - 1)):
				degrees[degree]['after'] = degrees_sorted[index + 1]
		logging.info("Degree vectors created.")
		logging.info("Saving degree vectors...")
		saveVariableOnDisk(degrees,'degrees_vector')

	def create_vectors_sdw(self):
		logging.info("Creating degree vectors...")
		# flatten
		degreeList = restoreVariableFromDisk('degreeList')
		data = []
		for i in sorted(degreeList.keys()):
			data.append(degreeList[i][0].flatten())
		array = np.array(data)
		result = {}
		sample_num = int(np.sqrt(array.shape[0]))
		for i in range(array.shape[0]):
			result[i] = set()

		kdtree = KDTree(array, leafsize=10)
		for i in range(array.shape[0]):
			_, indices = kdtree.query(array[i,:], k=sample_num)
			result[i] = result[i] | set(list(indices))
		
		for ix, nbs in result.items():
			for node in nbs:
				result[node] |= {ix}
    
		for i in range(array.shape[0]):
			result[i] = list(result[i] - set([i]))
   
		logging.info("Degree vectors created.")
		logging.info("Saving degree vectors...")
		saveVariableOnDisk(result,'degrees_vector')
  
	def calc_distances_all_vertices_presearch(self,bin_coeffient, compactDegree = False):

		logging.info("calc_distances_all_vertices_presearch Using compactDegree: {}".format(compactDegree))
		futures = {}
		vertices = list(reversed(sorted(list(self.G.keys()))))

		if(compactDegree):
			logging.info("Recovering compactDegreeList from disk...")
			degreeList = restoreVariableFromDisk('compactDegreeList')
		else:
			logging.info("Recovering degreeList from disk...")
			degreeList = restoreVariableFromDisk('degreeList')

		parts = self.workers
		chunks = partition(vertices,parts)

		t0 = time()
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part)+str(c))
				list_v = []
				for v in c:
					list_v.append([vd for vd in degreeList.keys() if vd > v])
				job = executor.submit(calc_distances_all_weighted_presearch, c, list_v, degreeList,part,0,bin_coeffient,  compactDegree = compactDegree)
				futures[job] = part
				part += 1
			logging.info("Receiving results...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} Completed.".format(r))

		logging.info('calc_distances_all_vertices_presearch Distances calculated.')
		t1 = time()
		logging.info('Time : {}m'.format((t1-t0)/60))

		best_k = calc_distances_all_weighted_presearch_find_best(parts)
		return best_k

	def calc_distances_all_vertices(self,bin_coeffient, compactDegree = False, best_k = 1):
		# maxA = np.log(np.sqrt(self.n_max_degree**2+self.p_max_degree**2)+1)
		# set_maxA(maxA)
		# logging.info("Using maxA: {}".format(maxA))
		logging.info("Using compactDegree: {}".format(compactDegree))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}

		count_calc = 0

		vertices = list(reversed(sorted(list(self.G.keys()))))

		if(compactDegree):
			logging.info("Recovering compactDegreeList from disk...")
			degreeList = restoreVariableFromDisk('compactDegreeList')
		else:
			logging.info("Recovering degreeList from disk...")
			degreeList = restoreVariableFromDisk('degreeList')

		parts = self.workers
		chunks = partition(vertices,parts)

		t0 = time()
#debug
		# part = 1
		# for c in chunks:
		# 	logging.info("Executing part {}...".format(part))
		# 	list_v = []
		# 	for v in c:
		# 		list_v.append([vd for vd in degreeList.keys() if vd > v])
		# 	calc_distances_all_weighted( c, list_v, degreeList,part,self.calcUntilLayer,bin_coeffient,  compactDegree = compactDegree)
   
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part)+str(c))
				list_v = []
				for v in c:
					list_v.append([vd for vd in degreeList.keys() if vd > v])
				job = executor.submit(calc_distances_all_weighted, c, list_v, degreeList,part,self.calcUntilLayer,bin_coeffient,  compactDegree = compactDegree, best_k = best_k)
				futures[job] = part
				part += 1


			logging.info("Receiving results...")

			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} Completed.".format(r))
# end debug
		logging.info('Distances calculated.')
		t1 = time()
		logging.info('Time : {}m'.format((t1-t0)/60))

		return


	def calc_distances(self, compactDegree = False):

		logging.info("Using compactDegree: {}".format(compactDegree))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}
		#distances = {}

		count_calc = 0

		G = self.G
		vertices = list(G.keys())

		parts = self.workers
		chunks = partition(vertices,parts)

		with ProcessPoolExecutor(max_workers = 1) as executor:

			logging.info("Split degree List...")
			part = 1
			for c in chunks:
				job = executor.submit(splitDegreeList,part,c,G,compactDegree)
				job.result()
				logging.info("degreeList {} completed.".format(part))
				part += 1


		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				job = executor.submit(calc_distances, part, compactDegree = compactDegree)
				futures[job] = part
				part += 1

			logging.info("Receiving results...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} completed.".format(r))


		return

	def calc_distances_sdw(self, bin_coeffient, compactDegree = False):
    
		logging.info("Using compactDegree: {}".format(compactDegree))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}
		#distances = {}

		count_calc = 0

		# G = self.G
		vertices = list(self.G.keys())

		parts = self.workers
		chunks = partition(vertices,parts)

		# part = 1
		# for c in chunks:
		# 	# print ('debug part = '+str(part)+ ' chunks'+str(chunks))
		# 	splitDegreeList_complex(part,c,self.G_p,self.G_n,compactDegree)
		# 	# print ('debug part = '+str(part))
		# 	part += 1
		with ProcessPoolExecutor(max_workers = 1) as executor:

			logging.info("Split degree List...")
			part = 1
			for c in chunks:
				job = executor.submit(splitDegreeList_sdw,part,c,self.G,compactDegree)
				job.result()
				logging.info("degreeList {} completed.".format(part))
				part += 1

		# part = 1
		
		# for c in chunks:
		# 	logging.info("Executing part {}...".format(part))
		# 	calc_distances_complex( part, compactDegree = compactDegree)
		# 	part += 1

		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				job = executor.submit(calc_distances_sdw, part, self.calcUntilLayer, bin_coeffient, compactDegree = compactDegree)
				futures[job] = part
				part += 1

			logging.info("Receiving results...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} completed.".format(r))


		return

	def consolide_distances(self):

		distances = {}

		parts = self.workers
		for part in range(1,parts + 1):
			d = restoreVariableFromDisk('distances-'+str(part))
			preprocess_consolides_distances(distances)
			distances.update(d)


		preprocess_consolides_distances(distances)
		saveVariableOnDisk(distances,'distances')


	def create_distances_network(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_distances_network,self.workers)

			job.result()

		return

	def preprocess_parameters_random_walk(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_parameters_random_walk,self.workers)

			job.result()

		return


	def simulate_walks(self,num_walks,walk_length,suffix):

		# for large graphs, it is serially executed, because of memory use.
		if(self.num_vertices > 500000):

			with ProcessPoolExecutor(max_workers=1) as executor:
				job = executor.submit(generate_random_walks_large_graphs,num_walks,walk_length,self.workers, list(reversed(sorted(list(self.G.keys())))), suffix)

				job.result()

		else:

			with ProcessPoolExecutor(max_workers=1) as executor:
				job = executor.submit(generate_random_walks,num_walks,walk_length,self.workers, list(reversed(sorted(list(self.G.keys())))), suffix)
				job.result()

		return	
