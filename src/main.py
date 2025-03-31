# -*- coding: utf-8 -*-

import argparse, logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time

import inspect
import os.path
import sdw2vec
import graph
import config
# logging.basicConfig(filename='weighteds2v.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')

def parse_args():
	'''
	Parses the sdw2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run sdw2vec.")

	parser.add_argument('--input', nargs='?', default='star_weighted.edgelist',
	                    help='Input graph path')
	parser.add_argument('--output', nargs='?', default=None,
	                    help='Output emb path, if Not given, follow input file name')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')
	parser.add_argument('--signed', action='store_true',
                      help='Flag for signed network.')
	parser.add_argument('--directed', action='store_true',
                      help='Flag for directed network.')
	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=20,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--until-layer', type=int, default=6,
                    	help='Calculation until the layer.')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--OPT1', action='store_true',
                      help='optimization 1')
	parser.add_argument('--OPT2', action='store_true',
                      help='optimization 2')
	parser.add_argument('--OPT3', action='store_true',
                      help='optimization 3')
	
	parser.add_argument('--rescale', type=str, default='auto',
					 help='Type of weight rescaling function from auto, cut, qcut')
	parser.add_argument('--num-bins', type=int, default=0,
					 help='Number of bins to rescale weights. Apply best search for 0.')
	
	parser.add_argument('--suffix', nargs='?', default='TEST',
	                    help='log file and pickles folder suffix')
	
	return parser.parse_args()

def read_graph_weighted():
	'''
	Reads the input weighted network.
	'''
	logging.info(" - Loading weighted graph...")
	G, bin_coeffient = graph.load_edgelist_weighted(args.input,args.rescale,args.num_bins,directed=args.directed, signed=args.signed)
	logging.info(" - Weighted Graph loaded.")
	return G, bin_coeffient

def learn_embeddings(basename, suffix=None):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	logging.info("Initializing creation of the representations...")
	walks = LineSentence('{}/random_walks.txt'.format(config.folder_pickles))
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, hs=0, sg=1,negative=5, ns_exponent=0.75, workers=args.workers, epochs=args.iter)
	model.wv.save_word2vec_format("emb/{}.emb".format(basename))
	logging.info("Representations created.")
	return

def exec_weighteds2v_complex(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	if(args.OPT3):
		until_layer = args.until_layer
	else:
		until_layer = None

	G, bin_coeffient = read_graph_weighted()
	print('read complete')
	G = sdw2vec.Graph_weighted(G, args.rescale, args.workers, untilLayer = until_layer)
	print('sdw2vec graph complete')
	if(args.OPT1):
		G.preprocess_neighbors_with_bfs_compact() # TODO
	else:
		G.preprocess_neighbors_with_bfs()

	if(args.OPT2):
		G.create_vectors_sdw()
		G.calc_distances_sdw(bin_coeffient, compactDegree = args.OPT1)
	else:
		best_k = G.calc_distances_all_vertices_presearch(bin_coeffient, compactDegree = args.OPT1)
		G.calc_distances_all_vertices(bin_coeffient, compactDegree = args.OPT1, best_k=best_k)
	logging.info("calc_distances_XXX end")
	G.create_distances_network()
	logging.info("create_distances_network end")
	G.preprocess_parameters_random_walk()
	print('multi-layer network generation complete')
	logging.info("multi-layer network generation complete")
	G.simulate_walks(args.num_walks, args.walk_length, args.suffix)
	print('random walk complete')
	return G

def main(args):
	print('Process start')
	G = exec_weighteds2v_complex(args)
	print('complete network generations.')
	if (args.output is not None):
		basename = args.output
	else:
		basename = args.input.split("/")[1].split(".")[0]
	learn_embeddings(basename, args.suffix)

if __name__ == "__main__":
	args = parse_args()
	print (args)
	dir_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	config.folder_pickles = dir_f+"/../pickles/{}/".format(args.suffix)
	os.makedirs(config.folder_pickles, exist_ok=True)
	logging.basicConfig(filename='{}/weighteds2v.log'.format(config.folder_pickles),filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')
	logging.info("Args={}".format(args))

	# print (config.folder_pickles)
	main(args)