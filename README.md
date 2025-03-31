# SDW2vec
Code for paper titled as "SDW2vec: Learning Structural Representations of Nodes in Weighted Networks"

SDW2vec learns the embeddings for nodes based on the structural features (degrees and edge properties). SDW2vec is capable for weighted networks, signed networks, directed networks.

# Installation
Tested with Python 3.10 and the following packages needed: 
	pip install futures==3.0.5
	pip install fastdtw==0.3.4
	pip install gensim==4.3.2
	pip install numpy==1.24.4
	pip install scipy==1.11.3

# Usage
## Input
Edgelist of a signed network, 3 columns (node1 node2 weight). wight \in R.
For undirected networks, 
## Parameters

--input: Input graph path (default: 'star_weighted.edgelist')\
--output: Output embedding path (default: None, follows input file name if not given)\
--dimensions: Number of dimensions (default: 128)\
--signed: Flag for signed network (action: store_true)\
--directed: Flag for directed network (action: store_true)\
--walk-length: Length of walk per source (default: 80)\
--num-walks: Number of walks per source (default: 20)\
--window-size: Context size for optimization (default: 10)\
--until-layer: Calculation until the layer. Only work for OPT3 mode (default: 6)\
--iter: Number of epochs in SGD (default: 5)\
--workers: Number of parallel workers (default: 8)\
--OPT3: Optimization flag to control the max layer to until-layer (action: store_true)\
--rescale: Type of weight rescaling function (default: 'auto', options: 'auto', 'cut', 'qcut')\
--num-bins: Number of bins to rescale weights (default: 0, applies best search for 0)\
--suffix: Log file and pickles folder suffix (default: 'TEST')\
  
## Command example
	python src/main.py --input graph/star_weighted.edgelist --output star_weighted --num-walks 100 --walk-length 80 --window-size 5 --dimensions 2 --until-layer 5 --workers 1 --OPT3 --rescale auto --num-bins 0 --suffix star  --directed --signed

# Miscellaneous
  Feel free to send email to liu@torilab.net for any questions about the code or the paper.
  
  For unweighted networks, we strongly recommend [sds2vec]() for signed directed networks or [signeds2v](https://link.springer.com/chapter/10.1007/978-3-031-21127-0_28) for signed undirected networks.

  For a pratical usage of structural embeddings, see [Gene2Role](https://www.biorxiv.org/content/10.1101/2024.05.18.594807v1).

  For hypergraphs structural embedding, see [HyperS2V](https://arxiv.org/pdf/2311.04149).

  All of these codes could be found in my Github.

  We would like to thank Leonardo F. R. Ribeiro et al., authors of struc2vec, for providing their code.
