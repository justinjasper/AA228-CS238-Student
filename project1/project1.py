import sys
import pandas as pd
import numpy as np
from scipy.special import gammaln
import networkx as nx
from collections import defaultdict
import networkx


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def counts(vars, data, parentNodes):
    m_ijk_count = {}
    j_i_count = {}
    for row in data.itertuples(index=False):
        var_value = getattr(row, vars)
        parent_values = tuple(getattr(row, p) for p in parentNodes)

        m_ijk_count[(parent_values, var_value)] = m_ijk_count.get((parent_values, var_value), 0) + 1
        j_i_count[parent_values] = j_i_count.get(parent_values, 0) + 1

    return m_ijk_count, j_i_count



def bayesian_score(graph, data):
    total_score = 0.0
    
    for node in graph.nodes():
        parentNodes = list(graph.predecessors(node))
        r_i = len(set(data[node])) # num possible values of the node
        
        # m_ijk_count is the joint counts of (parents = j, child = k)
        # j_i_count is the total times we see each parent instantiation j
        m_ijk_count, j_i_count = counts(data, node, parentNodes)

        for parent_values, m_ij0 in j_i_count.items():
            total_score += gammaln(r_i) - gammaln(r_i + m_ij0)
            for k in data[node].unique():
                num_Mijk = m_ijk_count.get((parent_values, k), 0)
                total_score += gammaln(1 + num_Mijk) - gammaln(1)
    return total_score


def compute(infile, outfile):
    #read CSV, 1st row = header
    df = pd.read_csv(infile, header=0)
    nodes = list(df.columns)
    #data = df.to_numpy(dtype=int)
    #n, d = data.shape #d = num vars
    
    # Number of discrete values per variable
    r  = [int(df[col].max()) for col in nodes]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    score = bayesian_score(G, df)
    print(f"Initial Bayesian Score: {score:.6f}")




    

    



    # RUN PROGRAM HERE



    #write_gph(G, idx2names, outfile)
    print(f"Wrote graph to {outfile}")


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
