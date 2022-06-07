import sys, os, subprocess, random, time, pickle, json
import networkx as nx
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from itertools import permutations
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
try:
    import cvxpy as cy
except:
    print("\ncvxpy not found\n")


NUM_CPUS = cpu_count()

#######################################
# Throughout most of this document,
# graphs will be represented by lists
# of edges
#######################################

# Take a list of edges and convert it to an adjacency matrix
def adjMatrix(nvertices, graph):
    mat = [[0 for i in range(nvertices)] for j in range(nvertices)]
    for pair in graph:
        # Note that we number our vertices starting at one
        mat[pair[0]-1][pair[1]-1] = 1
        mat[pair[1]-1][pair[0]-1] = 1
    return np.array(mat)

# Write graph to a string (e.g. in preparation to write to a file)
def graphListToString(graphList, verbose=False):
    ret = ""
    for pair in graphList:
        line = "{} {}".format(pair[0],pair[1])
        ret += line
        ret += "\n"
    if verbose:
        print()
        print(ret)
    return ret

# Read a graph from a string
def stringToGraphList(string):
    graph = string.split("\n")
    graph = [e.split() for e in graph]
    graph = [tuple([int(x) for x in e]) for e in graph]
    graph = [e for e in graph if len(e)==2]
    return graph

# write graph to a file
def graphToFile(graphList, fileName, verbose=False):
    to_write = graphListToString(graphList, verbose)
    file = open(fileName,'w')
    file.write(to_write)
    file.close()

# read a graph from a file
def read_file(name):
    file = open(name, 'r')
    string = file.read()
    file.close()
    graph = stringToGraphList(string)
    return graph


# visualize graph using networkx and pyplot
def visualizeGraph(graphList):
    G = nx.Graph()
    for pair in graphList:
        G.add_edge(pair[0],pair[1])
    nx.draw_circular(G, with_labels=True)
    plt.show()

######################
# A bunch of metrics
#####################

# Graph 'distance' based on difference between number of edges
def edgeCountDistance(graphA, graphB):
    return int(np.abs(len(graphA)-len(graphB)))

def specDistance(graphA, graphB):
    nodeVals = [pair[1] for pair in graphA] + [pair[1] for pair in graphB]
    if len(nodeVals)==0:
        return 0
    nvertices = max(nodeVals)
    matA = adjMatrix(nvertices, graphA)
    matB = adjMatrix(nvertices, graphB)
    lamA, vecA = np.linalg.eig(matA)
    lamB, vecB = np.linalg.eig(matB)
    #lamListA = list(lamA)
    #lamListB = list(lamB)
    #lamListA.sort()
    #lamListB.sort()
    lamA.sort()
    lamB.sort()
    distList = list(lamA-lamB)
    return float(np.sqrt(sum([x**2 for x in distList])))

def doublyStochasticMatrixDistance(graphA, graphB):
    nodeVals = [pair[1] for pair in graphA] + [pair[1] for pair in graphB]
    if len(nodeVals)==0:
        return 0
    nvertices = max(nodeVals)
    A = adjMatrix(nvertices, graphA)
    B = adjMatrix(nvertices, graphB)
    ones = cy.Constant(np.ones(nvertices))       # constant vector of all ones
    P    = cy.Variable((nvertices,nvertices),nonneg=True)   # matrix P
    obj  = cy.Minimize(cy.norm(A@P-P@B))     # objective function
    cons = ([P@ones == ones, ones.T@P == ones]) # doubly stochastic P
    p    = cy.Problem(obj, cons)            # problem definition

    # solve problem:
    p.solve(solver='SCS', verbose= False)
    return obj.value


def disagreementCount(graphA, graphB):
    diffs = [e for e in graphA if not e in graphB] + [e for e in graphB if not e in graphA]
    return len(diffs)

# Apply a permutation to a graph
def permuteGraph(perm, graph):
    nodeVals = [pair[1] for pair in graph]
    nvertices = max(nodeVals)
    ret = []
    for pair in graph:
        A = pair[0]
        B = pair[1]
        permA = perm[A-1]+1
        permB = perm[B-1]+1
        new_pair = [permA, permB]
        new_pair.sort()
        ret.append(tuple(new_pair))
    return ret


def minDistance(graphA, graphB):
    nodeVals = [pair[1] for pair in graphA] + [pair[1] for pair in graphB]
    if len(nodeVals)==0:
        nvertices = 0
        return 0
    nvertices = max(nodeVals)
    perms = permutations(list(range(nvertices)))
    ret = nvertices
    for perm in perms:
        dist = disagreementCount(graphA, permuteGraph(perm, graphB))
        if dist<ret:
            ret = dist
    return ret


def meanDistance(graphA, graphB):
    nodeVals = [pair[1] for pair in graphA] + [pair[1] for pair in graphB]
    if len(nodeVals)==0:
        nvertices = 0
        return 0
    nvertices = max(nodeVals)
    perms = permutations(list(range(nvertices)))
    ret = 0
    nperms = 0
    for perm in perms:
        ret += disagreementCount(graphA, permuteGraph(perm, graphB))
        nperms +=1
    ret = ret/nperms
    return ret

def minAndMeanDistCUDA(graphA, graphB, ngpus=1, randomGPU=True):
    gpu = 0
    if randomGPU:
        gpu = random.randint(0,ngpus-1)
    nodeVals = [pair[1] for pair in graphA] + [pair[1] for pair in graphB]
    if len(nodeVals)==0:
        nvertices = 0
        return 0
    nvertices = max(nodeVals)
    fileA ="graphs/tempA"+ str(random.randint(0,10**9))+ ".txt"
    fileB ="graphs/tempB"+ str(random.randint(0,10**9))+ ".txt"
    outputPermFile = "graphs/tempOut"+ str(random.randint(0,10**9))+ ".txt"
    outputFile2 = "graphs/tempOut2" + str(random.randint(0,10**9))+ ".txt"
    graphToFile(graphA, fileA)
    graphToFile(graphB, fileB)
    os.system("./bruteforce {} {} {} 1 0 0 {} {} >> {}".format(fileA, fileB, outputPermFile, nvertices, gpu, outputFile2))
    #The arguments must be filenameA, filenameB, outputfile, L1vsL2, directed/undirected, gpu/cpu, size, GPUID
    file = open(outputFile2,'r')
    output = file.read()
    try:
        outputLines = output.split('\n')
        outputLines1 = [line for line in outputLines if 'GPU Opt' in line]
        outputLines2 = [line for line in outputLines if 'GPU Mean' in line]
        minDist = float(outputLines1[0].split()[-1])
        meanDist = float(outputLines2[0].split()[-1])
        os.system("rm {}".format(outputPermFile))
        os.system("rm {}".format(outputFile2))
        os.system("rm {}".format(fileA))
        os.system("rm {}".format(fileB))
    except:
        minDist = -1
        meanDist = -1
    return minDist, meanDist

def minDistanceCUDA(graphA, graphB, randomGPU=True):
    return minAndMeanDistCUDA(graphA, graphB, randomGPU)[0]

def meanDistanceCUDA(graphA, graphB, randomGPU=True):
    return minAndMeanDistCUDA(graphA, graphB, randomGPU)[1]


#########################
# Generate random graphs
#########################


# Create a correlated pair of Erdos-Renyi graphs
def CorrelatedERPair(num_vertices, p11, p10, p01, verbose=False, to_file=False, fileNameA=None,fileNameB=None):

    pairs = [(a+1,b+1) for b in range(num_vertices) for a in range(b)]

    listA = []
    listB = []

    for pair in pairs:
        r = random.random()
        rpair = random.choices([(1,1),(1,0),(0,1),(0,0)],weights=[p11,p10,p01,1-p11-p10-p01])[0]
        if rpair[0]==1:
            listA.append(pair)
        if rpair[1]==1:
            listB.append(pair)

    if to_file:
        graphToFile(listA, fileNameA, verbose)
        graphToFile(listB, fileNameB, verbose)
    elif verbose:
        graphListToString(listA, True)
        graphListToString(listB, True)

    return listA, listB

# Different parameterization of CorrelatedERPair
def CorrelatedERPairPQ(num_vertices, p, flipProb0, flipProb1, verbose=False, to_file=False, fileNameA=None,fileNameB=None):
    p11 = p*(1-flipProb1)+(1-p)*flipProb0
    p01 = (1-p)*flipProb0
    p10 = p*flipProb1
    return CorrelatedERPair(num_vertices, p11, p10, p01, verbose, to_file, fileNameA,fileNameB)

# Given a graph, form a new graph correlated to it, such that if the first graph is ER, the pair is CER
def graphFromConditional(seedGraph, numVertices, flipProb0, flipProb1, verbose=False, to_file=False, filename=None):

    pairs = [(a+1,b+1) for b in range(numVertices) for a in range(b)]

    ret = []

    for pair in pairs:
        if pair in seedGraph:
            choice = random.choices([0,1], weights=[flipProb1, 1-flipProb1])[0]
        else:
            choice = random.choices([0,1], weights=[1-flipProb0,flipProb0])[0]
        if choice==1:
            ret.append(pair)

    if to_file:
        graphToFile(ret, filename, verbose)
    elif verbose:
        graphListToString(ret, True)

    return ret

# Create a chain of graphs which are pairwise CER
def graphChain(nvertices, ngraphs, p, flipProb0, flipProb1, verbose=False, to_file=False, fileNameRoot="graph"):
    if ngraphs<2:
        return
    fileNameList = ["{}{}.txt".format(fileNameRoot, i) for i in range(ngraphs)]
    g0, g1 = CorrelatedERPairPQ(nvertices, p, flipProb0, flipProb1, verbose, to_file, fileNameList[0],fileNameList[1])
    graphList = [g0, g1]
    for i in range(2,ngraphs):
        graphList.append(graphFromConditional(graphList[i-1], nvertices, flipProb0, flipProb1, verbose, to_file, fileNameList[i]))
    return graphList

# Given a list of graphs, order them according to their edge counts
def order_by_edge_counts(graphs, shuffle=False):
    to_order = [(i,len(graphs[i])) for i in range(len(graphs))]
    to_order.reverse()
    if shuffle:
        random.shuffle(to_order)
    ordered = sorted(to_order, key= lambda x: x[1])
    return [x[0] for x in ordered]

# implement greedy algorithm for a metric. This assumes that the first element of graphs is the true first element
def order_by_metric_greedy_cached(metric, graphs, shuffle=False):
    to_order = [(i, graphs[i]) for i in range(len(graphs))]
    metric_cache = {(a,b): -1 for b in range(len(graphs)) for a in range(b+1)}
    graphsLeft = to_order[1:]
    graphsLeft.reverse()
    if shuffle:
        random.shuffle(graphsLeft)
    graphsOrdered = [to_order[0]]
    while len(graphsLeft)>0:
        indexMin = 0
        for i in range(1,len(graphsLeft)):
            a = graphsOrdered[-1][0]
            b = graphsLeft[indexMin][0]
            c = graphsLeft[i][0]
            currentBestLen = metric_cache[(min([a,b]),max([a,b]))]
            if currentBestLen<0:
                currentBestLen = metric(graphsLeft[indexMin][1], graphsOrdered[-1][1])
                metric_cache[(min([a,b]),max([a,b]))] = currentBestLen
            lenAD = [ metric_cache[ min([a,d]),max([a,d]) ] for d in range(len(graphs)) if not d==a and not d==b ]
            lenCD = [ metric_cache[ min([c,d]),max([c,d]) ] for d in range(len(graphs)) if not d==a and not d==b ]
            LBList = [np.abs(lenAD[i] - lenCD[i]) for i in range(len(lenAD)) if lenAD[i]>=0 and lenCD[i]>=0]
            lowerBound = 0
            if len(LBList)>0:
                lowerBound = max(LBList)
            if currentBestLen > lowerBound:
                thisLen = metric(graphsLeft[i][1], graphsOrdered[-1][1])
                metric_cache[(min([a,c]),max([a,c]))] = thisLen
                if currentBestLen > thisLen:
                    indexMin = i
        graphsOrdered.append(graphsLeft.pop(indexMin))
    return [x[0] for x in graphsOrdered]

# implement greedy algorithm for a metric. This assumes that the first element of graphs is the true first element
def order_by_metric_greedy(metric, graphs, shuffle=False):
    to_order = [(i, graphs[i]) for i in range(len(graphs))]
    graphsLeft = to_order[1:]
    graphsLeft.reverse()
    if shuffle:
        random.shuffle(graphsLeft)
    graphsOrdered = [to_order[0]]
    while len(graphsLeft)>0:
        indexMin = 0
        currentBestLen = metric(graphsLeft[indexMin][1], graphsOrdered[-1][1])
        for i in range(1,len(graphsLeft)):
            thisLen = metric(graphsLeft[i][1], graphsOrdered[-1][1])
            if currentBestLen > thisLen:
                indexMin = i
                currentBestLen = thisLen
        graphsOrdered.append(graphsLeft.pop(indexMin))
    return [x[0] for x in graphsOrdered]

def order_from_distance_matrix(dmat, shuffle=False):
    ngraphs = max([key[1] for key in dmat.keys()])
    graphsLeft = [i for i in range(1,ngraphs)]
    graphsLeft.reverse()
    if shuffle:
        random.shuffle(graphsLeft)
    ordered = [0]
    while len(graphsLeft)>0:
        indexMin = 0
        bestPair = [ordered[-1],graphsLeft[indexMin]]
        bestPair = (min(bestPair), max(bestPair))
        currentBestLen = dmat[bestPair]
        for i in range(1,len(graphsLeft)):
            thisPair = [ordered[-1],graphsLeft[i]]
            thisPair = (min(thisPair), max(thisPair))
            thisLen = dmat[thisPair]
            if currentBestLen > thisLen:
                indexMin = i
                currentBestLen = thisLen
        ordered.append(graphsLeft.pop(indexMin))
    return ordered

def order_from_distance_matrix_list(dmat, shuffle=False):
    ngraphs = len(dmat)
    graphsLeft = [i for i in range(1,ngraphs)]
    graphsLeft.reverse()
    if shuffle:
        random.shuffle(graphsLeft)
    ordered = [0]
    while len(graphsLeft)>0:
        indexMin = 0
        bestPair = [ordered[-1],graphsLeft[indexMin]]
        currentBestLen = dmat[bestPair[0]][bestPair[1]]
        for i in range(1,len(graphsLeft)):
            thisPair = [ordered[-1],graphsLeft[i]]
            thisLen = dmat[thisPair[0]][thisPair[1]]
            if currentBestLen > thisLen:
                indexMin = i
                currentBestLen = thisLen
        ordered.append(graphsLeft.pop(indexMin))
    return ordered


def populate_folder(nshots, nvertices, ngraphs, p, q, qq, dir):
    subdir = "N{}_p_{}_q_{}_qq_{}".format(nvertices,p,q,qq)
    os.system("mkdir {}/{}".format(dir, subdir))
    for i in range(nshots):
        shotdir = "{}/{}/run{}".format(dir, subdir, i)
        os.system("mkdir {}".format(shotdir))
        fileRoot="{}/graph".format(shotdir)
        graphChain(nvertices, ngraphs, p, q, qq, verbose=False, to_file=True,fileNameRoot=fileRoot)

def read_chain(dir, ngraphs):
    files = bytes(subprocess.check_output(["ls",dir])).decode('utf-8').split()
    ngraphs = min([ngraphs,len(files)])
    chain = []
    for i in range(ngraphs):
        fstring = "{}/graph{}.txt".format(dir,i)
        graph = read_file(fstring)
        chain.append(graph)
    return chain

def read_ensemble(dir,nshots,chain_length):
    ensemble = []
    folders = bytes(subprocess.check_output(["ls",dir])).decode('utf-8').split()
    if nshots<len(folders):
        folders = folders[:nshots]
    for folder in folders:
        chain = read_chain("{}/{}".format(dir,folder), chain_length)
        ensemble.append(chain)
    return ensemble

# This is taken verbatim from https://stackoverflow.com/questions/16488684/how-to-unpickle-a-file-in-python
def unpickle(filename):
    f = open(filename, "rb")
    d = pickle.load(f)
    f.close()
    return d

def pairwise_distance_matrix(graphs,metric,to_file=False, file=None, parallel=False):
    L = len(graphs)
    distances = {}
    if parallel:
        pairs = [(i,j) for j in range(L) for i in range(j)]
        my_func = lambda x: (x[0], x[1], metric(graphs[x[0]],graphs[x[1]]))
        pool = Pool(NUM_CPUS)
        results = pool.map(my_func, pairs)
        distances = {(x[0], x[1]): x[2] for x in results}
    else:
        for j in range(L):
            for i in range(j):
                '''
                if parallel:
                    metricNames = {minDistanceCUDA: "minDistanceCUDA", meanDistanceCUDA: "meanDistanceCUDA", specDistance: "specDistance", edgeCountDistance: "edgeCountDistance", disagreementCount: "disagreementCount", doublyStochasticMatrixDistance: "doublyStochasticMatrixDistance"}
                    name = metricNames[metric]
                    d=float(bytes(subprocess.check_output(["python3", "CERDMatParallel.py",name,graphStrings[i],graphStrings[j]])).decode('utf-8').split()[0])
                else:
                '''
                d = metric(graphs[i],graphs[j])
                distances.update({(i,j): d})
    if to_file:
        outp = open(file,'wb')
        pickle.dump(distances, outp, -1)
    return distances

def distWrapper(x, graphs):
    i = x[0]
    j = x[1]
    dMin, dMean = minAndMeanDistCUDA(graphs[i], graphs[j])
    print ("\npair ({},{}) dMin = {} dMean = {}\n".format(i,j,dMin,dMean))
    return (i, j, (dMin, dMean))

def pairwise_distance_matrix_CUDA(graphs, ngpus=1, parallel=False):
    L = len(graphs)
    minDistances = {}
    meanDistances = {}
    if parallel:
        pairs = [(i,j) for j in range(L) for i in range(j)]
        my_func = lambda x: distWrapper(x, graphs)
        pool = Pool(NUM_CPUS)
        results = pool.map(my_func, pairs)
        minDistances = {(x[0], x[1]): x[2][0] for x in results}
        meannDistances = {(x[0], x[1]): x[2][1] for x in results}
    else:
        for j in range(L):
            for i in range(j):
                dMin, dMean = minAndMeanDistCUDA( graphs[i], graphs[j], ngpus)
                print("\nDoing pair ({},{}) dMin={} dMean={}\n".format(i,j,dMin,dMean))
                minDistances.update({(i,j): dMin})
                meanDistances.update({(i,j): dMean})
    return minDistances, meanDistances

def pairwise_distance_dict_to_list(dmat_dict, L=-1):
    if L<1:
        L = max([k[1] for k in dmat_dict.keys()])-1
    dmat_list = [[float(dmat_dict[(i,j)]) if i<j else dmat_dict[(j,i)] if j<i else 0 for i in range(L)] for j in range(L)]
    return dmat_list

def pairwise_distances_from_folder(folder, ngraphs, metric, to_file=False, file_name=None):
    graphs = read_chain(folder, ngraphs)
    file_path = "{}/{}.pkl".format(folder, file_name)
    distances = pairwise_distance_matrix(graphs, metric, to_file, file_path)
    return distances

def pairwise_distance_ensemble(dir, nshots, ngraphs, metric, metric_name):
    folders = bytes(subprocess.check_output(["ls",dir])).decode('utf-8').split()
    if nshots<len(folders):
        folders = folders[:nshots]
    for folder in folders:
        path = "{}/{}".format(dir,folder)
        pairwise_distances_from_folder(path, ngraphs, metric, True, metric_name)

def pairwise_distance_ensemble_parallel(dir, nshots, ngraphs, metric_name):
    folders = bytes(subprocess.check_output(["ls",dir])).decode('utf-8').split()
    if nshots<len(folders):
        folders = folders[:nshots]
    for folder in folders:
        path = "{}/{}".format(dir,folder)
        os.system("python3 CERParallel.py {} {} {}".format(ngraphs, metric_name, path))

def evaluate_perms(perms):
    total_number = len(perms)
    if len(perms)<1:
        return []
    nvertices = len(perms[0])
    perm_scores = [[int(p[i]==i) for i in range(nvertices)] for p in perms]
    correct_at_index = reduce(lambda x, y: [x[i]+y[i] for i in range(min([len(x),len(y)]))], perm_scores)
    correct_at_index = [p/total_number for p in correct_at_index]
    correct_to_index = []
    correct_perms = [p for p in perm_scores]
    for i in range(nvertices):
        correct_perms = [p for p in correct_perms if p[i]==1]
        correct_count = len(correct_perms)
        correct_to_index.append(correct_count/total_number)
    return correct_to_index, correct_at_index



def evaluate_metric(chains, orderFun):
    perms = [orderFun(chain) for chain in chains]
    return evaluate_perms(perms)

def evaluate_metric_from_dmats_list(dmats):
    perms = [order_from_distance_matrix_list(dmat) for dmat in dmats]
    return evaluate_perms(perms)

def evaluate_metric_from_dmats_dict(dmats):
    perms = [order_from_distance_matrix(dmat) for dmat in dmats]
    return evaluate_perms(perms)

def evaluate_all(dir,nshots,chain_length):

    chains = read_ensemble(dir,nshots,chain_length)
    #print()
    #print(chains[0][0])
    print()

    print("Performance of edge counting")
    print()
    print(evaluate_metric(chains, order_by_edge_counts))
    print()

    print("disagreementCount")
    print()
    print(evaluate_metric(chains, lambda x: order_by_metric_greedy(disagreementCount, x)))
    print()

    print("edgeCountDistance")
    print()
    print(evaluate_metric(chains, lambda x: order_by_metric_greedy(edgeCountDistance, x)))
    print()

    print("specDistance")
    print()
    print(evaluate_metric(chains, lambda x: order_by_metric_greedy(specDistance, x)))
    print()

    print("doublyStochasticMatrixDistance")
    print()
    print(evaluate_metric(chains, lambda x: order_by_metric_greedy_cached(doublyStochasticMatrixDistance, x)))
    print()

    print("minDistanceCUDA")
    print()
    print( evaluate_metric( chains, lambda x: order_by_metric_greedy_cached(minDistanceCUDA, x) ) )
    print()


###########################
# Store data in json
# instead of .txt and .pkl
# documenmts
###########################

def update_json_with_chains(fname, nshots, ngraphs, n, p, q, skip=False):
    chains = [graphChain(n, ngraphs, p, q, q) for i in range(nshots)]
    json_file = open(fname,"r")
    dict = json.load(json_file)
    json_file.close()
    n = str(n)
    p = str(p)
    q = str(q)
    if n in dict.keys():
        if p in dict[n].keys():
            if q in dict[n][p].keys():
                if 'chains' in dict[n][p][q].keys():
                    if not skip:
                        dict[n][p][q]['chains'] = dict[n][p][q]['chains'] + chains
                else:
                    dict[n][p][q].update({'chains': chains})
            else:
                dict[n][p].update({q: {'chains': chains}})
        else:
            dict[n].update({p: {q: {'chains': chains}}})
    else:
        dict.update({n: {p: {q: {'chains': chains}}}})
    json_file = open(fname, "w")
    json.dump(dict, json_file)
    json_file.close()

def update_json_with_chains_from_folders(fname, nshots, ngraphs, n, p, q):
    dir = "graphs/N{}_p_{}_q_{}_qq_{}".format(n, p, q, q)
    chains = read_ensemble(dir,nshots,ngraphs)
    json_file = open(fname,"r")
    dict = json.load(json_file)
    json_file.close()
    n = str(n)
    p = str(p)
    q = str(q)
    if n in dict.keys():
        if p in dict[n].keys():
            if q in dict[n][p].keys():
                if 'chains' in dict[n][p][q].keys():
                    dict[n][p][q]['chains'] = dict[n][p][q]['chains'] + chains
                else:
                    dict[n][p][q].update({'chains': chains})
            else:
                dict[n][p].update({q: {'chains': chains}})
        else:
            dict[n].update({p: {q: {'chains': chains}}})
    else:
        dict.update({n: {p: {q: {'chains': chains}}}})
    json_file = open(fname, "w")
    json.dump(dict, json_file)
    json_file.close()

def json_report(fname):
    json_file = open(fname,"r")
    dict = json.load(json_file)
    json_file.close()
    print()
    print(fname)
    print()
    for x in sorted([int(key) for key in dict.keys()]):
        n = str(x)
        print("N = {}:".format(n))
        for y in sorted([float(k) for k in dict[n].keys()]):
            p = str(y)
            print("  p = {}:".format(p))
            for z in sorted([float(k) for k in dict[n][p].keys()]):
                q = str(z)
                print("    q = {}".format(q))
                if 'chains' in dict[n][p][q].keys():
                    chains = dict[n][p][q]['chains']
                    nchains = len(chains)
                    print("      Number of chains: {}".format(nchains))
                    lengths = set([len(x) for x in chains])
                    for l in lengths:
                        lcount = len([x for x in chains if len(x)==l])
                        print("        L={}: {}".format(l,lcount))
                else:
                    for metric in dict[n][p][q].keys():
                        print("      {}:".format(metric))
                        #dmats = dict[n][p][q][metric]
                        #print("        {}".format(dmats))
                    '''
                    for metric in dict[n][p][q].keys():
                        print("      {}:".format(metric))
                        dmats = dict[n][p][q][metric]
                        try:
                            lengths = set([len(x) for x in  dmats if x>3])
                            ndmats = len(dmats)
                            print("        Number of distance matrices: {}".format(ndmats))
                            for l in lengths:
                                lcount = len([x for x in dmats if len(x)==l])
                                print("          L={}: {}".format(l,lcount))
                        except:
                            print("        {}".format(dmats))
                    '''

    print()
    print()


def chains_from_json(fname,n,p,q,L,strict=False):
    json_file = open(fname,"r")
    dict = json.load(json_file)
    json_file.close()
    n = str(n)
    p = str(p)
    q = str(q)
    chains = dict[n][p][q]['chains']
    if strict:
        chains = [x for x in chains if len(x)==L]
    else:
        chains = [x for x in chains if len(x)>=L]
        chains = [x[:L] for x in chains]

    return(chains)

def dmats_from_json(fname, metric, n, p, q, L, strict):
    chains = chains_from_json(fname, n, p, q, L, strict)
    dmats = [pairwise_distance_matrix(c, metric) for c in chains]
    return dmats

def chains_to_dmats_json(fin, fout, metric, metric_name, parallel=True):
    json_file = open(fin,"r")
    dict_in = json.load(json_file)
    json_file.close()
    try:
        json_file = open(fout,"r")
        dict_out = json.load(json_file)
        json_file.close()
    except:
        dict_out = {}
    nlist = [n for n in dict_in.keys() if int(n)<15]
    for n in nlist:
        for p in dict_in[n].keys():
            for q in dict_in[n][p].keys():
                chains = dict_in[n][p][q]['chains']
                if n in dict_out.keys():
                    if p in dict_out[n].keys():
                        if q in dict_out[n][p].keys():
                            if not metric_name in dict_out[n][p][q].keys():
                                dmats = [pairwise_distance_dict_to_list(pairwise_distance_matrix(c, metric, False, None, parallel)) for c in chains]
                                dict_out[n][p][q].update({metric_name: dmats})
                        else:
                            dmats = [pairwise_distance_dict_to_list(pairwise_distance_matrix(c, metric, False, None, parallel)) for c in chains]
                            dict_out[n][p].update({q: {metric_name: dmats}})
                    else:
                        dmats = [pairwise_distance_dict_to_list(pairwise_distance_matrix(c, metric, False, None, parallel)) for c in chains]
                        dict_out[n].update({p: {q: {metric_name: dmats}}})
                else:
                    dmats = [pairwise_distance_dict_to_list(pairwise_distance_matrix(c, metric, False, None, parallel)) for c in chains]
                    dict_out.update({n: {p: {q: {metric_name: dmats}}}})
    json_file = open(fout, "w")
    json.dump(dict_out, json_file)
    json_file.close()

def chains_to_dmats_json_CUDA(fin, fout, ngpus=1, parallel=True):
    json_file = open(fin,"r")
    dict_in = json.load(json_file)
    json_file.close()
    try:
        json_file = open(fout,"r")
        dict_out = json.load(json_file)
        json_file.close()
    except:
        dict_out = {}
    nlist = [n for n in dict_in.keys() if int(n)<15]
    metrics = [minDistanceCUDA, meanDistanceCUDA]
    metric_names = {minDistanceCUDA: "minDistanceCUDA", meanDistanceCUDA: "meanDistanceCUDA"}
    metric_pos = {minDistanceCUDA: 0, meanDistanceCUDA: 1}
    for n in nlist:
        for p in dict_in[n].keys():
            for q in dict_in[n][p].keys():
                chains = dict_in[n][p][q]['chains']
                dmat_pairs = [pairwise_distance_matrix_CUDA(c, ngpus, parallel) for c in chains]
                for metric in metrics:
                    metric_name = metric_names[metric]
                    if n in dict_out.keys():
                        if p in dict_out[n].keys():
                            if q in dict_out[n][p].keys():
                                if not metric_name in dict_out[n][p][q].keys():
                                    dmats = [pairwise_distance_dict_to_list(c[metric_pos[metric]]) for c in dmat_pairs]
                                    dict_out[n][p][q].update({metric_name: dmats})
                            else:
                                dmats = [pairwise_distance_dict_to_list(c[metric_pos[metric]]) for c in dmat_pairs]
                                dict_out[n][p].update({q: {metric_name: dmats}})
                        else:
                            dmats = [pairwise_distance_dict_to_list(c[metric_pos[metric]]) for c in dmat_pairs]
                            dict_out[n].update({p: {q: {metric_name: dmats}}})
                    else:
                        dmats = [pairwise_distance_dict_to_list(c[metric_pos[metric]]) for c in dmat_pairs]
                        dict_out.update({n: {p: {q: {metric_name: dmats}}}})
    json_file = open(fout, "w")
    json.dump(dict_out, json_file)
    json_file.close()

def chains_to_dmats_json_partial_CUDA(fin, fout, n, p, q, ngpus=1, parallel=True):
    json_file = open(fin,"r")
    dict_in = json.load(json_file)
    json_file.close()
    try:
        json_file = open(fout,"r")
        dict_out = json.load(json_file)
        json_file.close()
    except:
        dict_out = {}
    nlist = [n for n in dict_in.keys() if int(n)<15]
    metrics = [minDistanceCUDA, meanDistanceCUDA]
    metric_names = {minDistanceCUDA: "minDistanceCUDA", meanDistanceCUDA: "meanDistanceCUDA"}
    metric_pos = {minDistanceCUDA: 0, meanDistanceCUDA: 1}
    print("N={}, p={}, q={}, keys={}\n".format(n,p,q,dict_in.keys()))
    n = str(n)
    p = str(p)
    q = str(q)
    chains = dict_in[n][p][q]['chains']
    dmat_pairs = [pairwise_distance_matrix_CUDA(c, ngpus, parallel) for c in chains]
    for metric in metrics:
        metric_name = metric_names[metric]
        if n in dict_out.keys():
            if p in dict_out[n].keys():
                if q in dict_out[n][p].keys():
                    if not metric_name in dict_out[n][p][q].keys():
                        dmats = [pairwise_distance_dict_to_list(c[metric_pos[metric]]) for c in dmat_pairs]
                        dict_out[n][p][q].update({metric_name: dmats})
                else:
                    dmats = [pairwise_distance_dict_to_list(c[metric_pos[metric]]) for c in dmat_pairs]
                    dict_out[n][p].update({q: {metric_name: dmats}})
            else:
                dmats = [pairwise_distance_dict_to_list(c[metric_pos[metric]]) for c in dmat_pairs]
                dict_out[n].update({p: {q: {metric_name: dmats}}})
        else:
            dmats = [pairwise_distance_dict_to_list(c[metric_pos[metric]]) for c in dmat_pairs]
            dict_out.update({n: {p: {q: {metric_name: dmats}}}})
    json_file = open(fout, "w")
    json.dump(dict_out, json_file)
    json_file.close()

def chains_to_dmats_json_partial(fin, fout, metric, metric_name, n, p, q, parallel=True):
    json_file = open(fin,"r")
    dict_in = json.load(json_file)
    json_file.close()
    try:
        json_file = open(fout,"r")
        dict_out = json.load(json_file)
        json_file.close()
    except:
        dict_out = {}
    n = str(n)
    p = str(p)
    q = str(q)
    if n in dict_in.keys():
        if p in dict_in[n].keys():
            if  q in dict_in[n][p].keys():
                chains = dict_in[n][p][q]['chains']
                dmats = [pairwise_distance_dict_to_list(pairwise_distance_matrix(c, metric, False, None, parallel)) for c in chains]
                if n in dict_out.keys():
                    if p in dict_out[n].keys():
                        if q in dict_out[n][p].keys():
                            dict_out[n][p][q].update({metric_name: dmats})
                        else:
                            dict_out[n][p].update({q: {metric_name: dmats}})
                    else:
                        dict_out[n].update({p: {q: {metric_name: dmats}}})
                else:
                    dict_out.update({n: {p: {q: {metric_name: dmats}}}})
            else:
                print("q not found")
                return
        else:
            print("p not foundt")
            return
    else:
        print("n not found")
        return
    json_file = open(fout, "w")
    json.dump(dict_out, json_file)
    json_file.close()


def dmats_to_greedy_evals(fin,fout):
    json_file = open(fin,"r")
    dict_in = json.load(json_file)
    json_file.close()
    try:
        json_file = open(fout,"r")
        dict_out = json.load(json_file)
        json_file.close()
    except:
        dict_out = {}
    nlist = [n for n in dict_in.keys() if int(n)<30]
    for n in nlist:
        for p in dict_in[n].keys():
            for q in dict_in[n][p].keys():
                for metric in dict_in[n][p][q].keys():
                    dmats =dict_in[n][p][q][metric]
                    evalc_name = "correct to i for {} greedy".format(metric)
                    eval_name = "correct at i for {} greedy".format(metric)
                    if n in dict_out.keys():
                        if p in dict_out[n].keys():
                            if q in dict_out[n][p].keys():
                                if not eval_name in dict_out[n][p][q].keys():
                                    eval = evaluate_metric_from_dmats_list(dmats)[1]
                                    dict_out[n][p][q].update({eval_name: eval})
                                if not evalc_name in dict_out[n][p][q].keys():
                                    evalc = evaluate_metric_from_dmats_list(dmats)[0]
                                    dict_out[n][p][q].update({evalc_name: evalc})
                            else:
                                evalc, eval = evaluate_metric_from_dmats_list(dmats)
                                dict_out[n][p].update({q: {evalc_name: evalc, eval_name: eval}})
                        else:
                            evalc, eval = evaluate_metric_from_dmats_list(dmats)
                            dict_out[n].update({p: {q: {evalc_name: evalc, eval_name: eval}}})
                    else:
                        evalc, eval = evaluate_metric_from_dmats_list(dmats)
                        dict_out.update({n: {p: {q: {evalc_name: evalc, eval_name: eval}}}})
    json_file = open(fout, "w")
    json.dump(dict_out, json_file)
    json_file.close()

def chains_to_edge_counts(fin,fout):
    json_file = open(fin,"r")
    dict_in = json.load(json_file)
    json_file.close()
    try:
        json_file = open(fout,"r")
        dict_out = json.load(json_file)
        json_file.close()
    except:
        dict_out = {}
    nlist = [n for n in dict_in.keys() if int(n)<30]
    for n in nlist:
        for p in dict_in[n].keys():
            for q in dict_in[n][p].keys():
                chains = dict_in[n][p][q]['chains']
                if n in dict_out.keys():
                    if p in dict_out[n].keys():
                        if q in dict_out[n][p].keys():
                            if not "edge counts" in dict_out[n][p][q].keys():
                                edge_counts = [len(c) for c in chains]
                                dict_out[n][p][q].update({"edge counts": edge_counts})
                        else:
                            edge_counts = [[len(g) for g in c] for c in chains]
                            dict_out[n][p].update({q: {"edge counts": edge_counts}})
                    else:
                        edge_counts = [[len(g) for g in c] for c in chains]
                        dict_out[n].update({p: {q: {"edge counts": edge_counts}}})
                else:
                    edge_counts = [[len(g) for g in c] for c in chains]
                    dict_out.update({n: {p: {q: {"edge counts": edge_counts}}}})
    json_file = open(fout, "w")
    json.dump(dict_out, json_file)
    json_file.close()

def chains_to_edge_count_scores(fin, fout):
    json_file = open(fin,"r")
    dict_in = json.load(json_file)
    json_file.close()
    try:
        json_file = open(fout,"r")
        dict_out = json.load(json_file)
        json_file.close()
    except:
        dict_out = {}
    nlist = [n for n in dict_in.keys() if int(n)<30]
    for n in nlist:
        for p in dict_in[n].keys():
            for q in dict_in[n][p].keys():
                chains = dict_in[n][p][q]['chains']
                evalc,eval = evaluate_metric(chains, order_by_edge_counts)
                evalc_name = "correct to i for edge count order"
                eval_name = "correct at i for edge count order"
                if n in dict_out.keys():
                    if p in dict_out[n].keys():
                        if q in dict_out[n][p].keys():
                            if not evalc_name in dict_out[n][p][q].keys():
                                dict_out[n][p][q].update({evalc_name: evalc})
                            if not eval_name in dict_out[n][p][q].keys():
                                dict_out[n][p][q].update({eval_name: eval})
                        else:
                            dict_out[n][p].update({q: {evalc_name: evalc, eval_name: eval}})
                    else:
                        dict_out[n].update({p: {q: {evalc_name: evalc, eval_name: eval}}})
                else:
                    dict_out.update({n: {p: {q: {evalc_name: evalc, eval_name: eval}}}})
    json_file = open(fout, "w")
    json.dump(dict_out, json_file)
    json_file.close()



#########################
# Main
#########################


# Stuff that will actually be done
def main():
    nshots = 200
    #num_vertices = int(sys.argv[1])
    #num_graphs = int(sys.argv[2])
    p = float(sys.argv[1])
    ngpus = int(sys.argv[2])
    outfile = sys.argv[3]
    #q = float(sys.argv[4])
    #qq = q

    nvertices = 12
    ngraphs = 100
    #plist = [0.05,0.1,0.2,0.3,0.4,0.5]
    qlist = [0.01,0.05,0.1,0.2,0.3]

    metrics = [edgeCountDistance, disagreementCount, specDistance, minDistanceCUDA, meanDistanceCUDA, doublyStochasticMatrixDistance]
    metricNames = {minDistanceCUDA: "minDistanceCUDA", meanDistanceCUDA: "meanDistanceCUDA", specDistance: "specDistance", edgeCountDistance: "edgeCountDistance", disagreementCount: "disagreementCount", doublyStochasticMatrixDistance: "doublyStochasticMatrixDistance"}


    json_report("data.json")

    metric = doublyStochasticMatrixDistance
    mname = metricNames[metric]
    for q in qlist:
        chains_to_dmats_json_partial_CUDA("data.json", outfile, nvertices, p, q, ngpus, parallel=True)
        print("p={}, q={}, metric={} finished".format(p,q,mname))

    '''
    for p in plist:
        for q in qlist:
            update_json_with_chains('data.json',nshots,ngraphs,nvertices,p,q,skip=True)
            print("p={}, q={} finished".format(p,q))
    '''
    #json_report("data.json")

    #json_report("dmats.json")

    #chains_to_edge_count_scores("data.json","scores.json")

    #json_report("scores.json")

    #update_json_with_chains("data.json",nshots,num_graphs,num_vertices,p,q)

    #dmats_to_greedy_evals("dmats.json", "scores.json")

    #json_report("data.json")
    #json_report("scores.json")

# Do stuff
if __name__ == '__main__':
    main()
