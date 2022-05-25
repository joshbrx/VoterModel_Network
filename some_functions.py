def grado_prom(G):             # Grado promedio
    N = len(G); k = 0
    B = dict(G.degree())
    for key in B:
        k += B[key]
    return k/N
# --------------------------------------------------
def clustering_prom(G):        # Clustering coeff
    N = len(G); s = 0
    B = nx.clustering(G)
    for key in B:
        s += B[key]
    return s/N
# --------------------------------------------------
def betweenness_prom(G, k=len(G)):       # Betweenness
    N = len(G); s = 0
    B_dict = nx.betweenness_centrality(G, k=k, normalized=False)
    for key in B_dict:
        s += B_dict[key]
    return s/N
# --------------------------------------------------
def shortest_path_length_prom(G): # Camino más corto
    N = len(G); s = 0
    p = dict(nx.shortest_path_length(G))
    for i in p:
        for j in p[i]:
            s += p[i][j]
    return s/(N*(N-1))
# --------------------------------------------------
def closeness_prom(G):         # Closeness
    N = len(G); s = 0
    p = nx.closeness_centrality(G)
    for key in p:
        s += p[key]
    return s/N
# ---------------------------------------------------
def eccentricity_prom(G):      # Excentricidad
    N = len(G); s = 0
    p = nx.eccentricity(G, v=None)
    for key in p:
        s += p[key]
    return s/N
# ---------------------------------------------------
def eigenvalue_prom(G, max_iter=100, tol=1e-06):             # Centralidad de eigenvector
    N = len(G); s = 0
    B_dict = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
    for key in B_dict:
        s += B_dict[key]
    return s/N
# --------------------------------------------------
def katz_prom(G, alpha=0.1, beta=1.0): # Camino más corto
    N = len(G); s = 0
    p = nx.katz_centrality_numpy(G, alpha=alpha, beta=beta, normalized=False)
    for key in p:
        s += p[key]
    return s/N
# --------------------------------------------------
def pagerank_prom(G, alpha=0.85):         # Closeness
    N = len(G); s = 0
    p = nx.pagerank(G, alpha=alpha, nstart={1:0.000005, 2:0.000001,3:0.000003,4:0.000002,5:0.000001,6:0.000002,7:0.000001},
                    dangling=None, max_iter=1000)
    for key in p:
        s += p[key]
    return s/N
# --------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------
def hist_centrality(G, alpha_k=0.1, beta_k=1.0, delta_pr=0.85, max_iter=500, tol=1e-04):
    N = len(G); 
    s1, s2, s3 = [], [], [] 
    B1 = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
    B2 = nx.katz_centrality_numpy(G, alpha=alpha_k, beta=beta_k, normalized=False)
    B3 = nx.pagerank(G, alpha=delta_pr, nstart={1:0.000005, 2:0.000001,3:0.000003,4:0.000002,5:0.000001,6:0.000002,7:0.000001},
                    dangling=None, max_iter=1000)
    for key in B1:
        s1.append(B1[key])
    s1 = np.array(s1)
    
    for key in B2:
        s2.append(B2[key])
    s2 = np.array(s2)
    
    for key in B3:
        s3.append(B3[key])
    s3 = np.array(s3)
    
    fig, axes = plt.subplots(1,3, figsize=(13, 5), squeeze=True)
#   ---------------------------------------------------------
    sns.histplot(data=s1, ax=axes[0], y=None, hue=None, stat='count', shrink=1,
                     binwidth=None, binrange=None, discrete=None,
                     cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars', color="orange")
    axes[0].title.set_text("Eigenvector centrality")
#   ---------------------------------------------------------
    sns.histplot(data=s2, ax=axes[1], y=None, hue=None, stat='count', shrink=1,
                     binwidth=None, binrange=None, discrete=None,
                     cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars', color="orange")
    axes[1].title.set_text("Katz centrality")
#   ---------------------------------------------------------
    sns.histplot(data=s3, ax=axes[2], y=None, hue=None, stat='count', shrink=1,
                     binwidth=None, binrange=None, discrete=None,
                     cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars', color="orange")
    axes[2].title.set_text("PageRank centrality")
    
    fig.tight_layout()
    plt.show()
# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------
    
    
def is_weighted(edge_list):
    return len(edge_list[0]) > 2
# ----------------------------------------------------
def is_directed(edge_list):
    node_list = np.unique(np.array(edge_list)[:,0:2])
    temp_list = [(i[0],i[1]) for i in edge_list]
    s = is_weighted(edge_list)
    l_1 = len(edge_list)
    l_2 = len(list(nx.edge_dfs(nx.Graph(temp_list), node_list)))
    if l_1 == l_2:
        return False
    else: 
        return True
# ----------------------------------------------------
def make_graph(edge_list):
    if is_directed(edge_list) == False:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    if is_weighted(edge_list) == False:
        G.add_edges_from(edge_list)
    else:
        G.add_weighted_edges_from(edge_list, weight='weight')
    return G
# ----------------------------------------------------
def is_connected(G):
    cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    if len(cc)==1:
        s = True
    else: 
        s = False
    return s, cc[0]
# ----------------------------------------------------
def analyse_edges(edge_list):
    dict_E = dict()
    dict_E["Weighted"] = is_weighted(edge_list)
    dict_E["Directed"] = is_directed(edge_list)
    if dict_E["Directed"] == True:
        G_temp = make_graph(edge_list).to_undirected()
    else:
        G_temp = make_graph(edge_list)
    dict_E["Connected"] = is_connected(G_temp)[0]
    dict_E["Max. connected components"] = is_connected(G_temp)[1]
    
    return dict_E
# ----------------------------------------------------
def grado_in_prom(G):
    N = len(G); s = 0
    B = dict(G.in_degree())
    for key in B:
        s += B[key]
    return s/N
# ----------------------------------------------------    
def grado_out_prom(G):
    N = len(G); s = 0
    B = dict(G.out_degree())
    for key in B:
        s += B[key]
    return s/N
# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------
def analyse_graph(G, dict_E, k=len(G)):
    dict_G = dict()
    
    if dict_E["Directed"] == True:
        while True:
            try:
                L = nx.directed_laplacian_matrix(G, weight="weight")
                L_graph = plt.matshow(L, cmap='Greys')
                plt.title("Laplacian matrix")
            except MemoryError as err:
                print(err)
                break
        dict_G["In-degree"]  = grado_in_prom(G)
        dict_G["Out-degree"] = grado_out_prom(G)
    else:
        L = nx.laplacian_matrix(G, weight='weight')
        L_graph = plt.spy(L, markersize=1, c="black")
        dict_G["Degree"]     = grado_prom(G)
        plt.title("Laplacian matrix")

    dict_G["Clustering"]     = clustering_prom(G)
    dict_G["Betweenness"]    = betweenness_prom(G, k = k)
    dict_G["Shortest-path"]  = shortest_path_length_prom(G)
    
    if dict_E["Connected"] == True:
        if dict_E["Directed"] == False:
            dict_G["Diameter"]     = nx.diameter(G)
            dict_G["Eccentricity"] = eccentricity_prom(G)
        else:
            pass
    else:
        dict_G["Diameter"]     = float('inf')
        dict_G["Eccentricity"] = float('inf')
        
    dict_G["Closeness"]      = closeness_prom(G)
    dict_G["Eigen"]      = eigenvalue_prom(G, max_iter=500, tol=1e-03)
    dict_G["Katz"]       = katz_prom(G, alpha=0.1, beta=1.0)
    dict_G["PageRank"]   = pagerank_prom(G, alpha=0.85)
    
    return dict_G
# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------
def analyse_coefficients(G, bins=10):
    N = len(G); 
    s1, s2 = [], []
    
    p1 = nx.clustering(G)
    p2 = dict(nx.shortest_path_length(G))
    
    for key in p1:
        s1.append(p1[key])

    for i in p2:
        for j in p2[i]:
            s2.append(p2[i][j])
    
    s1 = np.array(s1)
    s2 = np.array(s2)        
#   ---------------------------------------------------------
    if (type(G) == "networkx.classes.digraph.DiGraph") or (type(G) == "networkx.classes.digraph.MultiDiGraph"): 
        k_in, k_out = [], []
        B1 = dict(G.in_degree())
        B2 = dict(G.out_degree())
        for key in B1:
            k_in.append(B1[key])
            k_out.append(B2[key])
        k_in = np.array(k_in)
        k_out = np.array(k_out)
        C = [s1, s2, k_in, k_out]
    else:
        k = []
        B = dict(G.degree())
        for key in B:
            k.append(B[key])
        k = np.array(k)  
        C = [s1, s2, k]
#   ---------------------------------------------------------
    hists, X_hists, bines = np.empty([len(C), bins]), np.empty([len(C), bins]), np.empty([len(C), bins+1])
    for i in range(len(C)):
        hists[i], bines[i] = np.histogram(C[i], bins = bins)
        X_hists[i] = [0.5 * (bines[i][j] + bines[i][j+1]) for j in range(bins)]
        
    fig, axes = plt.subplots(2,len(C), figsize=(13, 8), squeeze=True)
#   ---------------------------------------------------------
    if len(C) == 3:
        titles = ["Clustering coefficient (semi-log)", "Shortest-path length (semi-log)", "Node degree (semi-log)",
                  "Clustering coefficient (log-log)", "Shortest-path length (log-log)", "Node degree (log-log)"]
    else:
        titles = ["Clustering coefficient (semi-log)", "Shortest-path length (semi-log)",
                  "Degree [in] (semi-log)", "Degree [out] (semi-log)", 
                  "Clustering coefficient (log-log)", "Shortest-path length (log-log)",
                  "Degree [in] (log-log)", "Degree [out] (log-log)"]
#   ---------------------------------------------------------
    for i in range(2*len(C)):
        axes[i//len(C),i%len(C)].scatter(X_hists[i%len(C)], hists[i%len(C)], s=9, c="red", marker="o")
        axes[i//len(C),i%len(C)].set_xscale('log')
        if i > len(C) - 1:
            axes[i//len(C),i%len(C)].set_yscale('log')
        axes[i//len(C),i%len(C)].grid()
        axes[i//len(C),i%len(C)].title.set_text(titles[i])
#   ---------------------------------------------------------

    fig.tight_layout()
    plt.show()
    
# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------