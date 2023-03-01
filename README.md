# KNN-diagnosis-tool
A tool to distinguish different embeddings of a same corpus using different features

Neural natural language models are drawing more and more attention every day as they aim to give computers the ability to interpret human words and documents. Yet as there are more and more models available, it gets more challenging to select the optimal one for a certain task. This is a developing project where various features from datasets are derived and different  vector embeddings would be compared with eachother based on this features. These features and their correlations with the performance of each embedding would make it easier to compare multiple embeddings of the same corpus.

#### Input format (like the files in the Input folder):
* A .txt file where each line represents a document (text).
* A .txt file where each line represents the labels of the corresponding document (If available). It is possible to do the experiments without labels or use the keywords of the documents as their labels.
* Various .txt files where each line represents the embeddings of the corresponding document. It is also possible to pass the pickle files of the embeddings where they have been stored in numpy.array format.

#### External files (optional):
It is possible to upload .csv files of the unique tf-idf values for labels (if you are giving the labels as well) and/or a dataframe in which element (i,j) is the hamming distance between document i and j based on the characteristic vectors of these documents w.r.t labels. You can also have the program calculate these dataframes for you.


### Features:

Based on the embeddings and a similarty metric (cosine-similarity or L2 Norm), the Nearest Neighbor (KNN) graphs of the embeddings are created. The Recursive KNN
graphs are then created by using the KNN method recursively and finding the nearest neighbors of the neighbors of the original node and continuing the same process. A parameter called nHops in this project defines the number of times that this process is repeated (including the first KNN). The features below are derived from the Recursive KNNs:
* Number of neighbors in the whole levels of each node
* Mean of cos-similarities in the whole levels of each node
* Number of categories in the whole levels of each node

Chosen subgraphs and dense clusters of the KNN graph are also investigated to discover more useful features. To detect communities, Networkx library is used. 
* networkx.greedy_modularity_communities(G, weight=similarity, resolution=1, cutoff=1, best_n=None) [Networkx](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html#rce363827c0a4-2)


### Result:
The tool is tested with a sample of [Amazon Cat-13k](http://manikvarma.org/downloads/XC/XMLRepository.html). Three different models were used to achieve the embeddings of the sample. Below is the result of the experiments on all the nodes of Recursive KNN graphs, the top 5% of the neighbors with the most and least neighbors in their neighborhood:

![](/Results/PlotsCos.png)
*The ability of each feature to distinguish the embeddings is illustrated in the histograms of the documents w.r.t the various features.*

![](/Results/CommunitiesFeatures.png)
*Scatter plots of the communities of different KNN graphs.*

<img src="/Results/Table.png" width="400" height="200">
The embeddings typically yield a varying number of communities. This insight helps us in our future work and scoring methods because it provides a more thorough understanding of model training.
