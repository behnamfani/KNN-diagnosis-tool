KNN-diagnosis-tool is used to distinguish several embeddings of the same corpus that were acquired by various language models. It employs the following features:
* Number of neighbors in the whole levels of each node
* Mean of cos-similarities in the whole levels of each node
* Number of categories in the whole levels of each node
on the (Recursive) nearest neighbor graphs and their communities where each document is a node and the edges are drawn based on how related the nodes are to one another.
In this project, we attempt to combine this tool with additional evaluation techniques, such as key-phrase analysis and perplexity, to assess various language models.
Our hypothesis is when the perplexity value increases, the number of common key phrases
will decrease, and conversely, when the perplexity value decreases, the number of common
key phrases will increase.

<p align="center" width="100%">
    <img width="50%" src="/Experiment%20with%20key-phrases/RKNN.png"> 
</p>

Example of a node in Recursive KNN with K=4. Each node is a document with the number as an identifier. Some Recursive KNN graph nodes are repeated, 
as indicated by their colors. Each node has a set of key-phrases. We expect a decent model to contain a few unique nodes in the neighborhood of each 
node and a high number of common keyphrases
