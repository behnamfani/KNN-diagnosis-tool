import spacy
spacy.prefer_gpu()
import networkx as nx
import math
from icecream import ic
import operator
import pandas as pd


# # ------------------ Extractor --------------------
class Extractor:

    def extract(self, text:str, POS_KEPT:list)->list:
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 10000000 
        doc = nlp(text)
	
        lemma_graph = nx.Graph()
        seen_lemma = {}

        for sent in doc.sents:        
            self.link_sentence(doc, sent, lemma_graph, seen_lemma, POS_KEPT)
        
        
        labels = {}
        keys = list(seen_lemma.keys())

        for i in range(len(seen_lemma)):
            labels[i] = keys[i][0].lower()
            
        # import matplotlib.pyplot as plt

        # fig = plt.figure(figsize=(9, 9))
        # pos = nx.spring_layout(lemma_graph)

        # nx.draw(lemma_graph, pos=pos, with_labels=False, font_weight="bold")
        # nx.draw_networkx_labels(lemma_graph, pos, labels)

        ranks = nx.pagerank(lemma_graph)
        # for node_id, rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
        #     ic(node_id, rank, labels[node_id])


        phrases = {}
        counts = {}

        for chunk in doc.noun_chunks:
            self.collect_phrases(doc, seen_lemma, ranks, chunk, phrases, counts)

        min_phrases = {}

        for compound_key, rank_tuples in phrases.items():
            l = list(rank_tuples)
            l.sort(key=operator.itemgetter(1), reverse=True)
            
            phrase, rank = l[0]
            count = counts[compound_key]
            
            min_phrases[phrase] = (rank, count)

        return sorted(min_phrases.items(), key=lambda x: x[1][0], reverse=True)
            
            
 
    def increment_edge (self, graph, node0, node1):
    
        if graph.has_edge(node0, node1):
            graph[node0][node1]["weight"] += 1.0
        else:
            graph.add_edge(node0, node1, weight=1.0)


  
    def link_sentence (self, doc, sent, lemma_graph, seen_lemma, POS_KEPT):
        visited_tokens = []
        visited_nodes = []

        for i in range(sent.start, sent.end):
            token = doc[i]

            if token.pos_ in POS_KEPT:
                key = (token.lemma_, token.pos_)

                if key not in seen_lemma:
                    seen_lemma[key] = set([token.i])
                else:
                    seen_lemma[key].add(token.i)

                node_id = list(seen_lemma.keys()).index(key)
                
                if not node_id in lemma_graph:
                    lemma_graph.add_node(node_id)

                # ic(visited_tokens, visited_nodes)
                # ic(list(range(len(visited_tokens) - 1, -1, -1)))
                
                for prev_token in range(len(visited_tokens) - 1, -1, -1):
                    # ic(prev_token, (token.i - visited_tokens[prev_token]))
                    
                    if (token.i - visited_tokens[prev_token]) <= 3:
                        self.increment_edge(lemma_graph, node_id, visited_nodes[prev_token])
                    else:
                        break

                # ic(token.i, token.text, token.lemma_, token.pos_, visited_tokens, visited_nodes)

                visited_tokens.append(token.i)
                visited_nodes.append(node_id)


    
    def collect_phrases (self, doc, seen_lemma, ranks, chunk, phrases, counts):
        chunk_len = chunk.end - chunk.start
        sq_sum_rank = 0.0
        non_lemma = 0
        compound_key = set([])

        for i in range(chunk.start, chunk.end):
            token = doc[i]
            key = (token.lemma_, token.pos_)
            
            if key in seen_lemma:
                node_id = list(seen_lemma.keys()).index(key)
                rank = ranks[node_id]
                sq_sum_rank += rank
                compound_key.add(key)
            
                # ic(token.lemma_, token.pos_, node_id, rank)
            else:
                non_lemma += 1
        
        # Discount the ranks using a point estimate based on the number of non-lemma tokens within a phrase
        non_lemma_discount = chunk_len / (chunk_len + (2.0 * non_lemma) + 1.0)

        # Normalize the contributions of all the tokens using RMS
        phrase_rank = math.sqrt(sq_sum_rank / (chunk_len + non_lemma))
        phrase_rank *= non_lemma_discount

        # Remove spurious punctuation
        phrase = chunk.text.lower().replace("'", "")

        # Create a unique key for the the phrase based on its lemma components
        compound_key = tuple(sorted(list(compound_key)))
        
        if not compound_key in phrases:
            phrases[compound_key] = set([ (phrase, phrase_rank) ])
            counts[compound_key] = 1
        else:
            phrases[compound_key].add( (phrase, phrase_rank) )
            counts[compound_key] += 1

        # ic(phrase_rank, chunk.text, chunk.start, chunk.end, chunk_len, counts[compound_key])


# ------------------ Main --------------------
# Read Data
# text = "The UEFA Champions League (abbreviated as UCL, or sometimes, UEFA CL) is an annual club association football competition organised by the Union of European Football Associations (UEFA) and contested by top-division European clubs, deciding the competition winners through a round robin group stage to qualify for a double-legged knockout format, and a single leg final. It is the most-watched club competition in the world and the third most-watched football competition overall, behind only the UEFA European Championship and the FIFA World Cup. It is one of the most prestigious football tournaments in the world and the most prestigious club competition in European football, played by the national league champions (and, for some nations, one or more runners-up) of their national associations."
text = ''
with open(f'/home/IAIS/bfanitabas/Project/Dataset/AmazonCat-13K.raw/Sample_Test.txt', 'r') as f:
	text = f.read().replace('\n', ' ')
f.close()
# Extract keyphrases
extractor = Extractor()
POS_KEPT = ["ADJ", "NOUN", "PROPN", "VERB"]
result = extractor.extract(text, POS_KEPT)
# Save results in a dataframe
df = pd.DataFrame(result, columns=['Phrase', 'Mix'])
df[['Score', 'Count']] = pd.DataFrame(df['Mix'].tolist())
df.drop(columns=['Mix'], inplace=True)
# print(df)
df.to_csv('/home/IAIS/bfanitabas/Project/Results/keyphrases.csv', index=False)
