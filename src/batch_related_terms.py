import logging
from typing import List

from gensim.models import KeyedVectors

from .embeddings.embeddingmodel import GensimEmbeddingModel
from .post_filtering.postfilters import *


def get_batch_result_from_kv(vectors: KeyedVectors, terms: List[str], filter_value=0.7, filter_method = 'threshold'):

    embedding = GensimEmbeddingModel()
    embedding.load_model_in_memory(vectors, '')

    # for term in terms:
    #    print(term)
    #    print(embedding.search_neighbors([embedding.get_vector(term)], 5))
    #    break

    term_relterms_withweight = embedding.search_neighbors_cosine(terms, 200)
    # for term, related in term_relterms_withweight.items():
    #    print('\n',term,len(related[0]))
    #    for i in range(len(related[0])):
    #        print('\t',related[0][i],'   ',related[1][i])
    #
    # print("----")

    if filter_method == "threshold":
        term_relterms_withweight = PostFilters.filter_embedding_threshold(term_relterms_withweight, filter_value)

        # for term, related in term_relterms_withweight.items():
    # print('\n',term)
    #    for i in range(len(related[0])):
    #        print('\t',related[0][i],'   ',related[1][i])

    return term_relterms_withweight



def get_batch_result_from_kv_lsi(vectors: KeyedVectors, terms: List[str], lsi_data, embedding_filter_value=0.7, lsi_filter_value=0.7):

    embedding = GensimEmbeddingModel()
    embedding.load_model_in_memory(vectors, '')

    term_relterms_withweight = embedding.search_neighbors_cosine(terms, 200)


    term_relterms_withweight = PostFilters.filter_embedding_threshold_lsi_threshold(term_relterms_withweight,
                                                                                    embedding,
                                                                                    embedding_filter_value,
                                                                                    lsi_data,
                                                                                    lsi_filter_value)

    return term_relterms_withweight
