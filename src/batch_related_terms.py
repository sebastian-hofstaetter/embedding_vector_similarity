import logging
from typing import List

from gensim.models import KeyedVectors

from .embeddings.embeddingmodel import InMemoryGensimEmbeddingModel as embedmodel  # TextGensimEmbeddingModel
from .relatedterms.relatedtermsembedding import RelatedTermsEmbedding


def get_batch_result_from_kv(vectors:KeyedVectors, terms: List[str]):
    we_model = embedmodel(logging.getLogger())
    we_model.load_model(vectors, '')

    related_terms_embedding = RelatedTermsEmbedding()

    similarity_method = 'cos'
    vector_method = 'we'
    filter_method = 'threshold'
    filter_value = 0.7

    #for term in terms:
    #    print(term)
    #    print(we_model.search_neighbors([we_model.get_vector(term)], 5))
    #    break

    term_relterms_withweight = related_terms_embedding.get_relatedterms(we_model=we_model,
                                                                        similarityrepo=None,
                                                                        terms=terms,
                                                                        similarity_method=similarity_method,
                                                                        vector_method=vector_method)

    #for term, related in term_relterms_withweight.items():
    #    print('\n',term,len(related[0]))
    #    for i in range(len(related[0])):
    #        print('\t',related[0][i],'   ',related[1][i])
    #
    #print("----")

    term_relterms_withweight = related_terms_embedding.filter_relatedterms(
        term_relterms_withweight=term_relterms_withweight,
        filter_method=filter_method,
        filter_value=filter_value)

    #for term, related in term_relterms_withweight.items():
    #    print('\n',term)
    #    for i in range(len(related[0])):
    #        print('\t',related[0][i],'   ',related[1][i])

    return term_relterms_withweight
