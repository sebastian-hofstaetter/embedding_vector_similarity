'''
Created on 06.01.2015

@author: navid
'''
import logging
import sys
import pdb

from .embeddings.embeddingmodel import GensimEmbeddingModel
from . import parameters as parameters
from .relatedterms.relatedtermsembedding import RelatedTermsEmbedding

#parameters
gensim_w2v_path=parameters.params['gensim_w2v_path']
default_similarity_method=parameters.params['default_similarity_method']
default_vector_method=parameters.params['default_vector_method']
default_filter_method='countall'#parameters.params['default_filter_method']
default_filter_value=10#int(parameters.params['default_filter_value'])


port=int(parameters.params['port'])

#initialization
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

we_model=GensimEmbeddingModel(logger)
we_model.load_model(gensim_w2v_path, '')



relatedTermsEmbedding=RelatedTermsEmbedding()
term_relterms_withweight=relatedTermsEmbedding.get_relatedterms (we_model=we_model,
                                                                 similarityrepo=None,
                                                                 terms=['book', 'librari'],
                                                                 similarity_method=default_similarity_method,
                                                                 vector_method=default_vector_method)
term_relterms_withweight=relatedTermsEmbedding.filter_relatedterms(term_relterms_withweight=term_relterms_withweight,
                                                                   filter_method=default_filter_method,
                                                                   filter_value=default_filter_value)

for term in list(term_relterms_withweight.keys()):
    if len(term_relterms_withweight[term][0])==0:
        continue
    print(term+':'+str([(term_relterms_withweight[term][0][i], term_relterms_withweight[term][1][i]) for i in range(0,len(term_relterms_withweight[term][0]))]))

