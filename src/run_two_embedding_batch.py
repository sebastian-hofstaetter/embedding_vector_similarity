import argparse
import json

import numpy
import os
from gensim.models import KeyedVectors
from gensim.utils import to_unicode

from embedding_vector_similarity.src.embeddings.embeddingmodel import GensimEmbeddingModel
from embedding_vector_similarity.src.post_filtering.postfilters import PostFilters

parser = argparse.ArgumentParser()

parser.add_argument('--vec-1', action='store', dest='vec1_path',
                    help='word2vec model 1', required=True)

parser.add_argument('--vec-2', action='store', dest='vec2_path',
                    help='word2vec model 2', required=True)

parser.add_argument('--min-s', action='store', dest='sim_min', type=float,
                    help='sim_min', required=True)
parser.add_argument('--max-s', action='store', dest='sim_max', type=float,
                    help='sim_max', required=True)
parser.add_argument('--step-s', action='store', dest='sim_steps', type=float,
                    help='sim_steps', required=True)

parser.add_argument('--terms', action='store', dest='terms_path',
                    help='query term list', required=True)

parser.add_argument('--out', action='store', dest='out_path',
                    help='output folder', required=True)

parser.add_argument('--name', action='store', dest='name',
                    help='run nam', required=True)

args = parser.parse_args()

#
# query terms
#
query_term_list = open(args.terms_path, 'r').read().splitlines()
for i in range(len(query_term_list)):
    query_term_list[i] = to_unicode(query_term_list[i])
query_term_list = list(set(query_term_list))  # remove duplicates

#
# gensim models
#
model1 = KeyedVectors.load_word2vec_format(args.vec1_path, binary=False)
model2 = KeyedVectors.load_word2vec_format(args.vec2_path, binary=False)

embedding1 = GensimEmbeddingModel()
embedding1.load_model_in_memory(model1, '')

embedding2 = GensimEmbeddingModel()
embedding2.load_model_in_memory(model2, '')

print(args.vec1_path)
print(args.vec2_path)

#print("embedding equals: ", numpy.array_equal(embedding1.get_vectors_all(),embedding2.get_vectors_all()))

#
# per threshold - compute similar
#

vec_1_weight = 0.5

for vec_1_weight in numpy.arange(0.2, 0.801, 0.1):

    vec_2_weight = 1 - vec_1_weight

    for filter_value in numpy.arange(args.sim_min, args.sim_max + 0.0001, args.sim_steps):

        term_relterms_withweight1 = embedding1.search_neighbors_cosine(query_term_list, 400)
        term_relterms_withweight2 = embedding2.search_neighbors_cosine(query_term_list, 400)

        result = {}

        for term in term_relterms_withweight1:
            result[term] = [[], []]

            d1 = term_relterms_withweight1[term]
            d2 = term_relterms_withweight2[term]

            if d1[0] == d2[0] and d1[1] == d2[1]:
                print("same result @",term , filter_value,"for:", args.vec2_path)

            d1_term_indices = {}

            for i, sim_term in enumerate(d1[0]):
                result[term][0].append(sim_term)
                result[term][1].append(d1[1][i] * vec_1_weight)

                d1_term_indices[term] = i

            for i, sim_term in enumerate(d2[0]):
                # if term already in there from first embedding only add weight
                if sim_term in d1_term_indices:
                    result[term][1][d1_term_indices[sim_term]] += d2[1][i] * vec_2_weight
                else:
                    result[term][0].append(sim_term)
                    result[term][1].append(d2[1][i] * vec_2_weight)

        result = PostFilters.filter_embedding_threshold(result, filter_value)

        open(os.path.join(args.out_path,
                          args.name + "w" +
                          str("%0.3f" % vec_1_weight).replace('.', '-') + "_" +
                          str("%0.3f" % filter_value).replace('.', '-') + ".txt"),
             'w') \
            .write(json.dumps(result))
