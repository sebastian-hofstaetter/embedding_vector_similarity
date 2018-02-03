import argparse
import json

import numpy
import os
from gensim.models import KeyedVectors
from gensim.utils import to_unicode

from Fusion_embedding.src.evaluation.check_pivot_word_neighbors import create_save_pivot
from Fusion_embedding.src.evaluation.vector_space_comparison import create_save_vector_space_comparison
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

parser.add_argument('--out-eval', action='store', dest='out_eval_path',
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

#
# eval
#

create_save_vector_space_comparison(base_vectors=embedding1.get_vectors_all(),
                                    experiment_vectors=embedding2.get_vectors_all(),
                                    filename=args.out_eval_path + "all_space_comparison.csv",
                                    name="two_retrofitted")

create_save_pivot(embedding1.word_vectors,
                  embedding2.word_vectors,
                  os.path.join(args.out_eval_path, 'all_pivot_queryterm_'".csv"),
                  "two_retrofitted",
                  query_term_list,
                  numpy.arange(args.sim_min, args.sim_max + 0.0001, args.sim_steps))

pivot_file = open(args.out_eval_path+"all_neighbors_combined.csv", 'w')
pivot_file.write(",v1_weight,v2_weight,neighbor_threshold,,count_total, count_avg\n")


#print("embedding equals: ", numpy.array_equal(embedding1.get_vectors_all(),embedding2.get_vectors_all()))

#
# per threshold - compute similar
#

vec_1_weight = 0.5

for vec_1_weight in numpy.arange(0.1, 0.901, 0.1):

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

            for i, sim_term in enumerate(d1[0]):

                sim_score_1 = d1[1][i]
                sim_score_2 = 0

                for t, sim_term_2 in enumerate(d2[0]):
                    if sim_term_2 == sim_term:
                        sim_score_2 = d2[1][i]
                        break

                result[term][0].append(sim_term)
                # arithmetic mean
                result[term][1].append((sim_score_1 * vec_1_weight + sim_score_2 * vec_2_weight)/(vec_1_weight + vec_2_weight))
                # harmonic mean
                #result[term][1].append(pow(
                #    (pow(sim_score_1,-1) * vec_1_weight + pow(sim_score_2, -1) * vec_2_weight)/(vec_1_weight + vec_2_weight),
                #    -1))

        result = PostFilters.filter_embedding_threshold(result, filter_value)

        neighbors = 0
        terms = 0
        sims = 0
        for term, related in result.items():
            terms +=1
            neighbors += len (related[0])
            sims += numpy.sum(related[1])

        #print('avg @',args.vec2_path,vec_1_weight,filter_value,"val: ",neighbors/terms)
        #print('sum @',args.vec2_path,vec_1_weight,filter_value,"val: ",sims)

        #",v1_weight,v2_weight,neighbor_threshold,,count_total, count_avg\n"
        pivot_file.write(os.path.basename(args.vec2_path) + ","+str(vec_1_weight)+ ","+str(vec_2_weight)+
                         "," + str(filter_value) + ",," + str(neighbors)+"," + str(neighbors/terms)+ "\n")

        open(os.path.join(args.out_path,
                          args.name + "w" +
                          str("%0.3f" % vec_1_weight).replace('.', '-') + "_" +
                          str("%0.3f" % filter_value).replace('.', '-') + ".txt"),
             'w') \
            .write(json.dumps(result))
