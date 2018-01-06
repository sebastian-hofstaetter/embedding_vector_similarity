import argparse

from flask import Flask, request
import logging

import json
from embedding_vector_similarity.src.embeddings.embeddingmodel import GensimEmbeddingModel as embedmodel
from embedding_vector_similarity.src.post_filtering.postfilters import *

app = Flask(__name__)
we_model = None
default_filter_method = None
default_filter_value = None


@app.route('/')
def index():
    return "Hello, World!"


# get a list of similar terms and a list of weights per given term
# - returns json
#
# the default filter_method and filter_value can be overridden
#
@app.route('/api/v1.0/post_filtering', methods=['GET', 'POST'])
def get_tasks():

    dataDict = request.get_json()

    terms = dataDict['terms']

    #print('---')
    #print(str(terms))

    filter_method = (dataDict['filter_method'] if 'filter_method' in dataDict else default_filter_method)
    filter_value = float(dataDict['filter_value'] if 'filter_value' in dataDict else default_filter_value)

    term_relterms_withweight = we_model.search_neighbors_cosine(terms, 200)

    if filter_method == "threshold":
        term_relterms_withweight = PostFilters.filter_embedding_threshold(term_relterms_withweight, filter_value)
    if filter_method == "count":
        term_relterms_withweight = PostFilters.filter_count(term_relterms_withweight, filter_value)
    if filter_method == "countall":
        term_relterms_withweight = PostFilters.filter_countall(term_relterms_withweight, filter_value)

    # for term,related in term_relterms_withweight.items():
    #    print '\n',term
    #    for i in range(len(related[0])):
    #        print '\t',related[0][i],'   ',related[1][i]

    return json.dumps(term_relterms_withweight)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding', action='store', dest='w2v_path',type=str,
                        help='word2vec text file', required=True)

    parser.add_argument('--filter-type', action='store', dest='default_filter_type',type=str,
                        help='threshold/count/countall', required=True)

    parser.add_argument('--filter-value', action='store', dest='default_filter_value', type=float,
                        help='the value used for filtering f.e.: 0.7', required=True)

    parser.add_argument('--api-port', action='store', dest='api_port', type=int,
                        help='port used for the api', required=True)

    args = parser.parse_args()

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    we_model = embedmodel()
    we_model.load_model_txt(args.w2v_path, '')

    default_filter_method = args.default_filter_type
    default_filter_value = args.default_filter_value

    app.run(debug=True, use_reloader=False, port=args.api_port)
