from flask import Flask, jsonify, request
import logging

import json
from . import parameters as parameters
from .embeddings.embeddingmodel import GensimEmbeddingModel as embedmodel  # TextGensimEmbeddingModel
from .post_filtering.postfilters import *

app = Flask(__name__)
we_model = None
default_similarity_method = None
default_vector_method = None
default_filter_method = None
default_filter_value = None


@app.route('/')
def index():
    return "Hello, World!"


### Expected POST data
### terms : a list of query terms
### similarity_method : 'cos' (optional)
### vector_method : 'dense'/'expsg'/'rexpsg'/'prexpsg'  (optional)
### filter_method : 'threshold','count'  (optional)
### filter_method : a float value  (optional)
@app.route('/api/v1.0/post_filtering', methods=['GET', 'POST'])
def get_tasks():
    data = request.data
    # print str(data)
    dataDict = json.loads(data)
    # print str(dataDict)

    terms = dataDict['terms']

    print('---')
    print(str(terms))

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

    return jsonify(term_relterms_withweight)


if __name__ == '__main__':
    # parameters
    gensim_w2v_path = parameters.params['gensim_w2v_path']
    port = int(parameters.params['port'])

    # initialization
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    we_model = embedmodel()
    we_model.load_model_txt(gensim_w2v_path, '')

    default_similarity_method = parameters.params['default_similarity_method']
    default_vector_method = parameters.params['default_vector_method']
    default_filter_method = parameters.params['default_filter_method']
    default_filter_value = parameters.params['default_filter_value']

    app.run(debug=True, use_reloader=False, port=port)
