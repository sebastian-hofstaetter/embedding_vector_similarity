from flask import Flask, jsonify, request
import logging

import json
from . import parameters as parameters
from .embeddings.embeddingmodel import GensimEmbeddingModel as embedmodel #TextGensimEmbeddingModel
from .relatedterms.relatedtermsembedding import RelatedTermsEmbedding

app = Flask(__name__)
we_model=None
relatedTermsEmbedding=None
default_similarity_method=None
default_vector_method=None
default_filter_method=None
default_filter_value=None


@app.route('/')
def index():
    return "Hello, World!"

### Expected POST data
### terms : a list of query terms
### similarity_method : 'cos' (optional)
### vector_method : 'dense'/'expsg'/'rexpsg'/'prexpsg'  (optional)
### filter_method : 'threshold','count'  (optional)
### filter_method : a float value  (optional)
@app.route('/api/v1.0/relatedterms', methods=['GET', 'POST'])
def get_tasks():
    data = request.data
    #print str(data)
    dataDict = json.loads(data)
    #print str(dataDict)

    terms=dataDict['terms']

    print('---')
    print(str(terms))

    similarity_method=(dataDict['similarity_method'] if 'similarity_method' in dataDict else default_similarity_method)
    vector_method=(dataDict['vector_method'] if 'vector_method' in dataDict else default_vector_method)
    filter_method=(dataDict['filter_method'] if 'filter_method' in dataDict else default_filter_method)
    filter_value=float(dataDict['filter_value'] if 'filter_value' in dataDict else default_filter_value)

    term_relterms_withweight=relatedTermsEmbedding.get_relatedterms (we_model=we_model,
                                                                     similarityrepo=None,
                                                                     terms=terms,
                                                                     similarity_method=similarity_method,
                                                                     vector_method=vector_method)

    term_relterms_withweight=relatedTermsEmbedding.filter_relatedterms(term_relterms_withweight=term_relterms_withweight,
                                                                       filter_method=filter_method,
                                                                       filter_value=filter_value)

    #for term,related in term_relterms_withweight.items():
    #    print '\n',term
    #    for i in range(len(related[0])):
    #        print '\t',related[0][i],'   ',related[1][i]

    return jsonify(term_relterms_withweight)

if __name__ == '__main__':

    #parameters
    gensim_w2v_path=parameters.params['gensim_w2v_path']
    port=int(parameters.params['port'])

    #initialization
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    we_model=embedmodel()
    we_model.load_model_txt(gensim_w2v_path, '')

    default_similarity_method=parameters.params['default_similarity_method']
    default_vector_method=parameters.params['default_vector_method']
    default_filter_method=parameters.params['default_filter_method']
    default_filter_value=parameters.params['default_filter_value']

    relatedTermsEmbedding=RelatedTermsEmbedding()

    app.run(debug=True, use_reloader=False, port=port)

