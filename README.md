# Word embedding - vector similarity

**Based on: https://github.com/neds/similarityAPI**
 
Changed to Python 3, added more models and refactored code to be usable from other Python code. 

REST api interface for term similarity based on various models

# Getting similar terms from REST API

## Running the API server
(1) Copy parameters.py.template to parameters.py and setup the parameters. The essential parameter is gensim_w2v_path, the path to the folder of the gensim word embedding model
(2) run runrestapi.py

## Fecthing similar terms
### Quick start:
Example:
curl -H "Content-Type: application/json" -X POST -d '{"terms":["book","librari"]}' http://127.0.0.1:5000/api/v1.0/relatedterms

### Detailed POST parameter
terms : a list of query terms
similarity_method : 'cos' (optional)
vector_method : 'we'/'weexpsg'/'werexpsg'/'weprexpsg'  (optional)
filter_method : 'threshold','count'  (optional)
filter_method : a float value  (optional)

Example:
curl -H "Content-Type: application/json" -X POST -d '{"terms":["book","librari"],"vector_method":"we","similarity_method":"cos","filter_method":"threshold","filter_value":"0.7"}' http://127.0.0.1:5000/api/v1.0/relatedterms
