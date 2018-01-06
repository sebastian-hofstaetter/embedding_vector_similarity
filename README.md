# Word Embedding - Vector Similarity

**Based on: https://github.com/neds/similarityAPI**
 
Changed to Python 3, added more models and refactored code to be usable from other Python code. 

---

**REST API and library for term similarity based on various models**

# Getting similar terms

## (1) Running the API server

Call runrestapi.py with all needed parameters defined in runrestapi.py and make sure that PYTHONPATH is set to the outer folder of this main folder.
The embedding must be in word2vec text format - but with minimal code changes other formats that are supported by gensim can be used as well.


Once the API is running call: /api/v1.0/post_filtering 

````
curl -H "Content-Type: application/json" -X GET -d '{"terms":["book","librari"]}' http://127.0.0.1:5000/api/v1.0/post_filtering
````

## (2) Or: use as a library

Call one of the methods defined in batch_related_terms.py to call this api from python code. Here an already loaded gensim model is needed.
See the file for more information.