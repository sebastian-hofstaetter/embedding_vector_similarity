'''
Created on 07.01.2015

@author: navid
'''

import pickle
import pdb
import sqlite3
import os
import numpy as np
from scipy.special import expit
from gensim.models import Word2Vec, KeyedVectors
from gensim import utils
from sklearn.neighbors import BallTree
from scipy.sparse import csr_matrix
from gensim.corpora import Dictionary


class EmbeddingModel:

    def load_model(self, folderpath, modelname):
        raise Exception('not implemented!')

    def get_modelname(self):
        raise Exception('not implemented!')

    def get_vector(self, word):
        raise Exception('not implemented!')

    def get_vectors_all(self):
        raise Exception('not implemented!')

    def search_neighbors(self, vectors, num_neighbors):
        raise Exception('not implemented!')


class BallTreeANN:
    
    def __init__(self):
        '''
        Constructor
        '''
        self.nbrs=None
    
    def build_model(self, dataset, leaf_size):
        self.nbrs = BallTree(dataset, leaf_size=leaf_size, metric='euclidean')
        return self.nbrs
    
    def build_store_model(self, dataset, path, leaf_size):
        self.build_model(dataset, leaf_size)
        self.store_model(path)

    def store_model(self, path):
        #pdb.set_trace()
        with open(path, 'wb') as output1:
            pickle.dump(self.nbrs, output1, protocol=0)# error ('i' format requires -2147483648 <= number <= 2147483647) with pickle.HIGHEST_PROTOCOL

    def load_model(self, path):
        with open(path, 'rb') as input1:
            self.nbrs=pickle.load(input1)
    
    def search_in_radious(self, vector, radious = 2):
        distances, indices = self.nbrs.query_radius(vector, r=radious, return_distance=True)
        return distances, indices
    
    def search_neighbors(self, vector, num_neighbors):
        distances, indices = self.nbrs.query(vector, k=num_neighbors)
        return distances, indices


class BallTreeANNEmbeddingModel(EmbeddingModel):

    def __init__(self, logger):
        '''
        Constructor
        '''
        self.logger=logger
        self.balltreeann=BallTreeANN()
        self.model_to_word=None
        self.word_to_model=None
        

    def build_model(self, vectors, words, leaf_size):
        self.balltreeann.build_model(vectors, leaf_size)

        self.model_to_word=words

        self.word_to_model={}
        for word_i, word in enumerate(words):
            self.word_to_model[word]=word_i

    def build_store_model(self, vectors, words, path, leaf_size):
        self.logger.debug( 'building model...')
        self.build_model(vectors, words, leaf_size)

        self.logger.debug( 'saving model...')
        self.logger.debug('saving model - word_to_model...')
        with open(path+'.word_to_model', 'wb') as output1:
            pickle.dump(self.word_to_model, output1, pickle.HIGHEST_PROTOCOL)


        self.logger.debug('saving model - model_to_word...')
        with open(path+'.model_to_word', 'wb') as output1:
            pickle.dump(self.model_to_word, output1, pickle.HIGHEST_PROTOCOL)

        self.logger.debug('saving model - treeball...')
        self.balltreeann.store_model(path)

    def load_model(self, folderpath, modelname):
        self.modelname=modelname
        self.folderpath=folderpath
        path=self.folderpath+'/model.idx'
        
        #print 'loading embeddedANN...'
        #print 'loading embeddedANN - word_to_model...'
        with open(path+'.word_to_model', 'rb') as input1:
            self.word_to_model=pickle.load(input1)


        #print 'loading embeddedANN - model_to_word...'
        with open(path+'.model_to_word', 'rb') as input1:
            self.model_to_word=pickle.load(input1)

        #print 'loading embeddedANN - treeball...'
        self.balltreeann.load_model(path)
        
        #loading vector dot sqlite
        self.dbpath=self.folderpath+'/vectordotproduct.db'
        if not os.path.isfile(self.dbpath):
            self.logger.info('creating empty db vectordotproduct in '+self.dbpath)
            dbconn=sqlite3.connect(self.dbpath)
            dbconn.text_factory = str
            dbconn.executescript("""
            CREATE TABLE DOTPRODUCT(word TEXT, vector BLOB, PRIMARY KEY(word ASC));
            """)
            dbconn.commit()
        
    def get_modelname(self):
        return self.modelname

    def get_vector(self, word):
        if word not in self.word_to_model:
            return None

        word_idx=self.word_to_model[word]
        vector=self.balltreeann.nbrs.get_arrays()[0][word_idx]
        return vector

    def get_vectors_all(self):
        vectors=self.balltreeann.nbrs.get_arrays()[0]
        return vectors


    def search_in_radious_words(self, words, radious = 2):
        vectors=[]
        for word in words:
            vector=self.get_vector(word)
            if vector!=None:
                vectors.append(vector)
            else:
                return None, None
        return self.search_in_radious(vectors, radious)

    def search_neighbors_words(self, words, num_neighbors):
        vectors=[]
        for word in words:
            vector=self.get_vector(word)
            if vector!=None:
                vectors.append(vector)
            else:
                return None, None
        return self.search_neighbors(vectors, num_neighbors)

    def search_in_radious(self, vectors, radious = 2):
        return self.balltreeann.search_in_radious(vectors, radious)


    def search_neighbors(self, vectors, num_neighbors):
        _distsList, _indicesList = self.balltreeann.search_neighbors(vectors, num_neighbors)
        _wordsList=[]
        for _indices in _indicesList:
            _words=[]
            for _model in _indices:
                _words.append(self.model_to_word[_model])
            _wordsList.append(_words)
        return _distsList, _wordsList
    
    def get_dotvectors(self, words):
        dbconn=sqlite3.connect(self.dbpath)
        dbconn.text_factory = str
        
        dotvectors=[]
        for word in words:
            cursor = dbconn.execute("SELECT vector FROM DOTPRODUCT WHERE word=?", [word])
            data = cursor.fetchone()
            if data==None:
                dotvector=self.save_dotvectors(dbconn, [word])[0]
            else:
                dotvector=np.fromstring(data[0], dtype=float)
            
            dotvectors.append(dotvector)
        return dotvectors

    def save_dotvectors(self, dbconn, words):
        vectors_all=self.get_vectors_all()

        dotvectors=[]
        for word_i, word in enumerate(words):
            self.logger.debug('calculating and saving dot vector of '+word)

            vector=self.get_vector(word)
            if vector is not None:
                dotvector=np.dot(vectors_all, vector)
                dbconn.execute("INSERT INTO DOTPRODUCT (word, vector) VALUES(?, ?)", [word, dotvector.tostring()])
            else:
                dotvector=[]
            dotvectors.append(dotvector)

        dbconn.commit()

        return dotvectors


class GensimEmbeddingModel(EmbeddingModel):

    def __init__(self, logger):
        '''
        Constructor
        '''
        self.logger=logger
        self.vocabcount_w2vvec=None
        self.matrixcoo=None
        self.matrixcoodict=None
        self.matrixcoo_colsum_vec=None
        self.matrixcoo_sum=None
        self.matrixcoo_colsum_cds_vec=None
        self.matrixcoo_colsum_cds_vec_sum=None
        self.ns=None
        self.E_ctx_vec=None


    def initialize_matrixcoo(self, statspath, cds=0.75, ns=10.):
        self.logger.debug("Initializing and loading 'matrixcoo'")

        self.matrixcoodict=Dictionary.load(statspath+'/gensim_dictionary')
        loader = np.load(statspath+'/windowcoomatrix.npz')
        self.matrixcoo=csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        self.matrixcoo_colsum_vec=np.array(list(map(float, np.array(self.matrixcoo.sum(axis=0))[0])))
        self.matrixcoo_colsum_vec[self.matrixcoo_colsum_vec==0]=1.0
        self.matrixcoo_colsum_cds_vec=self.matrixcoo_colsum_vec**cds
        self.matrixcoo_colsum_cds_vec_sum=np.sum(self.matrixcoo_colsum_cds_vec)
        self.matrixcoo_sum=float(self.matrixcoo.sum())
        self.ns=ns

    def initialize_vocabcount_w2vvec(self):
        self.logger.debug("Initializing 'vocabcount_w2vvec'")

        _vocabcounttuple=[]
        for _wrd in list(self.model.wv.vocab.keys()):
            _vocabcounttuple.append((self.model.wv.vocab[_wrd].index, self.model.wv.vocab[_wrd].count))
        _vocabcounttuple.sort(key=lambda x:x[0])
        self.vocabcount_w2vvec=np.array([x[1] for x in _vocabcounttuple])

    def load_E_ctx_vec(self):
        self.logger.debug("Loading 'E_ctx_vec'")

        self.E_ctx_vec=np.array(pickle.load(open(self.folderpath+'/E_ctx_vec.pkl')))


    def createsave_E_ctx_vec(self):
        self.logger.debug("Creating 'E_ctx_vec'")

        self.E_ctx_vec=[]
        for _ctx_i in range(len(list(self.model.wv.vocab.keys()))):
            _ctx_vec=self.model.syn1neg[_ctx_i]

            ctx_simallwrd=expit(np.dot(self.model.wv.syn0, _ctx_vec))
            E_ctx=np.average(ctx_simallwrd, weights=self.vocabcount_w2vvec)

            self.E_ctx_vec.append(E_ctx)

        pickle.dump(self.E_ctx_vec, open(self.folderpath+'/E_ctx_vec.pkl','w'))

    def load_model(self, folderpath, modelname):
        self.modelname=modelname
        self.folderpath=folderpath
        path=self.folderpath+'/skipgram.model'

        self.model=Word2Vec.load(path)
        print('word vector length: ',len(self.model.wv[self.model.wv.index2word[0]]))
        print('word vocab count: ',len(self.model.wv.index2word))

    def get_modelname(self):
        return self.modelname

    def get_vector(self, word):
        word=utils.to_unicode(word)
        if word not in self.model:
            return None

        return self.model[word]

    def getexp_vector(self, word):
        _vec=self.get_vector(word)
        if _vec is not None:
            _expvec=expit(np.dot(self.model.syn1neg, _vec))
        else:
            return None

        return _expvec

    def getrexp_vector(self, word, cds=0.75):
        if self.vocabcount_w2vvec is None:
            raise Exception('vocabcount_w2vvec is not initialized')

        _expvec=self.getexp_vector(word)
        if _expvec is not None:
            E_wrd=np.average(_expvec, weights=self.vocabcount_w2vvec**cds)
            _rexpvec=_expvec-self.E_ctx_vec-E_wrd
        else:
            return None

        return _rexpvec

    def getprexp_vector(self, word, cds=0.75):
        if self.vocabcount_w2vvec is None:
            raise Exception('vocabcount_w2vvec is not initialized')

        _rexpvec=self.getrexp_vector(word, cds)
        if _rexpvec is not None:
            _prexpvec=_rexpvec
            _prexpvec[_prexpvec<0.0]=0.0
        else:
            return None

        return _prexpvec

    def getcoo_vector(self, word):
        if self.matrixcoo is None:
            raise Exception('matrixcoo is not initialized')
        if self.matrixcoodict is None:
            raise Exception('matrixcoodict is not initialized')

        if word not in self.matrixcoodict.token2id:
            self.logger.debug("Word '"+word+"' not found in matrixcoodict")
            return None

        _idx=self.matrixcoodict.token2id[word]
        _vec=np.array(list(map(float, self.matrixcoo.getrow(_idx).toarray()[0])))
        return _vec

    def getpmi_vector(self, word):
        if self.matrixcoo_colsum_vec is None:
            raise Exception('matrixcoo_colsum_vec is not initialized')
        if self.matrixcoo_sum is None:
            raise Exception('matrixcoo_sum is not initialized')

        _coo_vec=self.getcoo_vector(word)
        if _coo_vec is None:
            return None
        _vec=np.log((_coo_vec*self.matrixcoo_sum)/(np.sum(_coo_vec)*self.matrixcoo_colsum_vec))
        return _vec

    def getpmicds_vector(self, word):
        if self.matrixcoo_colsum_cds_vec is None:
            raise Exception('matrixcoo_colsum_cds_vec is not initialized')
        if self.matrixcoo_colsum_cds_vec_sum is None:
            raise Exception('matrixcoo_colsum_cds_vec_sum is not initialized')

        _coo_vec=self.getcoo_vector(word)
        if _coo_vec is None:
            return None
        _vec=np.log((_coo_vec)/(np.sum(_coo_vec)*(self.matrixcoo_colsum_cds_vec/self.matrixcoo_colsum_cds_vec_sum)))
        return _vec

    def getsppmicds_vector(self, word):
        if self.ns is None:
            raise Exception('ns is not initialized')

        _pmicds_vec=self.getpmicds_vector(word)
        if _pmicds_vec is None:
            return None
        _sppmicds_vec=_pmicds_vec-np.log(self.ns)
        _sppmicds_vec[_sppmicds_vec<0]=0.0

        return _sppmicds_vec

    def get_vectors_all(self):
        return self.model.wv.syn0

    def search_neighbors(self, vectors, num_neighbors):
        _distsList=[]
        _wordsList=[]

        for vector in vectors:
            tuples=self.model.wv.similar_by_vector(vector, topn=200)
            _distsList.append([x[1] for x in tuples])
            _wordsList.append([x[0] for x in tuples])

        return _distsList, _wordsList

class TextGensimEmbeddingModel(EmbeddingModel):

    def __init__(self, logger):
        '''
        Constructor
        '''
        self.logger=logger


    def load_model(self, folderpath, modelname):
        self.modelname=modelname
        self.folderpath=folderpath
        self.word_vectors = KeyedVectors.load_word2vec_format(folderpath, binary=False)  # C text format

    def get_modelname(self):
        return self.modelname

    def get_vector(self, word):
        word=utils.to_unicode(word)
        if word not in self.word_vectors:
            return None

        return self.word_vectors[word]

    def get_vectors_all(self):
        return self.word_vectors.syn0

    def search_neighbors(self, vectors, num_neighbors):
        _distsList=[]
        _wordsList=[]

        for vector in vectors:
            tuples=self.word_vectors.similar_by_vector(vector, topn=200)
            _distsList.append([x[1] for x in tuples])
            _wordsList.append([x[0] for x in tuples])

        return _distsList, _wordsList