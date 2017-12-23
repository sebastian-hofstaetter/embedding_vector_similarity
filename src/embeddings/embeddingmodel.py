from gensim import utils
from gensim.models import KeyedVectors


#
# Wraps a gensim word2vec model with convenient methods to call
#
class GensimEmbeddingModel:

    word_vectors = None
    model_name = ""

    def load_model_txt(self, folderpath, modelname):
        self.model_name=modelname
        self.word_vectors = KeyedVectors.load_word2vec_format(folderpath, binary=False)  # C text format

    def load_model_in_memory(self, vectors, modelname):
        self.model_name = modelname
        self.word_vectors = vectors

    def get_modelname(self):
        return self.model_name

    def get_vector(self, word):
        word=utils.to_unicode(word)
        if word not in self.word_vectors:
            return None

        return self.word_vectors[word]

    def get_vectors_all(self):
        return self.word_vectors.syn0

    def search_neighbors(self, vectors, num_neighbors=200):
        _distsList = []
        _wordsList = []

        for vector in vectors:
            tuples = self.word_vectors.similar_by_vector(vector, topn=num_neighbors)
            _distsList.append([x[1] for x in tuples])
            _wordsList.append([x[0] for x in tuples])

        return _distsList, _wordsList