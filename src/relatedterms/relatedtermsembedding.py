__author__='navid'

from scipy.spatial import distance
from .relatedterms import RelatedTerms

class RelatedTermsEmbedding(RelatedTerms):

    def __init__(self):
        pass

    def get_wordembedding_vector(self, we_model, term, vector_method):
        if vector_method=='we':
            term_vector=we_model.get_vector(term)
        elif vector_method=='weexp':
            term_vector=we_model.getexp_vector(term)
        elif vector_method=='werexp':
            term_vector=we_model.getrexp_vector(term)
        elif vector_method=='weprexp':
            term_vector=we_model.getprexp_vector(term)
        elif vector_method=='wesppmicds':
            term_vector=we_model.getsppmicds_vector(term)
        else:
            raise Exception('wordembedding_vector vector_method unknown!')

        return term_vector

    def get_relatedterms(self, we_model, similarityrepo, terms, similarity_method, vector_method):

        term_relterms_withweight={}

        #find neighbors
        for term in terms:
            #check in similarity repo
            relterms_withweight=None

            if similarityrepo!=None:
                relterms_withweight=similarityrepo.get_data(terms, vector_method, term)

            if relterms_withweight==None:
                relterms_withweight=[[],[]]
                term_vector=self.get_wordembedding_vector(we_model, term, 'we')
                term_simvector=self.get_wordembedding_vector(we_model, term, vector_method)
                if (term_vector is not None) and (len(term_vector)!=0) and\
                   (term_simvector is not None) and (len(term_simvector)!=0):
                    _, _wordsList=we_model.search_neighbors([term_vector], 200)

                    for expandterm in _wordsList[0]:
                        #remove stop words and terms with non-eng chars
                        if (expandterm==expandterm.encode('ascii', 'ignore')):
                            #calculate similarity
                            if 'cos' == similarity_method:
                                if expandterm!=term:
                                    expandterm_simvector=self.get_wordembedding_vector(we_model, expandterm, vector_method)
                                    if (expandterm_simvector is not None) and (len(expandterm_simvector)!=0):
                                        _sim=1-distance.cosine(term_simvector, expandterm_simvector)
                                        relterms_withweight[0].append(expandterm.encode('ascii', 'ignore'))
                                        relterms_withweight[1].append(_sim)
                                else:
                                    relterms_withweight[0].append(expandterm.encode('ascii', 'ignore'))
                                    relterms_withweight[1].append(1.0)
                            else:
                                raise Exception('similarity_method unknown!')
                else:
                    relterms_withweight[0].append(term)
                    relterms_withweight[1].append(1.0)

                if similarityrepo!=None:
                    similarityrepo.set_data(terms, vector_method, term, relterms_withweight)


            term_relterms_withweight[term]=relterms_withweight

        return term_relterms_withweight
