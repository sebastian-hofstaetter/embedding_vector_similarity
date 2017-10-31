__author__='navid'

import pdb
from scipy.spatial import distance

class RelatedTerms:

    def filter_relatedterms(self, term_relterms_withweight, filter_method, filter_value):
        #filter expanding terms
        if filter_method=='threshold':
            _term_relterms_withweight={}
            for term in list(term_relterms_withweight.keys()):
                _term_relterms_withweight[term]=[[],[]]
                for value_i, value in enumerate(term_relterms_withweight[term][1]):
                    if value>=filter_value:
                        _term_relterms_withweight[term][0].append(term_relterms_withweight[term][0][value_i])
                        _term_relterms_withweight[term][1].append(value)
            term_relterms_withweight=_term_relterms_withweight
        elif filter_method=='count':
            for term in list(term_relterms_withweight.keys()):
                list_simvals, list_words = list(zip(*sorted(zip(term_relterms_withweight[term][1],
                                                           term_relterms_withweight[term][0]), reverse=True)))
                term_relterms_withweight[term]=(list_words[0:filter_value+1],
                                                        list_simvals[0:filter_value+1]) #+1 is for word itself
        elif filter_method=='countall':
            _all_values=[]
            for term in list(term_relterms_withweight.keys()):
                _all_values.extend(term_relterms_withweight[term][1])
            if filter_value<len(_all_values):
                _threshold_filter_value=_all_values[filter_value]
            else:
                _threshold_filter_value=_all_values[-1]

            _term_relterms_withweight={}
            for term in list(term_relterms_withweight.keys()):
                _term_relterms_withweight[term]=[[],[]]
                for value_i, value in enumerate(term_relterms_withweight[term][1]):
                    if value>=_threshold_filter_value:
                        _term_relterms_withweight[term][0].append(term_relterms_withweight[term][0][value_i])
                        _term_relterms_withweight[term][1].append(value)
            term_relterms_withweight=_term_relterms_withweight
        elif filter_method=='all':
            if term_relterms_withweight!=None:
                for term in list(term_relterms_withweight.keys()):
                    term_relterms_withweight[term]=[[],[]]

        return term_relterms_withweight