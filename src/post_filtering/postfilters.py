#
# Contains various post filter methods for result of embeddingmodel.search_neighbors_cosine
#
class PostFilters:

    @staticmethod
    def filter_embedding_threshold_lsi_threshold(term_relterms_withweight, embedding, embedding_filter_value, lsi_data, lsi_filter_value):
        _term_relterms_withweight = {}
        for term in list(term_relterms_withweight.keys()):
            _term_relterms_withweight[term] = [[], []]
            for value_i, value in enumerate(term_relterms_withweight[term][1]):
                sim_term = term_relterms_withweight[term][0][value_i]

                t_id = embedding.get_id_from_word(term)
                t_sim_id = embedding.get_id_from_word(sim_term)
                lsi_sim = 1

                if lsi_data[t_id] is not None:
                    for id, sim in lsi_data[t_id]:
                       if id == t_sim_id:
                          lsi_sim = sim
                          break

                if value >= embedding_filter_value and lsi_sim > lsi_filter_value:
                    _term_relterms_withweight[term][0].append(sim_term)
                    _term_relterms_withweight[term][1].append(value)
        return _term_relterms_withweight

    @staticmethod
    def filter_embedding_threshold(term_relterms_withweight, filter_value):
        _term_relterms_withweight = {}
        for term in list(term_relterms_withweight.keys()):
            _term_relterms_withweight[term] = [[], []]
            for value_i, value in enumerate(term_relterms_withweight[term][1]):
                if value >= filter_value:
                    _term_relterms_withweight[term][0].append(term_relterms_withweight[term][0][value_i])
                    _term_relterms_withweight[term][1].append(value)
        return _term_relterms_withweight

    @staticmethod
    def filter_count(term_relterms_withweight, filter_value):
        for term in list(term_relterms_withweight.keys()):
            list_simvals, list_words = list(zip(*sorted(zip(term_relterms_withweight[term][1],
                                                            term_relterms_withweight[term][0]), reverse=True)))
            term_relterms_withweight[term] = (list_words[0:filter_value + 1],
                                              list_simvals[0:filter_value + 1])  # +1 is for word itself
        return term_relterms_withweight

    @staticmethod
    def filter_countall(term_relterms_withweight, filter_value):
        _all_values = []
        for term in list(term_relterms_withweight.keys()):
            _all_values.extend(term_relterms_withweight[term][1])
        if filter_value < len(_all_values):
            _threshold_filter_value = _all_values[filter_value]
        else:
            _threshold_filter_value = _all_values[-1]

        _term_relterms_withweight = {}
        for term in list(term_relterms_withweight.keys()):
            _term_relterms_withweight[term] = [[], []]
            for value_i, value in enumerate(term_relterms_withweight[term][1]):
                if value >= _threshold_filter_value:
                    _term_relterms_withweight[term][0].append(term_relterms_withweight[term][0][value_i])
                    _term_relterms_withweight[term][1].append(value)
        return _term_relterms_withweight
