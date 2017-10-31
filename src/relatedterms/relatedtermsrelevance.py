__author__='navid'

import common.src.misc.tools as tools
from .relatedterms import RelatedTerms

class RelatedTermsRelevance(RelatedTerms):

    def __init__(self):
        self.socket_address_ter4=('localhost', 33300)

    def read_prf_reply(self, reply):
        term_neighbors=[[],[]]
        for chunk in reply.strip().split(' '):
            values=chunk.split('^')
            if len(values)==2:
                term_neighbors[0].append(values[0])
                term_neighbors[1].append(float(values[1]))
        return term_neighbors


    def get_relatedterms(self, index_dir_path, terms):

        #all
        clientsocket = tools.create_clientsocket(self.socket_address_ter4)
        clientsocket.sendall(index_dir_path+'$queryexpandedterms$'+' '.join(terms)+'\n')
        reply = tools.recvall(clientsocket).strip('\n').strip('#')
        result_all=self.read_prf_reply(reply)
        clientsocket.close()

        return result_all

