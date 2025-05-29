import numpy as np
from gensim import downloader as api

class Encoding:
    def __init__(self, frasi, embedding_dim=16, use_pretrained=True):
        self.frasi = [frase.split() for frase in frasi] 
        self.embedding_dim = embedding_dim
        self.use_pretrained = use_pretrained
        self.vocabolario = self._build_vocabulary()
        self.model = self._load_model()
        self.embedding_matrix = self._get_embedding_matrix()
        self.embedding_finale = self._embedding_input()
        self.statevectors = self._embedding_to_statevectors()

    def _build_vocabulary(self):
        vocabolario = {}
        indice = 0
        for frase in self.frasi:
            for parola in frase:
                if parola not in vocabolario:
                    vocabolario[parola] = indice
                    indice += 1
        return vocabolario

    def _load_model(self):
        if self.use_pretrained:
            return api.load("word2vec-google-news-300")
        return None

    def _get_embedding_matrix(self):
        vocab_size = len(self.vocabolario)
        matrix = np.zeros((vocab_size, self.embedding_dim))
        for parola, indice in self.vocabolario.items():
            if self.model and parola in self.model:
                matrix[indice] = self.model[parola][:self.embedding_dim]
            else:
                matrix[indice] = np.random.uniform(0, 0.1, self.embedding_dim)
        return matrix

    def _positional_encoding(self, seq_len):
        d_model = self.embedding_dim
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    def getFrasi(self):
        return self.frasi

    def _embedding_input(self):
        frasi_tokenizzate = self.frasi
        tutte_embeddate = []
        for frase in frasi_tokenizzate:
            embeddings = []
            for parola in frase:
                indice = self.vocabolario[parola]
                embeddings.append(self.embedding_matrix[indice])
            embeddings = np.array(embeddings)
            pos = self._positional_encoding(len(frase))
            tutte_embeddate.append(embeddings + pos)
        return tutte_embeddate

    def _embedding_to_statevectors(self):
        tutte_frasi_quantistiche = []
        for frase_embed in self.embedding_finale:
            frase_quantistica = []
            for vettore in frase_embed:
                stato = vettore / np.linalg.norm(vettore)
                frase_quantistica.append(stato)
            tutte_frasi_quantistiche.append(frase_quantistica)
        return tutte_frasi_quantistiche

    def psi_globale(self):
        dim = self.embedding_dim
        psi = np.zeros(dim * dim)
        for frase in self.statevectors:
            for t in frase:
                t = t / np.linalg.norm(t)
                psi += np.kron(t, t)
        return psi / np.linalg.norm(psi)
    
    def psi_locale(self,indexPhrase,indexWord):
        dim = self.embedding_dim
        psi = np.zeros(dim * dim)
        phrase = self.statevectors[indexPhrase][:indexWord]
        
        for t in phrase:
            print(t)
            t = t / np.linalg.norm(t)
            psi += np.kron(t, t)
        return psi / np.linalg.norm(psi)
