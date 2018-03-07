#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from gensim.models import TfidfModel
from gensim import corpora, models, similarities
from gensim.models.doc2vec import LabeledSentence
import gensim.utils
import datetime
from sklearn.model_selection import train_test_split
import random
import pickle
import numpy as np


def read_data():

    file = pd.read_csv("./tuss_para_sigtap.txt", delimiter="\t", names=['codigo_tuss', 'texto_tuss', 'codigo_sigtap', 'texto_sigtap'], encoding='utf-8')
    file2 = pd.read_csv("./tuss_tab_full.txt", delimiter="\t", names=['codigo', 'cap', 'grupo', 'subgrupo', 'seq', 'digito', 'nome_cap', 'nome_grupo', 'nome_subgrupo', 'procedimento'])

    return file,file2


def resample_input_file( my_file, train_size=0.9 ):

    # fazer o sampling aqui apesar de VARZEA
    print( "%s shuffling the input file (lazy but workey)" % ( str(datetime.datetime.now()) ) )
    tmp = my_file.reindex( np.random.permutation(my_file.index) )

    # return resampled sets
    return tmp[ 0:int( train_size*len(my_file) ) ], tmp[ int( train_size*len(my_file) ): ]  


def create_labeled_sentence_vectors(file,file2,file2_test):

    docs = []
    for index, row in file.iterrows():
        docs.append(LabeledSentence(words = gensim.utils.simple_preprocess(str(row["texto_sigtap"]), deacc=True, min_len=3, max_len=301), tags = ['tuss_' + str(row["codigo_tuss"])]))

    docs2 = []
    for index, row in file2.iterrows():
        docs2.append(LabeledSentence(words = gensim.utils.simple_preprocess(str(row["nome_cap"]) + ' ' + str(row["nome_grupo"]) + ' ' + str(row["nome_subgrupo"]) + ' ' + str(row["procedimento"]), deacc=True, min_len=3, max_len=301), tags = ['tuss_' + str(row["codigo"])]))

    docs2_test = []
    for index, row in file2_test.iterrows():
        docs2_test.append(LabeledSentence(words = gensim.utils.simple_preprocess(str(row["nome_cap"]) + ' ' + str(row["nome_grupo"]) + ' ' + str(row["nome_subgrupo"]) + ' ' + str(row["procedimento"]), deacc=True, min_len=3, max_len=301), tags = ['tuss_' + str(row["codigo"])]))

    return docs, docs2, docs2_test


def create_model( docs2, num_features ):
    #massa_treino, massa_teste = train_test_split(docs, train_size=0.9, test_size=0.1)
    #print("Quantidade de massa de treino = " + str(len(massa_treino)) + " e teste = " + str(len(massa_teste)))
    print(str(datetime.datetime.now()) + " Carregando o modelo")
    dictionary = corpora.Dictionary([i.words for i in docs2])
    corpus = [dictionary.doc2bow(i.words) for i in docs2]
    tfidf = models.TfidfModel(corpus)
    model = similarities.MatrixSimilarity(tfidf[corpus], num_features=num_features)
    print(str(model))
    print(str(datetime.datetime.now()) + " Modelo treinado")
    #model = similarities.MatrixSimilarity(tfidf[corpus], num_features=6000)
    #print(str(model))
    #print(str(datetime.datetime.now()) + " Modelo treinado")
    #model = models.TfidfModel.load("c:\\tmp\\modelo_tfidf.vec")

    return model,dictionary,tfidf


def evaluate(docs, docs2, model, dictionary, tfidf ):
    print(str(datetime.datetime.now()) + " Validando Modelo de Comparacao")
    codigos_certos=0
    cap_certos=0
    grupo_certos=0
    subgrupo_certos=0
    resultado = []
    for linha in docs:
        vetor = dictionary.doc2bow(gensim.utils.simple_preprocess(' '.join(linha.words), deacc=True, min_len=3, max_len=301))
        lista_similar = model[tfidf[vetor]]
        sims = sorted(enumerate(lista_similar), key=lambda item: -item[1])
        try:
            mais_similar = docs2[sims[0][0]][1]
        except:
            print("error")
            continue

        resultado.append(   ' '.join(linha.words) + ","  + " ".join(mais_similar) )

    #    print(str(linha.tags[0])[5:] + '|' + str(mais_similar[0][5:]) + '|' + str(mais_similar[1]))
        if linha.tags[0][5:] == mais_similar[0][5:]:
            codigos_certos += 1
        if linha.tags[0][5:6] == mais_similar[0][5:6]:
            cap_certos += 1
        if linha.tags[0][6:8] == mais_similar[0][6:8]:
            grupo_certos += 1
        if linha.tags[0][8:10] == mais_similar[0][8:10]:
            subgrupo_certos += 1

    print(str(datetime.datetime.now()) + " Fim da validação")
    return  codigos_certos, cap_certos, grupo_certos, subgrupo_certos, resultado

def my_train_test_split( dataset, train_size=0 ):
    ds = dataset[:]
    random.shuffle(ds)
    return ds,ds


def main():

    # parametros TODO dá pau se mudar pra menos
    num_features = 6000

    resampling_rounds = 10


    # docs = conjunto de mapeamento (gold standard entre tuss e sigtap)
    # docs2 = categorização hierarquica do tuss
    file,file2 = read_data()

   
    for r_round in range(resampling_rounds): # up to 90%

        train_file2, test_file2 = resample_input_file( file2, train_size=(1-1/resampling_rounds) )

        #filename = "/tmp/file%d.xls"%(r_round)
        #writer = pd.ExcelWriter(filename)
        #train_file2.to_excel( writer, "Coisa 1"  )
        #writer.save()

        # TODO use docs2_test
        docs,docs2,docs2_test = create_labeled_sentence_vectors(file , train_file2, test_file2)


        print( "%s Iniciando round %d de resampling (usando mytrain_test_split para debug-1)" % ( str(datetime.datetime.now()),r_round ) )


        # cria um modelo baseado nos dados do TUSS
        model,dictionary,tfidf = create_model( docs2, num_features )

        # realiza avaliação de acurácia do modelo gerado
        codigos_certos, cap_certos, grupo_certos, subgrupo_certos, resultado = evaluate( docs, docs2, model, dictionary, tfidf )

        filename = "/tmp/file-%d.xls"%(r_round)
        writer = pd.ExcelWriter(filename)
        pd.DataFrame(resultado).to_excel( writer, "Coisa 1"  )
        writer.save()



        print("Teste do modelo: Codigos;" + str(codigos_certos) + " Capitulo=" + str(cap_certos) + " Grupo=" + str(grupo_certos) + " SubGrupo=" + str(subgrupo_certos) )
        print("Teste do modelo: Codigos;" + str(codigos_certos/len(docs)) + " Capitulo=" + str(cap_certos/len(docs)) + " Grupo=" + str(grupo_certos/len(docs)) + " SubGrupo=" + str(subgrupo_certos/len(docs)) )


        print("SELF>%d%%,%d,%d,%d,%d,%d"%(r_round,codigos_certos,cap_certos,grupo_certos,subgrupo_certos,len(docs)))
    

main()
