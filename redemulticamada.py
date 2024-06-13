"""
Criado Quarta-Feira dia 12-06-2024 as 15:20

@Autor: André de Paula

"""

import numpy as np

entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([[0],[1],[1],[0]])

pesos0 = 2*np.random.random((2,3)) -1
pesos1 = 2*np.random.random((3,1)) -1

taxaaprendizagem = 0.6
momento = 1
epocas = 1000000


def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def derivadasigmoid(sig):
    return sig * (1 - sig)



for j in range(epocas):
    sinapsentradaoculta = np.dot(entradas, pesos0)
    camadaoculta = sigmoid(sinapsentradaoculta)
    sinapseocultasaida = np.dot(camadaoculta,pesos1)
    camadasaida = sigmoid(sinapseocultasaida)
    
    #calculo do erro
    errosaida = saidas - camadasaida
    mediaerro = np.mean(np.abs(errosaida))
        
    #calculo do delta camada saida
    deltasaida = errosaida * derivadasigmoid(camadasaida)
    
    #calculo do dela camada oculta
    pesos1T = pesos1.T         
    deltaoculta = deltasaida *pesos1T * derivadasigmoid(camadaoculta)
    
    #backpropagation
    #camadaoculta
    camadaocultaT = camadaoculta.T
    pesosnovo1 = camadaocultaT.dot(deltasaida)
    pesos1 = (pesos1 * momento) + (pesosnovo1 * taxaaprendizagem)
    
    #camadaentrada
    camadaentradaT = entradas.T
    pesosnovos0 = camadaentradaT.dot(deltaoculta)
    pesos0 = (pesos0 * momento) + (pesosnovos0 * taxaaprendizagem)

    if j % 10000 == 0:
        print(f'Erro total: {mediaerro}')

print(camadasaida) 
         
        
    

