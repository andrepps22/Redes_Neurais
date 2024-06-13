import numpy as np


entrada = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0, 0, 0, 1])
pesos = np.array([0.0, 0.0])
taxaaprendizagem = 0.1


    
def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos) #Registros passaremos a variavel entrada que contem um array
    return stepFunction(s)

def treinar():
    errototal = 1
    while errototal != 0:
        errototal = 0
        for i in range(len(saidas)):
            #temos que tranformar o registro dentro do array como array o mesmo não está como array
            saidacalculada = calculaSaida(np.asarray(entrada[i]))
            #abs é para ter o valor absoluto
            erro = abs(saidas[i] - saidacalculada) 
            errototal += erro
            
            for j in range(len(pesos)): #Atualizar pejos
                pesos[j] = pesos[j] + (taxaaprendizagem * entrada[i][j] * erro)
                print(f'Peso atualizado: {pesos[j]}')
        print(f'Total de erros: {errototal}')


treinar()