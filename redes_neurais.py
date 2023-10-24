from sklearn.neural_network import MLPClassifier
import pickle #ler base de dados de extensão .pkl
from sklearn.metrics import accuracy_score, classification_report #Testando a acuracia
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open ('credit.pkl','rb') as f: #Abre o arquvi em modo de leitura binária (rb)
    X_credit_treino, y_credit_treino, X_credit_test, y_credit_test = pickle.load(f)

#Exibido os dados de treino. Com 1500 registros e 3 colunas para X e 1500 registros e 1 coluna para Y
print('dados de treino:',X_credit_treino.shape, y_credit_treino.shape) #Função shape exibe a quantidade de linha e coluna da base de dados
print('\ndados de teste: ',X_credit_test.shape, y_credit_test.shape)

#Dados no array
print(X_credit_treino)

RNN_credit = MLPClassifier(max_iter=1000, verbose=True, tol = 0.0001,solver='adam', activation='relu', hidden_layer_sizes=(20,20))
#Max_iter - numero de épocas que o algorimo vai rodar atualizando os pesos
#Verbose - exibe informação do erro de cada época
#Solver - algoritmo utilizado na descida do gradiente (Adam - por meio da derivada)
#Activation - função de ativação Relu (por default é deduzido essa função)
#Hidden_layer_sizes - Numero de neuronio da camada oculta - partindo de um principio ((atributos + saida)/2)
#Tol - Diferença do erro para encerrar as integrações

RNN_credit.fit(X_credit_treino, y_credit_treino) #Função 'fit' para realizar o treino
previsao = RNN_credit.predict(X_credit_test) #Realiza a predição comparando com os dados de X teste
print(previsao)
print(y_credit_test)

print(accuracy_score(y_credit_test, previsao)) 
cm = ConfusionMatrix(RNN_credit)
cm.fit(X_credit_treino, y_credit_treino)
cm.score(X_credit_test, y_credit_test)
plt.show()