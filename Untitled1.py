#!/usr/bin/env python
# coding: utf-8

# #Machine learning

# In[42]:


#importando a blibioteca pandas, e lendo o arquivo que tem os conjuntos de dados
#os dados são de caracteristicas de um um determinado vinho, e se ele é branco ou vermelho.
#o objetivo final é tentar prever qual a cor do vinho a parti de suas caracteristicas usando machine leraning,
import pandas as pd
arquivo = pd.read_csv('C:\\Users\\thelo\\Analise de Dados\\Projetos\\DADOS\\wine_dataset.csv')


# In[43]:


#ler as 5 primeiras linhas do conjunto de dados, e suas colunas.
arquivo.head()


# In[44]:


#Muda o valor de red para 0 nas linhas, na coluna style // pra usar a sklearn os rotulos precisam ser em numeros
# nesse caso como so tem dois tipos de vinho, decidi colocar 0 e 1
arquivo['style'] = arquivo['style'].replace('red',0)


# In[45]:


##Muda o valor de white para 1 nas linhas, na coluna style
arquivo['style'] = arquivo['style'].replace('white',1)


# In[46]:


#separa as variaveis
#y = coluna style, o rotulo, vermelho ou branco
y = arquivo['style']
# x = todas as colunas menos a style, as caracteristicas, ph, qualidade etc...
x = arquivo.drop('style',axis = 1)


# In[47]:


from sklearn.model_selection import train_test_split
#criando o conjunto de dados de treino e test:
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size = 0.3)


# In[48]:


from sklearn.ensemble import ExtraTreesClassifier
#criando modelos
modelo = ExtraTreesClassifier()
modelo.fit(x_treino,y_treino)

#imPRIMir resultados
resultado = modelo.score(x_teste,y_teste)
print("acuracia", resultado)


# In[49]:


#visualizando os dados a serem testados
y_teste[300:303]


# In[50]:


#visualizando os dados a serem testados
x_teste[300:303]


# In[51]:


#Prevendo o resultado do rotulo, aparti das caracteristicas
previsao = modelo.predict(x_teste[300:303])


# In[52]:


#printando a previsão
previsao

