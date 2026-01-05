#sprint 16 proyecto
#importamos librerias
import math
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier


################################################################################################################################################################################################################################################################################################P1
#Parte 1 carga de datos
df= pd.read_csv('C:/Users/drgarciacabo/Music/tripleten/DS/sprint 16 aprendizaje automático para textos/sprint 16 proyecto/imdb_reviews.tsv', sep='\t', dtype={'votes':'Int64'})

################################################################################################################################################################################################################################################################################################P2
#Parte 2 preprocesamiento de datos

df['average_rating']=df['average_rating'].fillna(0)
df['votes']=df['votes'].fillna(0)

df['review'] = df['review'].str.lower()

#print(df.head())
#df.info()


################################################################################################################################################################################################################################################################################################P3
#Parte 3 anállisis exploratorio de datos

#Analizaremos si hay desequilibrio de clases o no
print(df['pos'].value_counts())
plt.figure(figsize=(6, 5))
sns.countplot(x='pos', data=df)
plt.title('Distribución de Clases de Sentimiento')
plt.xlabel('Clase de Sentimiento (0=Negativa, 1=Positiva)')
plt.ylabel('Número de Reseñas')
plt.show()
#Con una diferencia de menos de 100 observaciones
# podemos concluir con seguridad que no hay un desequilibrio de clases significativo


# Analizaremos longitud de las resenas
df['review_length'] = df['review'].apply(len)
#print("Estadísticas de la longitud de las reseñas:")
#print(df['review_length'].describe())

#la longitud promedio es aceptable
#hay alta variabilidad en la longitud de las resenas
#son menos de 2 mil las resenas que tienen más de 3 mil palabras

################################################################################################################################################################################################################################################################################################P4
#Parte 4 preprocesamiento de datos para el modelado

train= df[df['ds_part'] == 'train']
test= df[df['ds_part'] == 'test']

features_train = train['review']
target_train = train['pos']
features_test = test['review']
target_test = test['pos']

count_tf_idf = TfidfVectorizer(stop_words='english')
features_train_idf = count_tf_idf.fit_transform(features_train)
features_test_idf = count_tf_idf.transform(features_test)


################################################################################################################################################################################################################################################################################################P5
#Parte 5 Entrenamiento de modelos
print("\n entrenando modelo LR")
#realizaremos un modelo de regresión logística
model_lr=LogisticRegression(random_state=54321, solver='liblinear')
model_lr.fit(features_train_idf, target_train)


print("\n entrenando modelo lgbm")
#creamos y entrenamos el modelo lgbm
model_lgbm = lgb.LGBMClassifier( objective='binary', random_state=12345)
model_lgbm.fit(features_train_idf, target_train )

print("\n entrenando modelo catboost")
#creamos y entrenamos el modelo catboost
model_cb = CatBoostClassifier(random_state= 12345, depth=5, iterations=100)
model_cb.fit(features_train_idf, target_train)



################################################################################################################################################################################################################################################################################################P6
#Parte 6 Prueba de modelos

#evaluamos el modelo de regresión logística 
predictions_lr = model_lr.predict(features_test_idf)
score_lr=f1_score(target_test, predictions_lr)
print(f"Valor F1 para modelo de regresión logística: {score_lr} ")

#evaluamos el modelo lgbm
predictions_lgbm = model_lgbm.predict(features_test_idf)
score_lgbm=f1_score(target_test, predictions_lgbm)
print(f"Valor F1 para modelo LightGBM: {score_lgbm} ")

#evaluamos el modelo catboost
predictions_cb= model_cb.predict(features_test_idf)
score_cb=f1_score(target_test, predictions_cb)
print(f"Valor F1 para modelo CatBoost: {score_cb} ")



################################################################################################################################################################################################################################################################################################P7
#Parte 7 Escribir algunas reseñas y clasificarlas

#creamos lista con las nuevas resenas y con ella creamos un nuevo df
new_reviews=[
    'Fantastic movie I loved every single frame on it, great actors and great director',
    'The worst movie I have ever seen in my life, terrible in every way',
    'It is a good movie however there are some boring elements ',
    'This is not an easy movie to watch, its very difficult to understand the plot',
    'I dont understand why people hates this movie, I think is brilliant',
    'Its an artistic and wonderfull piece but is not made for everyone, I can get why people dont like it at all'
    ]

my_revs=pd.DataFrame(new_reviews, columns=['review'])
my_revs['review'] = my_revs['review'].str.lower()

#obtenemos nuestras caracteristicas vectorizando las reviews
new_features_idf = count_tf_idf.transform(my_revs['review'])

#predicciones de modelos
my_revs['pred_lr'] = model_lr.predict(new_features_idf)
my_revs['pred_lgbm'] = model_lgbm.predict(new_features_idf)
my_revs['pred_cb'] = model_cb.predict(new_features_idf)

#vemos los resultados de los modelos, dropeo las reviews para verlos completos
print(my_revs.drop('review', axis=1).head(6))



################################################################################################################################################################################################################################################################################################P8
#Parte 8 diferencias entre los resultados de las pruebas de los modelos 

#crearemos una lista con el verdadero resultado de las criticas
#(mas adelante se hablará de esta cuestión) 
real_feel=[1,0,0,0,1,1]

my_revs['real']=real_feel

print("Evaluación de las predicciones con las nuevas reviews")
score_lr_new=f1_score(my_revs['real'], my_revs['pred_lr'])
score_lgbm_new=f1_score(my_revs['real'], my_revs['pred_lgbm'])
score_cb_new=f1_score(my_revs['real'], my_revs['pred_cb'])


print(f"F1 LR: {score_lr_new}")
print(f"F1 LGBM: {score_lgbm_new}")
print(f"F1 CatBoost:{score_cb_new}")


