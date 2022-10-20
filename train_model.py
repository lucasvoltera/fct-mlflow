# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC select * from sandbox_apoiadores.abt_dota_pre_match

# COMMAND ----------

# DBTITLE 1,Imports
## import das libs
from sklearn import model_selection
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
import mlflow

## sdf = sparkdataframe
## import dos dados
sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")

## convertendo do spark para pandas
df = sdf.toPandas()

## mostrar o dataframe com o spark
#sdf.display()

df.info(memory_usage = 'deep')

# COMMAND ----------

# DBTITLE 1,Definição das variaveis
target_column = 'radiant_win'

id_column = 'mach_id'

features_columns = list( set(df.columns.tolist()) - set([target_column, id_column]) )

#features_columns

X = df[features_columns]
y = df[target_column]


# COMMAND ----------

# DBTITLE 1,Split Test e Train

## dividir o dataset para treinar o modelo e para validar o modelo
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 42)

print(f'Tamanho dados X_train: {X_train.shape[0]}\nTamanho dados X_test: {X_test.shape[0]}\nTamanho y_train: {y_train.shape[0]}\nTamanho y_test: {y_test.shape[0]}')

# COMMAND ----------

# DBTITLE 1,Setup do experimento MLFlow
mlflow.set_experiment("/Users/lucas.voltera@unesp.br/dota-unesp-lucasvoltera")

# COMMAND ----------

# DBTITLE 1,Run do experimento

## instanciando o modelo
#model = tree.DecisionTreeClassifier()
#model = ensemble.RandomForestClassifier()
#model = linear_model.LogisticRegression()

with mlflow.start_run():
    
    mlflow.sklearn.autolog()
    
    #model = ensemble.AdaBoostClassifier(n_estimators = 100, learning_rate = 0.7)
    #model = linear_model.LogisticRegression()
    model = neighbors.KNeighborsClassifier()
    #model = svm.SVC()
    
    ## treinando o modelo
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)

    ## acuracia é taxa de acerto
    acc_train = metrics.accuracy_score(y_train, y_train_pred)

    print(f'Acurácia em train: {acc_train}')
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    ## acuracia é taxa de acerto
    acc_test = metrics.accuracy_score(y_test, y_test_pred)

    print(f'Acurácia em test: {acc_test}')

# COMMAND ----------


