from pandas_profiling import ProfileReport
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

'''En este ejercicio vamos a importar y pre procesar los datos de heart.csv que utilizaremos para entrenar un modelo de 
clasificación binaria con PySpark. Para ello, tendrás que inicializar una sesión de Spark, cargar los datos con el 
esquema correcto y analizar su distribución.Es decir, debes completar la parte de importación y análisis exploratorio 
de los datos.'''

spark = SparkSession.builder.appName('UCI Heart disease').getOrCreate()

heart = spark.read.csv('resources/heart.csv',
                       inferSchema=True,
                       header=True)

heart_pd = heart.toPandas()

profile = ProfileReport(heart_pd, minimal=True)

'''En este ejercicio vas a aplicar un pre-procesamiento mínimo en el conjunto de datos del ejercicio anterior,
   de manera que puedas utilizarlos para entrenar un modelo.'''

columns_df = heart_pd.columns

assembler = VectorAssembler(
    inputCols=['age',
               'sex',
               'cp',
               'trestbps',
               'chol',
               'fbs',
               'restecg',
               'thalach',
               'exang',
               'oldpeak',
               'slope',
               'ca',
               'thal'],
    outputCol="features")

output = assembler.transform(heart)

final_data = output.select("features", 'target')

'''En este ejercicio vas a entrenar un modelo de clasificación binaria con la librería de machine learning de PySpark, 
con el conjunto de datos pre-procesados del ejercicio anterior.
Una vez entrenado el modelo, vas a tener que realizar una predicción con el conjunto de datos de test y comparar los
 resultados. 
 A continuación, deberás obtener diferentes métricas de evaluación para determinar si el modelo es adecuado o no.'''

train, test = final_data.randomSplit([0.7, 0.3])

lr = LogisticRegression(labelCol="target",
                        featuresCol="features")

model = lr.fit(train)

predict_train = model.transform(train)

predict_test = model.transform(test)

predict_test.select("target", "prediction").show(10)

evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',
                                          labelCol='target')

predict_test.select("target", "rawPrediction", "prediction", "probability").show(5)

print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))

print("The area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))