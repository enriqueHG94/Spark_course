from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, DoubleType

'''En este ejercicio vamos a entrenar un modelo de clasificación para predecir la 
   probabilidad de un paciente de sufrir un ataque al corazón'''

spark = SparkSession.builder.appName('UCI Heart disease').getOrCreate()

heart = spark.read.csv('resources/heart.csv',
                       inferSchema=True,
                       header=True)

schema = StructType([StructField("age", LongType(), True),
                     StructField("sex", LongType(), True),
                     StructField("cp", LongType(), True),
                     StructField('trestbps', LongType(), True),
                     StructField("chol", LongType(), True),
                     StructField("fbs", LongType(), True),
                     StructField("restecg", LongType(), True),
                     StructField("thalach", LongType(), True),
                     StructField("exang", LongType(), True),
                     StructField("oldpeak", DoubleType(), True),
                     StructField("slope", LongType(), True),
                     StructField("ca", LongType(), True),
                     StructField("thal", LongType(), True),
                     StructField("target", LongType(), True),
                     ])

df = heart.withColumnRenamed("target", "label")

testDF, trainDF = df.randomSplit([0.3, 0.7])

# Creamos el modelo de regresión logística

lr = LogisticRegression(maxIter=10, regParam=0.01)

# Creamos un codificador en caliente.

ohe = OneHotEncoder(inputCols=['sex', 'cp', 'fbs', 'restecg', 'slope',
                               'exang', 'ca', 'thal'],
                    outputCols=['sex_ohe', 'cp_ohe', 'fbs_ohe',
                                'restecg_ohe', 'slp_ohe', 'exng_ohe',
                                'caa_ohe', 'thall_ohe'])

# Lista de entrada para el escalado

inputs = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Escalamos nuestras entradas

assembler1 = VectorAssembler(inputCols=inputs, outputCol="features_scaled1")
scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features_scaled")

# Creamos un segundo ensamblador para las columnas codificadas

assembler2 = VectorAssembler(inputCols=['sex_ohe', 'cp_ohe',
                                        'fbs_ohe', 'restecg_ohe',
                                        'slp_ohe', 'exng_ohe', 'caa_ohe',
                                        'thall_ohe', 'features_scaled'],
                             outputCol="features")

# Crear una lista de etapas

myStages = [assembler1, scaler, ohe, assembler2, lr]

# Configurar el pipeline

pipeline = Pipeline(stages=myStages)

# Ajustamos el modelo utilizando los datos de entrenamiento

pModel = pipeline.fit(trainDF)

# Transformamos los datos

trainingPred = pModel.transform(trainDF)

# Seleccionamos la etiqueta real, la probabilidad y las predicciones

trainingPred.select('label', 'probability', 'prediction').show()

pModel.save("resources/pipelines")
