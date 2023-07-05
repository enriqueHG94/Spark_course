import time
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, DoubleType

'''En este ejercicio debemos generar un proceso que realice predicciones sobre un conjunto de datos en streaming,
 utilizando un modelo de Machine Learning de clasificación.'''

spark = SparkSession.builder.appName('UCI Heart disease').getOrCreate()

heart = spark.read.csv('resources/heart.csv',
                       inferSchema=True,
                       header=True)

schema = StructType(
    [StructField("age", LongType(), True),
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

# CARGAR EL PIPELINE

pModel = PipelineModel.load("resources/pipelines")

# Transformamos los datos

trainingPred = pModel.transform(trainDF)

# Seleccionamos la etiqueta real, la probabilidad y las predicciones

trainingPred.select('label', 'probability', 'prediction')

testData = testDF.repartition(10)

# Creamos un directorio

testData.write.format("CSV").option("header", False).mode("overwrite").save("resources/heart_streaming/")

# CREANDO PREDICCIONES EN STREAMING

sourceStream = (
    spark.readStream.schema(schema)
    .option("maxFilesPerTrigger", 1)
    .csv("resources/heart_streaming")
    .withColumnRenamed("target", "label")
)

prediction1 = pModel.transform(sourceStream).select('label',
                                                    'probability',
                                                    'prediction')

# MOSTRANDO LOS PREDICCIONES EN CONSOLA

query1 = prediction1.writeStream.queryName("prediction1") \
    .format("console") \
    .trigger(once=True) \
    .start() \
    .awaitTermination()

# GUARDANDO LAS PREDICCIONES EN MEMORIA

query2 = (
    prediction1.writeStream.queryName("prediction4")
    .format("memory")
    .outputMode("append")
    .start())

time.sleep(5)

for x in range(2):
    df = spark.sql(
        "SELECT * FROM prediction4")
    df.show(10)

# Validando que el proceso de streaming está activo y después muestra el estado

is_active = spark.streams.active[0].isActive

status = query2.status

