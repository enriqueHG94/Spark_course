import matplotlib.pyplot as plt
import seaborn as sb
from pyspark.sql import SparkSession
from pyspark.sql.functions import min, max
from pyspark.sql.types import StructField, IntegerType, DateType, StringType, DoubleType, StructType

spark = SparkSession.builder \
    .master("local[*]") \
    .appName('PySpark_Tutorial') \
    .getOrCreate()

# En este ejercicio vas a tener que cargar el dataset de “data/stocks_price.csv”, con el esquema correcto de datos

data_schema = [
    StructField('num', IntegerType(), True),
    StructField('symbol', StringType(), True),
    StructField('date', DateType(), True),
    StructField('open', DoubleType(), True),
    StructField('high', DoubleType(), True),
    StructField('low', DoubleType(), True),
    StructField('close', DoubleType(), True),
    StructField('volume', IntegerType(), True),
    StructField('adjusted', DoubleType(), True),
    StructField('market.cap', StringType(), True),
    StructField('sector', StringType(), True),
    StructField('industry', StringType(), True),
    StructField('exchange', StringType(), True),
]

final_struc = StructType(fields=data_schema)

data = spark.read.csv('resources/stock_price.csv',
                      sep=',',
                      header=True,
                      schema=final_struc
                      )

data.printSchema()

# Eliminar o renombrar la variable market.cap

data = data.drop('market.cap')

# Filtrar los datos donde el valor de “open” es nulo y eliminar esas filas

data.filter(data.open.isNull()).show()

data_new = data.na.drop(subset=["open"])

# Calcular el mínimo y máximo de data, open, close y adjusted

data_transf = data_new.groupBy("sector") \
    .agg(min("date").alias("From"),
         max("date").alias("To"),

         min("open").alias("Minimum Opening"),
         max("open").alias("Maximum Opening"),

         min("close").alias("Minimum Closing"),
         max("close").alias("Maximum Closing"),

         min("adjusted").alias("Minimum Adjusted Closing"),
         max("adjusted").alias("Maximum Adjusted Closing"),

         )

# Convertir un DataFrame de Spark en un DataFrame de pandas

data_transf_pd = data_transf.toPandas()

# Calcular la media de las variables open, close y adjusted por industria

data_df = data_new.select(['industry', 'open', 'close', 'adjusted']).groupBy('industry').mean().toPandas()

# Generar un gráfico de líneas donde se muestre la media de open por industria

data_df[['industry', 'avg(open)']].plot()
plt.show()

# Generar un heatmap con seaborn donde se muestre la correlación entre las medias de open, close y adjusted

corr = data_df[['avg(open)', 'avg(close)', 'avg(adjusted)']].corr()
sb.heatmap(corr, cmap="Blues", annot=True)

# Guardar en un archivo parquet una selección de datos

data_new.select(['date', 'open', 'close', 'adjusted']) \
    .write.save('dataset.parquet', format='parquet')

plt.show()

data_parquet = spark.read.parquet('dataset.parquet')

data_parquet.show()
