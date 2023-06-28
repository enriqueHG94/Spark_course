from pyspark.sql import SparkSession

spark = SparkSession.builder\
        .master("local[*]")\
        .appName('PySpark_Df')\
        .getOrCreate()

# Importa el csv de "data/WorldCupPlayers.csv"
# Visualiza los datos

fifa_df = spark.read.csv ("resources/WorldCupPlayers.csv",
                          inferSchema = True,
                          header = True)
fifa_df.show()

# ¿que tipo de datos contiene cada variable?

fifa_df.printSchema()

# ¿Cuantos registros hay?

print(fifa_df.count())

# Obtén los principales estadísticos de Position

fifa_df.describe('Position').show()

# Slecciona y muestra los registros distintos de 'Player Name','Coach Name'

fifa_df.select('Player Name','Coach Name').distinct().show()

# ¿Cuantos partidos con el ID de 1096 ha habido?

print(fifa_df.filter(fifa_df.MatchID =='1096').count())

# Muestra los datos donde la posicion haya sido C y el evento sea G40

fifa_df.filter((fifa_df.Position == 'C') & (fifa_df.Event == "G40'")).show()

# Utiliza Spark SQL para mostras los registros donde el MatchID sea mayor o igual a 20

fifa_df.createOrReplaceTempView("temp_table")

spark.sql("select * from temp_table where MatchID >= 20").show()