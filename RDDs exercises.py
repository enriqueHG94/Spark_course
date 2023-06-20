from pyspark.sql import SparkSession
from operator import add

spark = SparkSession.builder.appName("RDDs exercises").getOrCreate()

sc = spark.sparkContext

# Genera un RDD (myRDD) con la siguiente lista [1, 2, 3, 4, 5]

data = [1, 2, 3, 4, 5, 6]

myRDD = sc.parallelize(data)

# Multiplica por 2 todos los elementos del RDD anterior

mapRDD = myRDD.map(lambda x: x*2)

# Filtra el RDD anterior por los elementos pares

filtRDD = myRDD.filter(lambda x: x%2 == 0)

# Muestra los elementos unicos del RDD

distRDD= myRDD.distinct()

# Genera un nuevo RDD con un par clave valor [('a', 1), ('a', 2), ('a', 3), ('b', 1)]

newRDD = sc.parallelize([('a', 1), ('a', 2), ('a', 3), ('b', 1)])

# Obten la suma de los valores agrupados por el key

keyRDD = newRDD.reduceByKey(add)

# Ordena los RDDs en base al key

sortRDD = newRDD.sortByKey()

# Genera un nuevo RDD para multiplicarlos entre si y obtener un resultado

data_new = [1, 2, 3, 4, 5]

dataRDD = sc.parallelize(data_new)

result = dataRDD.reduce(lambda x, y: x * y)

# Genera un nuevo RDD con ['Python', 'Scala', 'Python', 'R', 'Python', 'Java', 'R' ]

new_data = ['Python', 'Scala', 'Python', 'R', 'Python', 'Java', 'R' ]

lenguajesRDD = sc.parallelize(new_data)

# Cuenta cuantas veces aparece cada valor

count = lenguajesRDD.countByValue().items()

# Genera un nuevo RDD con [('a', 1), ('b', 1), ('c', 1), ('a', 1)]

key_data= [('a', 1), ('b', 1), ('c', 1), ('a', 1)]

key_RDD = sc.parallelize(key_data)

# Cuenta cuantas veces aparece cada una de las keys

count_key = key_RDD.countByKey().items()
