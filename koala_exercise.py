import numpy as np
import pandas as pd
import databricks.koalas as ks

# FUNCIONES BÁSICAS

pser = pd.Series([1, 3, 5, np.nan, 6, 8])

# Crea una serie de Koalas con [1, 3, 5, np.nan, 6, 8]

kserie = ks.Series([1, 3, 5, np.nan, 6, 8])

# Pasa la serie de pandas pser a Koalas con el nombre de kser

kser = ks.from_pandas(pser)

# Ordena kser por el index

kser_sorted = kser.sort_index()

pdf = pd.DataFrame({'A': np.random.rand(5),
                    'B': np.random.rand(5)})

# Genera un Dataframe de Koalas con el pdf de pandas y llámalo kdf

kdf = ks.from_pandas(pdf)

# VISUALIZACIÓN DE DATOS

# Describe los datos de kdf

kdf_desc = kdf.describe()

# Ordena los datos de kdf por la columna B

kdf_sorted_values = kdf.sort_values(by='B')

# Transpón los datos de kdf

kdf_transposed = kdf.transpose()

# SELECCIÓN

# Selecciona las varaibles A y B de Kdf

var = kdf[['A', 'B']]

# Selecciona las filas 1, 2 de kdf

row = kdf.loc[1:2]

# Selecciona las filas 0, 1 y 2 de la variable B

row_var = kdf.iloc[:3, 1:2]

# APLICANDO FUNCIONES DE PYTHON A KOALAS

# Aplica la función de Python de np.cumsum a kdf

kdf_cums = kdf.apply(np.cumsum)

# Eleva al cuadrado los valores de kdf

kdf_squared = kdf.apply(lambda x: x ** 2)

# AGRUPANDO DATOS

# Obten la suma de los valores al agrupar por A y por B

kdf_grouped = kdf.groupby(['A', 'B']).sum()

# GENERANDO GRÁFICOS

speed = [0.1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant',
         'rabbit', 'giraffe', 'coyote', 'horse']

kdf_gra = ks.DataFrame({'speed': speed,
                   'lifespan': lifespan}, index=index)

# Genera un grafico de barras con kdf y matplotlib

kdf_gra.plot.barh()

kdf_area = ks.DataFrame({
    'sales': [3, 2, 3, 9, 10, 6, 3],
    'signups': [5, 5, 6, 12, 14, 13, 9],
    'visits': [20, 42, 28, 62, 81, 50, 90],
}, index=pd.date_range(start='2019/08/15', end='2020/03/09',
                       freq='M'))

# Genera un grafico de areas con kdf y matplotlib

kdf_area.plot.area()

# UTILIZANDO SQL EN KOALAS

koalas_df = ks.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                    'pig': [20, 18, 489, 675, 1776],
                    'horse': [4, 25, 281, 600, 1900]})

# Con una consulta SQL selecciona los datos donde pig sea mayor que 100

ks.sql("SELECT * FROM {koalas_df} WHERE pig > 100")

pdf_join = pd.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                    'sheep': [22, 50, 121, 445, 791],
                    'chicken': [250, 326, 589, 1241, 2118]})

# Haz un inner join entre koalas_df y pdf_join en la variable year, selecciona el pig y el chicken
# ordena los datos por pig y chicken

ks.sql('''
    SELECT pig, chicken
    FROM (SELECT * FROM {koalas_df}) AS koalas_df
    INNER JOIN (SELECT * FROM {pdf_join}) AS pdf_join
    ON koalas_df.year = pdf_join.year
    ORDER BY pig, chicken''')

# TRABAJANDO CON PYSPARK

kdf_py = ks.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# Convierete el dataframe de Koalas a Dataframe de Spark

sdf = kdf_py.to_spark()

type(sdf)

# muestra los datos

sdf.show()
