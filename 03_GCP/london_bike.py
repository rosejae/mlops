from datetime import datetime
import pandas as pd
import tensorflow as tf

#
# BigQuery to Workbench
#

from google.cloud.bigquery import Client, QueryJobConfig
client = Client()

query = """WITH staging AS (
  SELECT
    STRUCT(
      start_stn.name,
      ST_GEOGPOINT(start_stn.longitude, start_stn.latitude) AS POINT,
      start_stn.docks_count,
      start_stn.install_date
    ) AS starting,
    STRUCT(
      end_stn.name,
      ST_GEOGPOINT(end_stn.longitude, end_stn.latitude) AS point,
      end_stn.docks_count,
      end_stn.install_date
    ) AS ending,
    STRUCT(
      rental_id,
      bike_id,
      duration,  -- seconds
      ST_DISTANCE(
        ST_GEOGPOINT(start_stn.longitude, start_stn.latitude),
        ST_GEOGPOINT(end_stn.longitude, end_stn.latitude)
      ) AS distance,  -- meters
      start_date,
      end_date
    ) AS bike
    FROM `bigquery-public-data.london_bicycles.cycle_stations` AS start_stn
    LEFT JOIN `bigquery-public-data.london_bicycles.cycle_hire` as b
    ON start_stn.id = b.start_station_id
    LEFT JOIN `bigquery-public-data.london_bicycles.cycle_stations` AS end_stn
    ON end_stn.id = b.end_station_id
    LIMIT 100000)

SELECT * FROM STAGING"""
job = client.query(query)
df = job.to_dataframe()

values = df["bike"].values

duration = list(map(lambda a: a["duration"], values))
distance = list(map(lambda a: a["distance"], values))
dates = list(map(lambda a: a["start_date"], values)) 
data = pd.DataFrame(data={'duration': duration, 'distance': distance, 'start_date': dates})

data = data.dropna() # 결측값이 있는 데이터는 제거

# start_date -> weekday, hour
# duration -> minutes unit

data["weekday"] = data["start_date"].apply(lambda a: a.weekday())
data["hour"] = data["start_date"].apply(lambda a: a.time().hour)
data = data.drop(columns=["start_date"])
data["duration"] = data["duration"].apply(lambda x: float(x/60))

#
# dataset split (train/validation)
#

def df_to_dataset(dataframe, label, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(label)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

train_size = int(len(data)*0.8)
train_data = data[:train_size]
val_data = data[train_size:]

print(len(train_data)/len(data))
print(len(val_data)/len(data))

train_dataset = df_to_dataset(train_data, 'duration')
validation_dataset = df_to_dataset(val_data, 'duration')

#
# model
#

def get_normalization_layer(name, dataset):
    normalizer = tf.keras.layers.Normalization(axis=None)
    feature_ds = dataset.map(lambda x, y: x[name])
    
    normalizer.adapt(feature_ds)
    
    return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)
    
    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
    
    encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    
    return lambda feature: encoder(index(feature))

numeric_col = tf.keras.Input(shape=(1,), name='distance')
hour_col = tf.keras.Input(shape=(1,), name='hour', dtype='int64')
weekday_col = tf.keras.Input(shape=(1,), name='weekday', dtype='int64')

all_inputs = []
encoded_features = []

# Pass 'distance' input to normalization layer
normalization_layer = get_normalization_layer('distance', train_dataset)
encoded_numeric_col = normalization_layer(numeric_col)
all_inputs.append(numeric_col)
encoded_features.append(encoded_numeric_col)

# Pass 'hour' input to category encoding layer
encoding_layer = get_category_encoding_layer('hour', train_dataset, dtype='int64')
encoded_hour_col = encoding_layer(hour_col)
all_inputs.append(hour_col)
encoded_features.append(encoded_hour_col)

# Pass 'weekday' input to category encoding layer
encoding_layer = get_category_encoding_layer('weekday', train_dataset, dtype='int64')
encoded_weekday_col = encoding_layer(weekday_col)
all_inputs.append(weekday_col)
encoded_features.append(encoded_weekday_col)

all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(64, activation="relu")(all_features)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mean_squared_logarithmic_error")
model.fit(train_dataset, validation_data=validation_dataset, epochs=5)