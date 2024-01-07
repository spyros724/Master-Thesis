import tensorflow as tf
import numpy as np

def ds_generator_historical(data_ts, batch_s, fh):
    # TS samples
    ts_x = data_ts[:, :-fh]
    ts_y = data_ts[:, -fh:]
    train_length = len(ts_x)

    # Training data
    data_train = tf.data.Dataset.from_tensor_slices((ts_x, ts_y))
    data_train = data_train.shuffle(buffer_size=train_length)
    data_train = data_train.repeat()
    data_train = data_train.batch(batch_size=batch_s, num_parallel_calls=tf.data.AUTOTUNE)
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train

def ds_generator_historical_calendar(data_ts, data_calendar, batch_s, fh):
    # TS samples
    ts_x = data_ts[:, :-fh]
    ts_y = data_ts[:, -fh:]
    train_length = len(ts_x)

    # Training data
    ds_x_ts = tf.data.Dataset.from_tensor_slices((ts_x))
    ds_x_exog = tf.data.Dataset.from_tensor_slices((data_calendar))
    ds_y = tf.data.Dataset.from_tensor_slices((ts_y))

    data_train = tf.data.Dataset.zip(((ds_x_ts, ds_x_exog), ds_y))
    data_train = data_train.shuffle(buffer_size=train_length)
    data_train = data_train.repeat()
    data_train = data_train.batch(batch_size=batch_s, num_parallel_calls=tf.data.AUTOTUNE)
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train

def ds_generator_historical_weather(data_ts, data_weather, batch_s, fh):
    # TS samples
    ts_x = data_ts[:, :-fh]
    ts_y = data_ts[:, -fh:]
    train_length = len(ts_x)

    # Training data
    ds_x_ts = tf.data.Dataset.from_tensor_slices((ts_x))
    ds_x_weather = tf.data.Dataset.from_tensor_slices((data_weather))
    ds_y = tf.data.Dataset.from_tensor_slices((ts_y))

    # data_train = tf.data.Dataset.zip(((ds_x_ts, ds_x_weather), ds_y))
    data_train = tf.data.Dataset.zip(((ds_x_ts, ds_x_weather), ds_y))
    data_train = data_train.shuffle(buffer_size=train_length)
    data_train = data_train.repeat()
    data_train = data_train.batch(batch_size=batch_s, num_parallel_calls=tf.data.AUTOTUNE)
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train

def ds_generator_historical_country(data_ts, data_country, batch_s, fh):
    # TS samples
    ts_x = data_ts[:, :-fh]
    ts_y = data_ts[:, -fh:]
    train_length = len(ts_x)

    # Training data
    ds_x_ts = tf.data.Dataset.from_tensor_slices((ts_x))
    ds_x_country = tf.data.Dataset.from_tensor_slices((data_country))
    ds_y = tf.data.Dataset.from_tensor_slices((ts_y))

    # data_train = tf.data.Dataset.zip(((ds_x_ts, ds_x_weather), ds_y))
    data_train = tf.data.Dataset.zip(((ds_x_ts, ds_x_country), ds_y))
    data_train = data_train.shuffle(buffer_size=train_length)
    data_train = data_train.repeat()
    data_train = data_train.batch(batch_size=batch_s, num_parallel_calls=tf.data.AUTOTUNE)
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train

def ds_generator_historical_calendar_country(data_ts, data_calendar, data_country, batch_s, fh):
    # TS samples
    ts_x = data_ts[:, :-fh]
    ts_y = data_ts[:, -fh:]
    train_length = len(ts_x)

    # Training data
    ds_x_ts = tf.data.Dataset.from_tensor_slices((ts_x))
    ds_x_calendar = tf.data.Dataset.from_tensor_slices((data_calendar))
    ds_x_country = tf.data.Dataset.from_tensor_slices((data_country))
    ds_y = tf.data.Dataset.from_tensor_slices((ts_y))

    data_train = tf.data.Dataset.zip(((ds_x_ts, ds_x_calendar, ds_x_country), ds_y))
    data_train = data_train.shuffle(buffer_size=train_length)
    data_train = data_train.repeat()
    data_train = data_train.batch(batch_size=batch_s, num_parallel_calls=tf.data.AUTOTUNE)
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train

def ds_generator_historical_calendar_weather(data_ts, data_calendar, data_weather, batch_s, fh):
    # TS samples
    ts_x = data_ts[:, :-fh]
    ts_y = data_ts[:, -fh:]
    train_length = len(ts_x)

    # Training data
    ds_x_ts = tf.data.Dataset.from_tensor_slices((ts_x))
    ds_x_calendar = tf.data.Dataset.from_tensor_slices((data_calendar))
    ds_x_weather = tf.data.Dataset.from_tensor_slices((data_weather))
    ds_y = tf.data.Dataset.from_tensor_slices((ts_y))

    data_train = tf.data.Dataset.zip(((ds_x_ts, ds_x_calendar, ds_x_weather), ds_y))
    data_train = data_train.shuffle(buffer_size=train_length)
    data_train = data_train.repeat()
    data_train = data_train.batch(batch_size=batch_s, num_parallel_calls=tf.data.AUTOTUNE)
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train

def ds_generator_historical_calendar_country_weather(data_ts, data_calendar, data_country, data_weather, batch_s, fh):
    # TS samples
    ts_x = data_ts[:, :-fh]
    ts_y = data_ts[:, -fh:]
    train_length = len(ts_x)

    # Training data
    ds_x_ts = tf.data.Dataset.from_tensor_slices((ts_x))
    ds_x_calendar = tf.data.Dataset.from_tensor_slices((data_calendar))
    ds_x_country = tf.data.Dataset.from_tensor_slices((data_country))
    ds_x_weather = tf.data.Dataset.from_tensor_slices((data_weather))
    ds_y = tf.data.Dataset.from_tensor_slices((ts_y))

    data_train = tf.data.Dataset.zip(((ds_x_ts, ds_x_calendar, ds_x_country, ds_x_weather), ds_y))
    data_train = data_train.shuffle(buffer_size=train_length)
    data_train = data_train.repeat()
    data_train = data_train.batch(batch_size=batch_s, num_parallel_calls=tf.data.AUTOTUNE)
    data_train = data_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    return data_train

