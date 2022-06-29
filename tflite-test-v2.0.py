print('TFLite LSTM Energy Classivation V2.0')

# imports
import tflite_runtime.interpreter as tflite
import numpy as np
import pandas as pd
import cv2
import time
from datetime import datetime
import csv
import multiprocessing

# definitions
foldername = 'gaf_output_eval'
filename = 'output_dataset_eval.csv'
files_list = []
labels = ['abnormal', 'normal']
predictions = []

# import data
edge_dataset = pd.read_csv(filename, header=0)
files_list = edge_dataset['image_name'].values
print(edge_dataset.head())

# prepare tflite model
tflite_model_eval2_time_start = time.clock()
tflite_model_path = 'model.tflite'
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('Display inputs and outputs:')
print(input_details)
print(output_details)

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('width:', width, 'height', height)

# execute model (i.e. infer) on data files
for file in files_list:
  image_path = foldername + '/' + file
  image = cv2.imread(image_path)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  imH, imW, _ = image.shape 
  image_resized = cv2.resize(image_rgb, (width, height))
  input_data = np.expand_dims(image_resized, axis=0)

  # Perform the actual detection by running the model with the image as input
  interpreter.set_tensor(input_details[0]['index'],input_data)
  interpreter.invoke()
  prediction_tensor = interpreter.get_tensor(output_details[0]['index']) 
  prediction = np.argmax(prediction_tensor) + 1
  prediction_label = labels[np.argmax(prediction_tensor)]
  _dataset_prediction = edge_dataset.loc[edge_dataset['image_name'] == file].values[0][2]
  prediction_outcome = (prediction != _dataset_prediction)
  predictions.append(prediction_outcome)
  print('Filename:', file, '- TFLite tensor:', prediction_tensor, '- prediction label:', prediction_label, '- prediction:', prediction, '- isTrue?', prediction_outcome)
tflite_model_eval2_time_elapsed = time.clock() - tflite_model_eval2_time_start

# Calculate model evaluation metrics
predictions = np.array(predictions)
no_true_positives = len(predictions[predictions == 1])
tflite_accuracy = no_true_positives / len(predictions) # (no. of correct prediction)/(no. of all predictions)
print('TFLite accuracy: ', tflite_accuracy)

# Print running stats
timestamp = str(datetime.now())
stats = f'''Run stats (in sec) - {timestamp}
---------------------------TFLite-----------------------
tflite_model_eval2_time_elapsed (using interpreter): {tflite_model_eval2_time_elapsed} - accuracy: {tflite_accuracy}
-------------------------------------------------------
No. of CPU cores: {multiprocessing.cpu_count()}
'''
print(stats)

# csv file header: ['timestamp', 'tflite_eval_time_elapsed', 'accuracy'])
stats_list = [timestamp, tflite_model_eval2_time_elapsed, tflite_accuracy]
with open('stats_odroid.csv', "a") as csv_output_file:
    writer = csv.writer(csv_output_file)
    writer.writerow(stats_list)