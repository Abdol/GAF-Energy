print('Hi')

# imports
import tflite_runtime.interpreter as tflite
import numpy as np
import pandas as pd
import cv2

# definitions
foldername = 'dataset_edge_test'
filename = 'dataset_edge_test.csv'
files_list = []
labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
predictions = []

# import data
edge_dataset = pd.read_csv(filename, header=0)
files_list = edge_dataset['image_name'].values
print(edge_dataset.head())

# prepare tflite model
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
  image_path = foldername + '/' + file # 'dandelion.jpg' # 'rose.jpg' # 'sunflower.jpg' # 'daisy.jpg' # 'daisy.jpg'
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
  # print(file, _dataset_prediction)
  prediction_outcome = (prediction == _dataset_prediction)
  predictions.append(prediction_outcome)
  print('Filename:', file, '- TFLite tensor:', prediction_tensor, '- prediction label:', prediction_label, '- prediction:', prediction, '- isTrue?', prediction_outcome)

# calculate model evaluation metrics
predictions = np.array(predictions)
no_true_positives = len(predictions[predictions == 1])
tflite_accuracy = no_true_positives / len(predictions) # (no. of correct prediction)/(no. of all predictions)
print('TFLite accuracy: ', tflite_accuracy)