
import pandas as pd
import argparse
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pathlib
import numpy as np


def load_model(model, weights = None):
    my_model = tf.keras.models.load_model(model)
    my_model.summary()
    return my_model

def decode_img(img, img_height, img_width):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])

def get_image_and_label(df, classes, img_height, img_width):
  map = {}
  i=0
  class_list = list(classes)
  class_list.sort()
  for c in class_list:
    map[str(c).strip()]=i
    i=i+1
  test_path = df['image_path'].values
  test_labels = []
  test_images = []
  for c in df[' label'].values:
    test_labels.append(map[str(c).strip()])
  for pathVal in test_path:
    pathVal = str(os.getcwd())+'/'+pathVal
    img = tf.io.read_file(pathVal)
    img = decode_img(img, img_height, img_width)
    test_images.append(img)
  return tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(len(df))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Trasnfer Learning Test")
    parser.add_argument('--model', type=str, default='VXA200020_model.h5', help='Saved model')
    parser.add_argument('--weights', type=str, default=None, help='weight file if needed')
    parser.add_argument('--test_csv', type=str, default='flowers_test.csv', help='CSV file with true labels')

    args = parser.parse_args()
    model = args.model
    weights = args.weights
    test_csv = args.test_csv

    test_df = pd.read_csv(test_csv)
    classes = {'astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy','carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip'}
    my_model = load_model(model)
    test_dataset = get_image_and_label(test_df,classes,480,480)
    loss, acc = my_model.evaluate(test_dataset)
    print('Test model, accuracy: {:5.5f}%'.format(100 * acc))
