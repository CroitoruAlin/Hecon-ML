from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
import cv2
import tensorflow as tf
graph = tf.get_default_graph()

class Prediction(Resource):
    def __init__(self, model):
        self.model = model
    def post(self): 
        parser = reqparse.RequestParser()
        parser.add_argument('name')
        fields = parser.parse_args()
        img = cv2.imread(fields['name'])
        print(fields['name'])
        img = cv2.resize(img, (224,224))
        if img.shape[2] ==1:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        img_array=[]
        img_array.append(img)
        img_array = np.array(img_array)
        global graph
        with graph.as_default():
            img_array = self.model.predict(img_array)
        output = {'positive_probability':str(img_array[0][1]),'negative_probability':str(img_array[0][0])}
        return output