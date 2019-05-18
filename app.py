from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
from prediction import Prediction
from build_model import get_model

app = Flask(__name__)
api = Api(app)


model = get_model()

api.add_resource(Prediction, '/',resource_class_kwargs={'model': model})
if __name__ == '__main__':
    app.run(debug=True)






