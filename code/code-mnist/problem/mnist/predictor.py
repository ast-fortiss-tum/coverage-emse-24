# For Python 3.6 we use the base keras
import keras
#from tensorflow import keras

import numpy as np

from problem.mnist.properties import MODEL, EXPECTED_LABEL, num_classes

# Load the pre-trained model.
model = keras.models.load_model(MODEL)
print("Loaded model from disk")

class Predictor:

    @staticmethod
    def predict(img):
        explabel = (np.expand_dims(EXPECTED_LABEL, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, num_classes)

        # print(f"Explabel {explabel}")
        explabel = np.argmax(explabel.squeeze())

         #Predictions vector
        predictions = model.predict(img)

        prediction1, prediction2 = np.argsort(-predictions[0])[:2]

        # print(f"Predicted vs expected: {prediction1} / {explabel}")
        # print(f"EXPECTED_LABEL: {EXPECTED_LABEL}")

        
        # Activation level corresponding to the expected class
        confidence_expclass = predictions[0][explabel]

        if prediction1 != EXPECTED_LABEL:
            confidence_notclass = predictions[0][prediction1]
        else:
            confidence_notclass = predictions[0][prediction2]

        confidence = confidence_expclass - confidence_notclass

        return prediction1, confidence

    ''' Return the list of all predictions (softmax layer values)'''
    @staticmethod
    def predict_extended(img):
        explabel = (np.expand_dims(EXPECTED_LABEL, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, num_classes)
        explabel = np.argmax(explabel.squeeze())

         #Predictions vector
        predictions = model.predict(img)

        # print(f"Predictions {predictions}")

        
        return predictions[0]
