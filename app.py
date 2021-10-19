from flask import Flask, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('saved_model.h5')
paths_zero = ['/Users/trent/Downloads/resources/IDC_regular_ps50_idx5/9322/0/9322_idx5_x951_y751_class0.png',
             '/Users/trent/Downloads/resources/IDC_regular_ps50_idx5/14210/0/14210_idx5_x1801_y1851_class0.png']
paths_one = ['/Users/trent/Downloads/resources/IDC_regular_ps50_idx5/9383/1/9383_idx5_x1951_y1051_class1.png',
            '/Users/trent/Downloads/resources/IDC_regular_ps50_idx5/12883/1/12883_idx5_x401_y251_class1.png']
output_list = []
for path in paths_zero:
    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    if(image_arr.shape == (50,50,3)):
        X_test = image_arr
        X_test_scaled = X_test/255
        X_test_scaled = X_test_scaled.reshape(1,50,50,3)
        y_pred = model.predict(X_test_scaled)
        print()
        if y_pred.item(0) > .5:
            output_list.append('Predicts 0, actual is 0. Yay!')
        else:
            output_list.append('Predicts 1, actual is 0. Noooo!')
    else:
        output_list.append('no prediction due to shape')
for path in paths_one:
    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    if(image_arr.shape == (50,50,3)):
        X_test = image_arr
        X_test_scaled = X_test/255
        X_test_scaled = X_test_scaled.reshape(1,50,50,3)
        y_pred = model.predict(X_test_scaled)
        print()
        if y_pred.item(0) > .5:
            output_list.append('Predicts 0, actual is 1. Noooo!')
        else:
            output_list.append('Predicts 1, actual is 1. Yay!')
    else:
        output_list.append('no prediction due to shape')

@app.route("/")
def index():
    title = 'Breast cancer predictor'
    paragraph = 'predicts based on an image if the sample is malignant (1) or benign 0. (Doesn\'t currently accept input. That feauture will be added later. For now, it takes a given input of 4 images, two positive and two negative'
    return render_template('index.html',title=title,paragraphs=[paragraph],list_items=output_list)


