import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
print(tf.__version__)

mnist=tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test)=mnist.load_data()

x_train,x_test=x_train/255.0,x_test/255.0

model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10,activation='softmax')

])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


model.fit(x_train,y_train,epochs=5)

test_loss,test_acc= model.evaluate(x_test,y_test,verbose=2)

print('\nTest accuracy:', test_acc)

predictions=model.predict(x_test)



def plot_image(i,predictions_array,true_label,img):
    true_label,img=true_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])


    plt.imshow(img,cmap=plt.cm.binary)

    predicted_label=np.argmax(predictions_array)
    if predicted_label==true_label:
        color='blue'
    else:
        color='red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,100*np.max(predictions.array),true_label),color=color)


    num_rows=5
    num_cols=3
    num_images=num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols,2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows,2*num_cols,2*i+1)
        plot_image(i,predictions[i],y_test,x_test)

    plt.tight_layout()
    plt.show()
 