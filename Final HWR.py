from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
 
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
 
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
 
test_loss, test_acc = model.evaluate(test_images, test_labels)
 
print(test_acc)
 
model.save('mnist.h5')

from tkinter import *
 
import cv2
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model
import webbrowser
 
model = load_model('mnist.h5')
image_folder = "img/"
 
root = Tk()
root.resizable(0, 0)
root.title("Digit Recognition System")
 
lastx, lasty = None, None
image_number = 0
 
cv = Canvas(root, width=800, height=600, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=NSEW, columnspan=2)
 
 
def clear_widget():
    global cv
    cv.delete('all')
 
 
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y
 
 
def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y
 
 
cv.bind('<Button-1>', activate_event)
 
 
def Recognize_Digit():
    global image_number
    filename = f'img_{image_number}.png'
    widget = cv
 
    x = root.winfo_rootx() + widget.winfo_rootx()
    y = root.winfo_rooty() + widget.winfo_rooty()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()
    print(x, y, x1, y1)
 
    # get image and save
    ImageGrab.grab().crop((x, y, x1, y1)).save(image_folder + filename)
 
    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
 
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # make a rectangle box around each curve
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
 
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = th[y:y + h, x:x + w]
 
        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))
 
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
 
        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0
 
        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)
 
        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'
 
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)
 
    cv2.imshow('Predictions', image)
    cv2.waitKey(0)
    
def callback():
        webbrowser.open_new(r"www.google.com")    
btn_save = Button(text='Recognize Digit',width=15, height=3, command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text='Clear Output',width=15, height=3, command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)
button_info = Button(text='Feedback', width=15, height=2, command=callback)
button_info.grid(row=3, column=0, pady=1, padx=1)
root.mainloop()
