import tkinter as tk
from PIL import Image,ImageTk
from tkinter import filedialog
import os
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np


def model(path):

  # Disable scientific notation for clarity
  np.set_printoptions(suppress=True)

  # Load the model
  model = load_model('skin_cancer_model.h5',
                     compile=False)

  # Load the labels
  class_names = open("lab.txt", "r").readlines()

  # Create the array of the right shape to feed into the keras model
  # The 'length' or number of images you can put into the array is
  # determined by the first position in the shape tuple, in this case 1
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  # Replace this with the path to your image
  image = Image.open(path).convert("RGB")

  # resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  # turn the image into a numpy array
  image_array = np.asarray(image)

  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  # Load the image into the array
  data[0] = normalized_image_array

  # Predicts the model
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  # Print prediction and confidence score

  print("Class:", class_name[2:], end="")
  print(class_name[2:])
  print("Confidence Score:", confidence_score)

  label=tk.Label(window, text=f"{class_name[2:]}         ", bg="white",fg="black",font=("Arial",30))
  label.place(x=800,y=120)
  # text = canvas.create_text(850, 180, text=f"                 ", font=("Arial", 20), anchor=tk.CENTER, fill="black")
  # text=canvas.create_text(850,180,text=f"{class_name[2:]}",font=("Arial",20),anchor=tk.CENTER,fill="black")





def select():
  file=filedialog.askopenfile(mode="r",filetypes=[("Files","*.jpg")])
  if file:
    file_path=os.path.abspath(file.name)
    model(file.name)
    intial_image = Image.open(file_path)
    intial_photo = ImageTk.PhotoImage(image=intial_image)
    canvas.img = intial_photo
    canvas.create_image(760, 280, anchor=tk.NW, image=intial_photo)


# Create the main window
window = tk.Tk()
canvas=tk.Canvas(window,width=1400,height=720,bg="white")
canvas.pack(fill="both",expand=True)

button = tk.Button(window, text="Select_Image",bg="red", command=select)

# Pack the button widget onto the window
button.place(x=300,y=650,width=150,height=30)


intial_image=Image.open("cancer1.jpg")
intial_photo=ImageTk.PhotoImage(image=intial_image)
canvas.img=intial_photo
canvas.create_image(0,0,anchor=tk.NW,image=intial_photo)

# Start the Tkinter event loop
window.mainloop()