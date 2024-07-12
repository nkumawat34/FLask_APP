from flask import Flask,request
from flask_cors import CORS

app = Flask(__name__)
import qrcode
import requests
import face_recognition
import cv2
import os
import jsonify
import PyPDF2

import os
from io import BytesIO
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Specify the allowed origin
#they are used for send data from server to client and client to server.They are highly used for api's callings
cors=CORS(app)


@app.route('/', methods = ['GET', 'POST']) 
def home(): 
    if(request.method == 'GET'): 
  
        data = "hello world"
        return data

@app.route('/myfunction', methods=['GET'])
def hello():
    # Access query parameters
    document_cid = request.args.get('param2')
    document_name = request.args.get('param1')
    print(document_cid,document_name)
    # Check if the parameters are present
   

    # Your code to process the query parameters goes here
    result = my_function(document_cid+".ipfs.w3s.link"+"/"+document_name)

    # Return a response
    return "Done"

def my_function(url):
  
  # Replace this with your IPFS CID
  ipfs_cid = url
  
  # Create a QR code
  qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
  qr.add_data(ipfs_cid)
  qr.make(fit=True)

  # Create an image of the QR code
  img = qr.make_image(fill_color="black", back_color="white")

  # Save or display the image
  img.save("IPFS"+".png")  # Save the image to a file
  img.show()  # Display the image
  return "Done"




@app.route('/liveface', methods=['GET'])
def live_face():
  document_cid = request.args.get('param1')
  document_name = request.args.get('param2')
  print(document_cid)
  # URL of the image you want to compare against
  
  image_url = "https://ipfs.io/ipfs/"
  image_url+=str(document_cid)
  image_url+='/'
  image_url+=str(document_name)
  print(image_url)
# Download the image from the URL
  response = requests.get(image_url)
  
  known_image = face_recognition.load_image_file(BytesIO(response.content))
  known_face_encoding = face_recognition.face_encodings(known_image)[0]

  # Initialize the webcam
  cap = cv2.VideoCapture(0)
  flag=0
  while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading the webcam feed.")
        break

    # Find face locations in the live frame
    face_locations = face_recognition.face_locations(frame)
    
    if len(face_locations) > 0:
        # Encode the live face
        live_face_encoding = face_recognition.face_encodings(frame, face_locations)[0]

        # Compare the live face to the known face
        results = face_recognition.compare_faces([known_face_encoding], live_face_encoding)

        if results[0]:
            print("Match found! The live face matches the known face.")
            flag=1
            break
            
        else:
            print("No match found. The live face does not match the known face.")
            

    cv2.imshow('Live Face Comparison', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  if flag==1: 
    return "Yes"
  else:
    return "No"       
  cap.release()
  cv2.destroyAllWindows()

@app.route('/encryptpdf', methods=['GET'])
def encrypt_pdf():

  input_pdf=request.args.get("param1")
  output_pdf=request.args.get("param2")
  password=request.args.get("param3")

  with open(input_pdf, 'rb') as file:
      pdf_reader = PyPDF2.PdfReader(file)
      pdf_writer = PyPDF2.PdfWriter()

      for page_num in range(len(pdf_reader.pages)):
          pdf_writer.add_page(pdf_reader.pages[page_num])

      pdf_writer.encrypt(password)

      with open(output_pdf, 'wb') as output_file:
          pdf_writer.write(output_file)

  return "Done"

@app.route('/document_classification',methods=['GET'])
def document_classification():

  document_name=request.args.get("param1")

  # Define paths
  dataset_path = 'dataset'  # Path to your dataset directory
  genuine_path = os.path.join(dataset_path, 'genuine')
  forged_path = os.path.join(dataset_path, 'forged')

  # Load genuine images
  genuine_images = []
  for filename in os.listdir(genuine_path):
    img = tf.keras.preprocessing.image.load_img(
    os.path.join(genuine_path, filename), target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    genuine_images.append(img_array)
  genuine_images = np.array(genuine_images)
  genuine_labels = np.ones(len(genuine_images))  # Assign label 1 for genuine documents

  # Load forged images
  forged_images = []
  for filename in os.listdir(forged_path):
    img = tf.keras.preprocessing.image.load_img(
    os.path.join(forged_path, filename), target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    forged_images.append(img_array)
  forged_images = np.array(forged_images)
  forged_labels = np.zeros(len(forged_images))  # Assign label 0 for forged documents

# Concatenate genuine and forged images and labels
  all_images = np.concatenate([genuine_images, forged_images], axis=0)
  all_labels = np.concatenate([genuine_labels, forged_labels], axis=0)

# Split the dataset into training and testing sets
  train_data, test_data, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42)

  # Define the CNN model
  model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

  # Compile the model
  model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
  history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)
  # Load and preprocess the document image
  document_path = 'forged-type01-001.tif'  # Replace 'path_to_your_document.jpg' with the path to your document image
  img = load_img(document_path, target_size=(150, 150))
  img_array = img_to_array(img)
  img_array = img_array / 255.0  # Normalize pixel values
  document = np.expand_dims(img_array, axis=0)

  # Load the trained model
  #model = tf.keras.models.load_model('path_to_your_trained_model.h5')  # Replace 'path_to_your_trained_model.h5' with the path to your trained model file

  # Make prediction
  prediction = model.predict(document)[0][0]
  predicted_label = round(prediction)

  # Define class names
  class_names = ['Forged', 'Genuine']

  # Display the document image and prediction result
  plt.imshow(img)  # Display the document image
  plt.title(f'Predicted: {class_names[int(predicted_label)]}')
  plt.axis('off')
  plt.show()

  # Evaluate the model
  #test_loss, test_acc = model.evaluate(test_data, test_labels)
#print('Test accuracy:', test_acc*100)

app.run()