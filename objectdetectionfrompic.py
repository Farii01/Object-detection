# detecting from pictures
import cv2

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('detect.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
# parameters (grayscale, scalefactor, minimum neighbors)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces (haarcascade)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()
