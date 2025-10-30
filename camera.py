import cv2 
import matplotlib.pyplot as plt
img = cv2.imread("opencv-test.jpg")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3)
print(f"no of faces found = {len(faces)}")
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    face_img = img[y:y+h, x:x+w]

face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)
     
plt.imshow(face_img)
plt.axis("off")
plt.show()

# positives -> folder -> images having face
# negatives -> folder => imges havin non face
# description files
# positives.txt => path/to/image/file
# negatives.txt -> path/to/negative / image 
# edge, color, intensity , line
# cascade classifier  no of predefined classes , each classes is stricter than 
# previous one
