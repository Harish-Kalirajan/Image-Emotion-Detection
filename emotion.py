import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
img=cv2.imread("/content/emotion.jpeg")
plt.imshow(img)
predictions=DeepFace.analyze(img)
print(predictions)
cv2.putText(img,predictions[0]['dominant_emotion'],(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
