# FOYDALANISH
# python maska_detec.py

# kerakli paketlarni import qilish
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# ramkaning o'lchamlarini va blok qo'ying
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (500, 500),
		(104.0, 177.0, 123.0))

	# blobni set orqali o'tkazish va yuzni aniqlash
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# yuzlar ro'yxatini, ularning tegishli joylarini aniqlash,
        # va yuz maskalari ajratish
	faces = []
	locs = []
	preds = []

	
	for i in range(0, detections.shape[2]):
	
		# aniqlash
		confidence = detections[0, 0, i, 2]

		# ishonchni hosil qilish uchun aniqlanishlarni filtrlash
		if confidence > args["confidence"]:
			# chegara maydonining (x, y) - koordinatalarini hisoblash
			# ob'ekt
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# cheklov ramka o'lchamlariga mos kelishini ta'minlash			
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# yuz ROI-ni chiqarib olish, uni BGR-dan RGB-ga o'tkazish
			# buyurtma qilish, o'lchamini 224x224 ga o'zgartirish va oldindan qayta ishlash
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# yuz va cheklov ramkasini o'ziga qo'shish
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# faqat kamida bitta yuz aniqlanganda ishga tushurish
	if len(faces) > 0:

		# tezroq xulosa qilish uchun biz * hammasi * bo'yicha umumiy aniqlashtirish qilamiz
		# yuzlarni birma-bir aniqlashdan ko'ra bir vaqtning o'zida 

		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	
	# joylar
	return (locs, preds)

# argument tahlilini tuzish va argumentlarni tahlil qilish
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# seriyali yuz detektori modelimizni diskdan yuklash
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# yuz niqobi detektori modelini diskdan yuklash
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# video kamerani ishga tushirish 
print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)

# video oqimidan kadrlar bo'ylab 
while True:
	# freymni olish va o'lchamini o'zgartirish
	# maksimal kengligi 400 piksel bo'lishi kerak
	
	frame = vs.read()
	frame = imutils.resize(frame, width=1090)

	# kadrdagi yuzlarni aniqlash va ularning yopilganligini aniqlash
	# yuz niqobi bor yoki yo'q
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# aniqlangan yuz joylari va ularga mos keladigan ramka
	# joylar
	for (box, pred) in zip(locs, preds):
		# chegara ramkasini aniqlash va kiritish
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# biz chizish uchun foydalanadigan ramkani va rangini aniqlang
		# cheklovchi ramka va matni
		label = "Maskali" if mask > withoutMask else "Maskasiz"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# ehtimollikni ramkaga kiritish
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# chiqishda nomini va chegara ramkasini ko'rsatish
		# ramka
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# chiqish ramkasini ko'rsatish
	cv2.imshow("Maska_Detec", frame)
	key = cv2.waitKey(1) & 0xFF

	# agar "q`" tugmachasi bosilgan bosilsaa, chiqish
	if key == ord("q"):
		break

# hamma oynani tozalash
cv2.destroyAllWindows()
vs.stop()
