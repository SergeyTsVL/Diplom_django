import cv2
import numpy as np
from django.core.files.base import ContentFile
from .models import ImageFeed, DetectedObject
import time


VOC_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]




def process_image(image_feed_id):
    try:
        image_feed = ImageFeed.objects.get(id=image_feed_id)
        image_path = image_feed.image.path

        model_path = 'object_detection/mobilenet_iter_73000.caffemodel'
        config_path = 'object_detection/mobilenet_ssd_deploy.prototxt'
        net = cv2.dnn.readNetFromCaffe(config_path, model_path)

        img = cv2.imread(image_path)
        if img is None:
            print("Не удалось загрузить изображение.")
            return False

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                class_id = int(detections[0, 0, i, 1])
                class_label = VOC_LABELS[class_id]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"{class_label}: {confidence:.2f}"
                cv2.putText(img, label, (startX+5, startY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                DetectedObject.objects.create(
                    image_feed=image_feed,
                    object_type=class_label,
                    location=f"{startX},{startY},{endX},{endY}",
                    confidence=float(confidence)
                )

        result, encoded_img = cv2.imencode('.jpg', img)
        if result:
            content = ContentFile(encoded_img.tobytes(), f'processed_{image_feed.image.name}')
            image_feed.processed_image.save(content.name, content, save=True)

        return True

    except ImageFeed.DoesNotExist:
        print("Лента изображений не найдена.")
        return False

def process_video():
    source = cv2.VideoCapture(0)

    while True:

        # time.sleep(0.01)
        ret, img = source.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.Canny(img, 80, 100)
        faces = cv2.CascadeClassifier('detection_site/media/haarcascade_profileface.xml')
        results = faces.detectMultiScale(gray, scaleFactor=1.40, minNeighbors=3)

        for (x, y, w, h) in results:  # x, y, w, h - размеры квадрата выделяющего лица
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

        cv2.imshow("Result", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    source.release()