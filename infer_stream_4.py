from ultralytics import YOLO
import cv2
import datetime
import requests
import json
import os
import pandas as pd
import json
from collections import defaultdict
from image2base64.converters import rgb2base64
import datetime
import pafy
from image2base64.converters import rgb2base64

url_api = "https://dashboard.tdemo.biz/api/violations"
model = YOLO("latest_best_22.pt")

##RTSP Stream: 'rtsp://192.168.18.194:1935'


url = "https://youtu.be/PFfluLWsabk?si=A5G_SourhTh8msMc"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
#best.url


cap = cv2.VideoCapture('worker_1.mp4')

link = 'local CCTV Stream Lahore Office'
headers = {"Content-Type": "application/json"}


detection_consistency = {}
all_classes = ['gloves', 'helmet', 'person', 'shoe', 'vest']
class_names = {0: 'gloves', 1: 'helmet', 2: 'person', 3: 'shoe', 4: 'vest'}

def calculate_percentage(bbox, original_shape):
    bbox_area = (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])
    original_shape_area = original_shape[0] * original_shape[1]
    percentage = (bbox_area / original_shape_area) * 100
    return percentage


frame_count = {}
max_missing_frames = 20
save_path = r"C:\YOLO_v8\detection"

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, conf=0.5, show=False, show_conf=False, show_labels=False, persist=True, tracker="bytetrack.yaml")
        timestamp = datetime.datetime.now()

        person_boxes = []
        other_boxes = {}

        class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        boxv={}

        for box in results[0].boxes:
            class_id = int(box.cls)
            pc = box.xywh            
            x_center, y_center, width, height = pc[0].tolist()

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)

            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), class_colors[class_id], 1)


            
            if class_id == 2:
                if box.id !=None:
                    
                    box_id = box.id.tolist()
                    
                    cv2.putText(frame, f"ID: {box.id[0]}", (x1+30, y1 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    person_boxes.append((x1, y1, x2, y2, box_id[0]))
                

            else:
                if class_id not in other_boxes:
                    other_boxes[class_id] = []
                other_boxes[class_id].append((x1, y1, x2, y2))
                

        for person_box in person_boxes:
            x1, y1, x2, y2, object_id = person_box
            text_offset = 10


            box_key = object_id
            
            
            text_lines = []
            missing_classes = defaultdict(list) #Define


            
            for class_id, class_name in enumerate(all_classes):
                if class_id == 2:
                    continue

                intersection = False
                for box in other_boxes.get(class_id, []):
                    bx1, by1, bx2, by2 = box
                    if x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1:
                        intersection = True
                        break
                
                if intersection:
                    text_line = f"{class_name}: OK"
                    
                  
                else:
                    text_line = f"{class_name}: Missing"
                    missing_classes[object_id].append(class_name) #Define


                    # Increment missing frame count for this person object ID
                    frame_count.setdefault(box_key, {}).setdefault(class_id, 0)
                    frame_count[box_key][class_id] += 1

                    # Check if missing frames exceed the threshold
                    if frame_count[box_key][class_id] >= max_missing_frames:
                        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        # Save cropped person bounding box
                        cropped_person = frame[y1:y2, x1:x2]
                        file_name = f"person_{timestamp_str}_{object_id}_{x1}_{y1}_{x2}_{y2}.jpg"

                        cv2.imwrite(f"{save_path}/{file_name}", cropped_person)


                        # Reset the frame count for this person object ID and class
                        frame_count[box_key][class_id] = 0

                        image_base64 = rgb2base64(cropped_person)


                        boxv["name"] = str(missing_classes[object_id]) #Define
                        boxv["image_path"] = image_base64
                        boxv["input"] = "http://testings.new"
                        boxv["timestamp"] = timestamp_str
                        boxv["track_id"] = object_id

                        

                        print(boxv)

                        response = requests.post(url_api, headers=headers, json=boxv)
                        if response:
                            print("Request successful!")
                            boxv.clear()
                        else:
                            print("Request failed with status code:", response.status_code)
                        
                                            
                text_lines.append(text_line)


            for i, text_line in enumerate(text_lines):
                color = (255,0,0) if "OK" in text_line else (0, 0, 0)  # White for Present, Red for Missing
                cv2.putText(frame, text_line, (x1, y1-34 + i * text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
   


        cv2.imshow("Object Detection", frame)

    else:
        break


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
