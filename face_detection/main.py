import cv2
import mediapipe as mp
import os
import argparse

def process_img(img, face_detection):
    H, W, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    out = face_detection.process(img_rgb)
    
    print(out.detections)
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            
            # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)
            img[y1:y1 + h, x1: x1 + w, :] = cv2.blur(img[y1:y1 + h, x1: x1 + w, :], (30,30))
    return img

args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)
args = args.parse_args()



output_dir = "./output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
    if args.mode in ["image"]:
        img  = cv2.imread(args.filePath)
        
        img = process_img(img, face_detection)
                
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)
    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)
        
        # Kiểm tra xem video có được mở thành công không
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video tại đường dẫn {args.filePath}")
            exit()
            
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Lỗi: Không thể đọc frame đầu tiên từ video")
            cap.release()
            exit()
            
        output_video = cv2.VideoWriter(os.path.join(output_dir, "output.mp4"), 
                                     cv2.VideoWriter_fourcc(*'MP4V'), 
                                     25, 
                                     (frame.shape[1], frame.shape[0]))
        
        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame) 
            ret, frame = cap.read()
            
        cap.release()
        output_video.release()  
    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        
        ret, frame = cap.read()
        output_video = cv2.VideoWriter(os.path.join(output_dir, "output.mp4"), 
                                     cv2.VideoWriter_fourcc(*'MP4V'), 
                                     25, 
                                     (frame.shape[1], frame.shape[0]))
        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow("frame", frame)
            cv2.waitKey(25)
            
            ret, frame = cap.read()
            
        cap.release()
