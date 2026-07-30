import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import zipfile

# ====== SETUP ======
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

target_path = "C:/Users/Dnyaneshwar/Desktop/prof.jpg"
target_img = Image.open(target_path).convert('RGB')
target_face = mtcnn(target_img)

if target_face is None:
    raise ValueError("No face detected in target image!")

target_embedding = resnet(target_face[0].unsqueeze(0).to(device)).detach()

# ====== INIT ======
threshold = 0.6  # Lowered for better match sensitivity
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Unable to access webcam.")
    exit()

matched_faces_dir = "matched_faces"
matched_full_frame_dir = "matched_full_frames"
os.makedirs(matched_faces_dir, exist_ok=True)
os.makedirs(matched_full_frame_dir, exist_ok=True)

frame_count = 0
matched_full_frames = []
matched_images = []

print("🔍 Real-time face recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    boxes, _ = mtcnn.detect(pil_img)
    faces = mtcnn(pil_img)

    matched = False
    if faces is not None and boxes is not None:
        embeddings = resnet(faces.to(device)).detach()
        for i, emb in enumerate(embeddings):
            similarity = torch.nn.functional.cosine_similarity(target_embedding, emb.unsqueeze(0)).item()

            if similarity > threshold:
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Match {similarity:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Save matched face
                face_crop = frame[y1:y2, x1:x2]
                face_path = os.path.join(matched_faces_dir, f"face_{frame_count}.jpg")
                cv2.imwrite(face_path, face_crop)
                matched_images.append(face_path)

                # Save full frame
                full_frame_path = os.path.join(matched_full_frame_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(full_frame_path, frame)
                matched_full_frames.append(full_frame_path)
                print(f"✅ Match saved at frame {frame_count}")
                matched = True
                break

    if not matched:
        print(f"❌ No match at frame {frame_count}")

    cv2.imshow("Face Recognition", frame)

    # Wait for key and quit if 'q' pressed
    key = cv2.waitKey(10)
    if key == ord('q'):
        print("🛑 'q' pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

# ====== VIDEO GENERATION ======
output_video_path = "C:/Users/Dnyaneshwar/Desktop/matched_output.mp4"

if matched_full_frames:
    print("🎥 Creating matched video...")
    sample_frame = cv2.imread(matched_full_frames[0])
    if sample_frame is None:
        print("❌ Error reading sample frame.")
    else:
        height, width = sample_frame.shape[:2]
        print(f"🎞️ Video resolution: {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

        if not out.isOpened():
            print("❌ Failed to open video writer.")
        else:
            for img_path in matched_full_frames:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Couldn't read {img_path}, skipping...")
                    continue
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))
                out.write(img)

            out.release()
            print(f"✅ MP4 video created: {output_video_path}")
else:
    print("⚠️ No matched frames. Video not generated.")

# ====== ZIP MATCHED IMAGES ======
if matched_images:
    zip_path = "matched_faces.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img_path in matched_images:
            zipf.write(img_path, arcname=os.path.basename(img_path))
    print(f"🗜️ Matched faces zipped: {zip_path}")
else:
    print("⚠️ No matched faces to zip.")
