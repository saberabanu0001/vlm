import cv2
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# Load a lightweight Visual Language Model (BLIP)
model_name = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name)

# Open your MacBook webcam
cap = cv2.VideoCapture(0)
print("🎥 VLM live test started — press 'q' to quit")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Every 60 frames (~2 s) create a caption
    if frame_count % 60 == 0:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)

        caption = processor.batch_decode(out, skip_special_tokens=True)[0]
        print(f"🧠 {caption}")

        # Display caption on frame
        cv2.putText(frame, caption, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("VLM Webcam Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
