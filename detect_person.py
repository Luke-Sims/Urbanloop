import cv2
import time
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
INPUT_VIDEO = "rail.avi"
OUTPUT_VIDEO = "person_output.mp4"

CONF_THRESHOLD = 0.5
RESIZE_WIDTH = 640


def load_model(model_path):
    model = YOLO(model_path)
    print(f"Using device: {model.device}")
    return model


def preprocess_frame(frame):
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

    return frame

def detect_persons(model, frame):

    results = model.predict(frame, verbose=False)

    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls == 0 and conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0]
                detections.append(
                    ([int(x1), int(y1), int(x2), int(y2)], conf)
                )

    return detections


def draw_detections(frame, detections):

    for bbox, confidence in detections:
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"PERSON: {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return frame


def main():
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Error: Cannot open rail.avi")
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = RESIZE_WIDTH / width
    resized_height = int(height * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        input_fps,
        (RESIZE_WIDTH, resized_height),
    )

    frame_count = 0
    start_time = time.time()

    print("Processing rail.avi")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)

        detections = detect_persons(model, processed_frame)

        processed_frame = draw_detections(processed_frame, detections)

        out.write(processed_frame)

        frame_count += 1

        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            print(f"\rProcessing FPS: {fps:.2f}", end="")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nDone. Output saved as:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
