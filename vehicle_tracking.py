import argparse
from ultralytics import YOLO
import cv2
import os, json
import math
from pathlib import Path

def track_vehicle_trajectories_from_frames(frames_dir_path, fps=30):
    print(f"Starting Processing for frames in {frames_dir_path}")
    frames_dir = Path(frames_dir_path)

    if not frames_dir.exists() or not frames_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frames_dir_path}")

    # Load image paths and sort them by name
    image_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not image_files:
        raise ValueError("No image files found in directory.")

    model_path = './yolo_weights/yolov8m.pt'
    model = YOLO(model_path)

    vehicle_classes = ['car', 'truck', 'bus']
    vehicle_class_indices = [i for i, name in model.names.items()
                             if name.lower() in [cls.lower() for cls in vehicle_classes]]

    vehicle_trajectories = {}

    for frame_count, image_path in enumerate(image_files, start=1):
        timestamp = frame_count / fps  # seconds
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: could not read {image_path}")
            continue

        results = model.track(frame, persist=True, tracker="./vehicle_tracking/botsort.yaml", verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                if int(cls) not in vehicle_class_indices:
                    continue

                x1, y1, x2, y2 = [int(coord.item()) for coord in box]
                area = (x2 - x1) * (y2 - y1)
                if area < 5000:  # Skip small boxes
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                box_coords = [x1, y1, x2, y2]

                if track_id not in vehicle_trajectories:
                    vehicle_trajectories[track_id] = []

                vehicle_trajectories[track_id].append((timestamp, box_coords))

        cv2.imshow('Vehicle Tracking Validation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    return vehicle_trajectories

def track_vehicle_trajectories(input_video_path):
    print(f"Starting Processing for {input_video_path}")
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    model_path = '../yolo_weights/yolo11m.pt'
    model = YOLO(model_path)

    vehicle_classes = ['car', 'truck', 'bus']
    vehicle_class_indices = [i for i, name in model.names.items()
                             if name.lower() in [cls.lower() for cls in vehicle_classes]]

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    vehicle_trajectories = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        timestamp = frame_count / fps  # seconds

        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                if int(cls) not in vehicle_class_indices:
                    continue
                print(f"Box: {box}, Track ID: {track_id}, Coordinates: {box.tolist()}, Class: {model.names[int(cls)]}")

                x1, y1, x2, y2 = [int(coord.item()) for coord in box]
                area = (x2 - x1) * (y2 - y1)
                if area < 5000:  # Skip small boxes
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                box_coords = [x1, y1, x2, y2]

                if track_id not in vehicle_trajectories:
                    vehicle_trajectories[track_id] = []

                vehicle_trajectories[track_id].append((timestamp, box_coords))

        cv2.imshow('Vehicle Tracking Validation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return vehicle_trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track vehicle trajectories from video or image frames.")
    parser.add_argument("--video_path", type=str, help="Path to the input video file")
    parser.add_argument("--frames_dir", type=str, help="Directory containing sequential image frames")
    parser.add_argument("--fps", type=int, default=30, help="FPS for frames-based input (default: 30)")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save output JSON file")
    args = parser.parse_args()

    if args.frames_dir:
        tracks = track_vehicle_trajectories_from_frames(args.frames_dir, args.fps)
        base_name = os.path.basename(os.path.normpath(args.frames_dir))  # folder name
    elif args.video_path:
        tracks = track_vehicle_trajectories(args.video_path)
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]  # video filename without extension
    else:
        raise ValueError("Please provide either --video_path or --frames_dir")

    # If output filename not provided, create one based on input
    if args.output_json is None:
        args.output_json = "output_tracking/" + base_name + ".json"

    # Convert tuples to lists for JSON serialization
    serializable_tracks = {
    str(track_id): [[round(t, 2), [int(x1), int(y1), int(x2), int(y2)]] for t, (x1, y1, x2, y2) in traj]
    for track_id, traj in tracks.items()
    }

    # Save to JSON file
    output_dir = os.path.dirname(args.output_json)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(serializable_tracks, f, indent=4)

    print(f"Tracking results saved to {args.output_json}")

    # Optional: print summary
    for track_id in tracks:
        print(f"\nVehicle ID {track_id}:")
        for t, box_coords in tracks[track_id]:
            print(f"  Time: {t:.2f}s, Position: {box_coords}")