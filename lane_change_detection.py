import os, re, json, torch, cv2, tqdm
import numpy as np
from utils.common import merge_config, get_model
from data.dataset import LaneTestDataset
from utils.dist_utils import dist_print
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from process_video import process_video
from collections import defaultdict

def robust_fit_line(points, img_height=1080, upper_portion_bound=0.48, lower_portion_bound=0.9):
    if len(points) < 20:
        return None 

    points_np = np.array(points, dtype=np.float32)
    x = points_np[:, 0]
    y = points_np[:, 1]

    X = x.reshape(-1, 1)
    y_target = y          

    try:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=10.0,   # noise level
            max_trials=100,
            random_state=42
        )
        ransac.fit(X, y_target)
        m = ransac.estimator_.coef_[0]
        c = ransac.estimator_.intercept_

        y_fit_min = np.min(y)
        y_fit_max = np.max(y)

        y_ext_bound = int(img_height * upper_portion_bound)
        
        if y_ext_bound > y_fit_min:
            y_ext_bound  = None
        
        return {
            'coefficients': [float(m), float(c)], 
            'y_bounds': [int(y_fit_min), int(y_fit_max)],
            'extrapolated_y_range': None if y_ext_bound is None else [int(y_fit_min), int(y_ext_bound)]
        }

    except Exception as e:
        print(f"[RANSAC Fit Failed] {e}")
        return None

def deduplicate_lane_changes(events, threshold=0.6):
    """
    Groups lane change events that occur within `threshold` seconds
    for the same vehicle (track_id), and keeps only the first event.
    """
    # Group by track_id
    grouped = defaultdict(list)
    for event in sorted(events, key=lambda x: (x["track_id"], x["timestamp"])):
        grouped[event["track_id"]].append(event["timestamp"])

    deduped = []
    for track_id, timestamps in grouped.items():
        group_start = timestamps[0]
        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i - 1] > threshold:
                deduped.append({"timestamp": round(group_start, 2), "track_id": track_id})
                group_start = timestamps[i]
        deduped.append({"timestamp": round(group_start, 2), "track_id": track_id})
    return deduped

def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1920, original_image_height=1080):
    """
    Convert model predictions to pixel coordinates.
    """
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    _, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()
    cfg.batch_size = 1
    dist_print("start testing...")
    video_path = cfg.video_path
    if video_path is None:
        raise ValueError("❌ No video path provided. Please specify --video_path argument.")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    dist_print(f"Processing video: {video_name}")
    process_video(video_path)

    if cfg.backbone not in ['18','34','50','101','152','50next','101next','50wide','101wide']:
        raise ValueError("❌ Unsupported backbone specified.")

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        splits = [video_name + '.txt']
        img_w, img_h = 1920, 1080
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, 'lists', split),
                                    img_transform=None, crop_size=cfg.train_height) for split in splits]

    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        splits = [video_name + '.txt']
        img_w, img_h = 1920, 1080
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, split),
                                    img_transform=None, crop_size=cfg.train_height) for split in splits]

    elif cfg.dataset == 'CurveLanes':
        cls_num_per_lane = 72
        splits = [video_name + '.txt']
        img_w, img_h = 1920, 1080
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, 'lists', split),
                                    img_transform=None, crop_size=cfg.train_height) for split in splits]
    else:
        raise NotImplementedError("❌ Unknown dataset: {}".format(cfg.dataset))

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = get_model(cfg)
    state_dict = torch.load(cfg.test_model, map_location=device)['model']
    net.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    net.to(device)
    net.eval()
    print(f'✅ Model loaded on {device}')

    # yolo model for vehicle tracking
    yolo_model = YOLO('./yolo_weights/yolov8m.pt')  # Update to correct model path
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    vehicle_class_indices = [i for i, name in yolo_model.names.items()
                            if name.lower() in [cls.lower() for cls in vehicle_classes]]

    vehicle_trajectories = {}

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for dataset in datasets:
        dataset.img_transform = img_transforms

    for split, dataset in zip(splits, datasets):
        lane_results = []
        lane_change_events = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        video_name = os.path.splitext(os.path.basename(split))[0]
        output_dir = os.path.join("output", video_name)
        os.makedirs(output_dir, exist_ok=True)

        avi_path= os.path.join(output_dir, f"{video_name}_output.avi")
        json_path = os.path.join(output_dir, f"{video_name}_lane_change.json")

        vout = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (img_w, img_h))
        if not vout.isOpened():
            raise RuntimeError("❌ VideoWriter failed to open. Check codec and frame size.")

        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.to(device)
            with torch.no_grad():
                pred = net(imgs)

            img_path = os.path.join(cfg.data_root, names[0])
            vis = cv2.imread(img_path)
            coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=img_w, original_image_height=img_h)

            # Timestamp extraction
            basename = os.path.basename(names[0])
            match = re.search(r"frame_(\d+)_(\d+)s\.jpg", basename)
            if match:
                seconds = int(match.group(1))
                fraction = int(match.group(2)) / 100
                timestamp = round(seconds + fraction, 2)
            else:
                timestamp = i / 30.0

            # Draw fitted lanes (original and extrapolated)
            fitted_lanes = {
                "timestamp": timestamp,
                "filename": names[0],
                "lanes": []}
            for idx, lane in enumerate(coords):
                fit_result = robust_fit_line(lane)

                if fit_result is not None:
                    fitted_lanes["lanes"].append(fit_result)
                    coeffs = fit_result['coefficients']
                    y_min, y_max = fit_result['y_bounds']
                    y_ext_range = fit_result['extrapolated_y_range']

                    # Draw original fitted line (yellow)
                    y_vals = np.linspace(y_min, y_max, num=100)
                    x_vals = (y_vals - coeffs[1]) / coeffs[0]  # solve x = (y - c) / m
                    line_pts = np.stack([x_vals, y_vals], axis=1).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [line_pts], isClosed=False, color=(0, 255, 255), thickness=2)  # yellow (BGR)

                    # Draw extrapolated portion (red)
                    if y_ext_range is not None:
                        y_ext_min, y_ext_max = y_ext_range
                        y_vals_ext = np.linspace(y_ext_min, y_ext_max, num=100)
                        x_vals_ext = (y_vals_ext - coeffs[1]) / coeffs[0]
                        ext_line_pts = np.stack([x_vals_ext, y_vals_ext], axis=1).astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(vis, [ext_line_pts], isClosed=False, color=(0, 0, 255), thickness=2)  # red (BGR)
            if fitted_lanes["lanes"]:
                lane_results.append(fitted_lanes)

            for lane in coords:
                for coord in lane:
                    cv2.circle(vis, coord, 5, (0, 255, 0), -1)
            
            # --- Vehicle Tracking ---
            yolo_results = yolo_model.track(vis, persist=True, tracker="./vehicle_tracking/botsort.yaml", verbose=False)

            if yolo_results[0].boxes.id is not None:
                boxes = yolo_results[0].boxes.xyxy.cpu()
                track_ids = yolo_results[0].boxes.id.int().cpu().tolist()
                classes = yolo_results[0].boxes.cls.cpu().tolist()

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    if int(cls) not in vehicle_class_indices:
                        continue
                    x1, y1, x2, y2 = [int(coord.item()) for coord in box]
                    area = (x2 - x1) * (y2 - y1)
                    if area < 5000:
                        continue
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # --- Lane Crossing Detection ---
                    box_bottom_y = y2
                    box_mid_start = int(x1 + 0.45 * (x2 - x1))
                    box_mid_end = int(x2 - 0.45 * (x2 - x1))

                    crossed_lane = False

                    for lane_result in fitted_lanes["lanes"]:
                        coeffs = lane_result["coefficients"]
                        y_bounds = lane_result["y_bounds"]
                        ext_bounds = lane_result["extrapolated_y_range"]

                        # Helper function to sample points from line
                        def sample_points(y_start, y_end):
                            y_vals = np.linspace(y_start, y_end, num=100)
                            x_vals = (y_vals - coeffs[1]) / coeffs[0]
                            return np.stack([x_vals, y_vals], axis=1)

                        # Original line
                        orig_pts = sample_points(*y_bounds)
                        # Extrapolated line (if any)
                        ext_pts = sample_points(*ext_bounds) if ext_bounds else []

                        for pts in [orig_pts, ext_pts]:
                            for j in range(len(pts) - 1):
                                x0, y0 = pts[j]
                                x1_, y1_ = pts[j + 1]
                                if (y0 - box_bottom_y) * (y1_ - box_bottom_y) <= 0:  # crossing happens
                                    if y1_ != y0:
                                        x_cross = x0 + (x1_ - x0) * (box_bottom_y - y0) / (y1_ - y0)
                                        if box_mid_start <= x_cross <= box_mid_end:
                                            crossed_lane = True
                                            break
                            if crossed_lane:
                                break

                    if crossed_lane:
                        cv2.putText(vis, f"LANE CHANGE", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        lane_change_events.append({
                            "timestamp": timestamp,
                            "track_id": track_id,
                        })
                    cv2.putText(vis, f"ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    vehicle_trajectories.setdefault(track_id, []).append((timestamp, [x1, y1, x2, y2]))
            
            # Save output
            vout.write(vis)

        vout.release()

        lane_change_events_deduped = deduplicate_lane_changes(lane_change_events, threshold=1.0)
        # Save deduplicated events
        with open(json_path, "w") as f:
            json.dump(lane_change_events_deduped, f, indent=4)
        print(f"✅ Lane change detection results saved to {json_path}")
