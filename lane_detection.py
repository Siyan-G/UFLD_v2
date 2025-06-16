import os, re, json, torch, cv2
import numpy as np
import tqdm
from utils.common import merge_config, get_model
from data.dataset import LaneTestDataset
from utils.dist_utils import dist_print
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def fit_lane_polyline(points, degree=1, num_samples=50):
    """
    Fit a polynomial curve through lane points and return smoothed points.
    """
    if len(points) < degree + 1:
        return None

    points_np = np.array(points, dtype=np.float32)
    x = points_np[:, 0]
    y = points_np[:, 1]

    try:
        coeffs = np.polyfit(y, x, deg=degree)
        poly_func = np.poly1d(coeffs)
        y_fit = np.linspace(min(y), max(y), num_samples)
        x_fit = poly_func(y_fit)
        fitted_points = np.array(list(zip(x_fit, y_fit)), dtype=np.int32)
        return fitted_points.reshape((-1, 1, 2))
    except (np.RankWarning, ValueError):
        return None

def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1920, original_image_height=1080):
    """
    Convert model predictions to pixel coordinates.
    """
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    _, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cuda()
    valid_row = pred['exist_row'].argmax(1).cuda()
    max_indices_col = pred['loc_col'].argmax(1).cuda()
    valid_col = pred['exist_col'].argmax(1).cuda()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.arange(
                        max(0, max_indices_row[0,k,i] - local_width),
                        min(num_grid_row - 1, max_indices_row[0,k,i] + local_width) + 1,
                        device=max_indices_row.device
                    )
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.arange(
                        max(0, max_indices_col[0,k,i] - local_width),
                        min(num_grid_col - 1, max_indices_col[0,k,i] + local_width) + 1,
                        device=max_indices_col.device
                    )
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

    if cfg.backbone not in ['18','34','50','101','152','50next','101next','50wide','101wide']:
        raise ValueError("Unsupported backbone specified.")

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        splits = ['test.txt']
        img_w, img_h = 1920, 1080
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, 'lists', split),
                                    img_transform=None, crop_size=cfg.train_height) for split in splits]

    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        splits = ['lane_change.txt']
        img_w, img_h = 1920, 1080
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, split),
                                    img_transform=None, crop_size=cfg.train_height) for split in splits]

    elif cfg.dataset == 'CurveLanes':
        cls_num_per_lane = 72
        splits = ['example.txt']
        img_w, img_h = 1920, 1080
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, 'lists', split),
                                    img_transform=None, crop_size=cfg.train_height) for split in splits]
    else:
        raise NotImplementedError("Unknown dataset: {}".format(cfg.dataset))

    # Load model
    device = torch.device("cuda:0")
    net = get_model(cfg)
    state_dict = torch.load(cfg.test_model)['model']
    net.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    net.to(device)
    net.eval()
    print("✅ Model loaded on CUDA device.")

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for dataset in datasets:
        dataset.img_transform = img_transforms

    for split, dataset in zip(splits, datasets):
        lane_results = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        video_name = os.path.splitext(os.path.basename(split))[0]
        output_dir = os.path.join("output", video_name)
        os.makedirs(output_dir, exist_ok=True)

        avi_path = os.path.join(output_dir, f"{video_name}.avi")
        json_path = os.path.join(output_dir, f"{video_name}.json")
        vout = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (img_w, img_h))
        if not vout.isOpened():
            raise RuntimeError("❌ VideoWriter failed to open. Check codec and frame size.")

        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.to(device)
            with torch.no_grad():
                pred = net(imgs)

            vis = cv2.imread(os.path.join(cfg.data_root, names[0]))
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

            lane_results.append({
                "timestamp": timestamp,
                "filename": names[0],
                "lanes": coords
            })

            # Plot lanes using raw points
            # for lane in coords:
            #     for coord in lane:
            #         cv2.circle(vis, coord, 5, (0, 255, 0), -1)

            # Plot polylines for each lane
            # for lane in coords:
            #     if len(lane) < 10:
            #         continue
            #     lane_pts = [(int(x), int(y)) for x, y in lane]
            #     if len(lane_pts) >= 2:
            #         cv2.polylines(vis, [np.array(lane_pts, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=3)

            # Define distinct BGR colors for up to 10 lanes
            lane_colors = [
                (255, 0, 0),    # Blue
                (0, 255, 0),    # Green
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 128),  # Purple
                (0, 128, 255),  # Orange
                (0, 0, 128),    # Navy
                (128, 128, 0),  # Olive
            ]

            # Draw fitted lanes with different colors
            for idx, lane in enumerate(coords):
                if len(lane) < 10:
                    continue
                fit_pts = fit_lane_polyline(lane)
                if fit_pts is not None:
                    color = lane_colors[idx % len(lane_colors)]
                    cv2.polylines(vis, [fit_pts], isClosed=False, color=color, thickness=3)

            vout.write(vis)

        vout.release()
        with open(json_path, "w") as f:
            json.dump(lane_results, f, indent=2)
        print(f"✅ Lane detection results saved to {json_path}")
