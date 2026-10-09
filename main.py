import argparse
import json
from pathlib import Path

import cv2
import numpy as np

VIDEO_PATH = "record.mp4"
JSONL_PATH = "tracking_data.jsonl"
OUTPUT_DIR = Path("output_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_SIZE = 600
SMOOTH_WINDOW = 5

XR_BODY_BONES = [
    (13, 12),
    (12, 14),
    (12, 15),
    (16, 13),
    (17, 14),
    (17, 9),
    (9, 3),
    (3, 0),
    (9, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    (19, 17),
    (21, 19),
    (0, 1),
    (1, 4),
    (4, 7),
    (7, 10),
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),
]


def parse_pose_string(s):
    parts = str(s).split(",")
    numbers = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            try:
                numbers.append(float(parts[i] + "." + parts[i + 1]))
            except ValueError:
                numbers.append(0.0)
            i += 2
        else:
            try:
                numbers.append(float(parts[i]))
            except ValueError:
                numbers.append(0.0)
            i += 1
    return numbers


def load_tracking_data(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "Body" not in data or "timeStampNs" not in data:
                    continue
                records.append(data)
            except json.JSONDecodeError:
                continue
    return records


def extract_positions(json_data):
    result = {"body": [], "left_hand": [], "right_hand": []}

    for idx, joint in enumerate(json_data.get("Body", {}).get("joints", [])):
        p = parse_pose_string(joint["p"])
        if len(p) >= 3:
            result["body"].append({"pos": [p[0], p[1], p[2]], "type": joint.get("t", -1), "index": idx})

    lh = json_data.get("Hand", {}).get("leftHand", {})
    if lh.get("isActive"):
        for joint in lh.get("HandJointLocations", []):
            p = parse_pose_string(joint["p"])
            if len(p) >= 3:
                result["left_hand"].append([p[0], p[1], p[2]])

    rh = json_data.get("Hand", {}).get("rightHand", {})
    if rh.get("isActive"):
        for joint in rh.get("HandJointLocations", []):
            p = parse_pose_string(joint["p"])
            if len(p) >= 3:
                result["right_hand"].append([p[0], p[1], p[2]])

    return result


class PositionSmoother:
    def __init__(self, window=5):
        self.window = window
        self.history = []

    def smooth(self, positions):
        self.history.append(positions)
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) < 2:
            return positions

        if isinstance(positions, list) and len(positions) > 0 and isinstance(positions[0], dict):
            smoothed = []
            for i in range(len(positions)):
                xs, ys, zs = [], [], []
                for h in self.history:
                    if i < len(h):
                        pos = h[i]["pos"]
                        xs.append(pos[0])
                        ys.append(pos[1])
                        zs.append(pos[2])
                if xs:
                    smoothed.append(
                        {
                            "pos": [np.mean(xs), np.mean(ys), np.mean(zs)],
                            "type": positions[i].get("type", -1),
                            "index": positions[i].get("index", i),
                        }
                    )
                else:
                    smoothed.append(positions[i])
            return smoothed
        else:
            smoothed = []
            for i in range(len(positions)):
                xs, ys, zs = [], [], []
                for h in self.history:
                    if i < len(h):
                        xs.append(h[i][0])
                        ys.append(h[i][1])
                        zs.append(h[i][2])
                if xs:
                    smoothed.append([np.mean(xs), np.mean(ys), np.mean(zs)])
                else:
                    smoothed.append(positions[i])
            return smoothed


def calculate_scale_and_offset(all_positions, panel_size):
    if not all_positions:
        return 1.0, 0.0, 0.0, 0.0
    positions = np.array(all_positions)
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 0.1)
    scale = (panel_size * 0.8) / max_range

    return scale, (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2


def draw_skeleton_view(canvas, body_data, left_hand, right_hand, view_type):
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2

    if view_type == "front":
        h_axis, v_axis, h_inv, v_inv = 0, 1, False, True
        title, h_label, v_label = "Front View (X, Y)", "X", "Y"
    elif view_type == "side":
        h_axis, v_axis, h_inv, v_inv = 2, 1, True, True
        title, h_label, v_label = "Side View (Z, Y)", "Z", "Y"
    elif view_type == "top":
        h_axis, v_axis, h_inv, v_inv = 0, 2, False, False
        title, h_label, v_label = "Top View (X, Z)", "X", "Z"
    else:
        h_axis, v_axis, h_inv, v_inv = 0, 1, False, True
        title, h_label, v_label = "View", "X", "Y"

    all_positions = [b["pos"] for b in body_data] + left_hand + right_hand
    scale, x_center, y_center, z_center = calculate_scale_and_offset(all_positions, h)

    def project(x, y, z):
        x -= x_center
        y -= y_center
        z -= z_center
        h_val = [x, y, z][h_axis] * (-1 if h_inv else 1)
        v_val = [x, y, z][v_axis] * (-1 if v_inv else 1)
        return int(cx + h_val * scale), int(cy + v_val * scale)

    body_by_index = {b["index"]: b for b in body_data}

    for bone_start, bone_end in XR_BODY_BONES:
        if bone_start in body_by_index and bone_end in body_by_index:
            pos_start = body_by_index[bone_start]["pos"]
            pos_end = body_by_index[bone_end]["pos"]
            px1, py1 = project(pos_start[0], pos_start[1], pos_start[2])
            px2, py2 = project(pos_end[0], pos_end[1], pos_end[2])
            cv2.line(canvas, (px1, py1), (px2, py2), (0, 150, 0), 3)

    for b in body_data:
        px, py = project(b["pos"][0], b["pos"][1], b["pos"][2])
        if b["type"] == 416218016093:
            cv2.circle(canvas, (px, py), 9, (0, 255, 255), -1)
        else:
            cv2.circle(canvas, (px, py), 7, (0, 200, 0), -1)
        cv2.putText(canvas, str(b["index"]), (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if left_hand:
        lh_pts = [project(pos[0], pos[1], pos[2]) for pos in left_hand]
        for i in range(len(lh_pts) - 1):
            cv2.line(canvas, lh_pts[i], lh_pts[i + 1], (255, 100, 0), 2)
        for px, py in lh_pts:
            cv2.circle(canvas, (px, py), 3, (255, 140, 0), -1)

    if right_hand:
        rh_pts = [project(pos[0], pos[1], pos[2]) for pos in right_hand]
        for i in range(len(rh_pts) - 1):
            cv2.line(canvas, rh_pts[i], rh_pts[i + 1], (0, 0, 180), 2)
        for px, py in rh_pts:
            cv2.circle(canvas, (px, py), 3, (0, 0, 220), -1)

    cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    cv2.putText(
        canvas,
        f"Body: {len(body_data)} | LH: {len(left_hand)} | RH: {len(right_hand)}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )

    ox, oy = 50, h - 50
    cv2.arrowedLine(canvas, (ox, oy), (ox + 50, oy), (0, 0, 255), 2)
    cv2.arrowedLine(canvas, (ox, oy), (ox, oy - 50), (0, 255, 0), 2)
    cv2.putText(canvas, h_label, (ox + 55, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(canvas, v_label, (ox - 10, oy - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return canvas


def sync_video_and_tracking(video_path, tracking_records):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not tracking_records:
        return [], total_frames, fps

    t0_track = tracking_records[0]["timeStampNs"] / 1e9
    track_times = [(r["timeStampNs"] / 1e9) - t0_track for r in tracking_records]

    sync_indices = []
    for frame_idx in range(total_frames):
        video_time = frame_idx / fps
        best_idx, best_diff = 0, float("inf")
        for ti, tt in enumerate(track_times):
            diff = abs(tt - video_time)
            if diff < best_diff:
                best_diff, best_idx = diff, ti
            elif tt > video_time + 1.0:
                break
        sync_indices.append(best_idx)
    return sync_indices, total_frames, fps


def export_comparison_frames(video_path, jsonl_path, output_dir, num_frames=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Загрузка данных...")
    records = load_tracking_data(jsonl_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_sbs = width > (height * 1.5)
    cap.release()

    print(f"   Видео: {width}x{height}, {fps} FPS, SBS: {is_sbs} | Трекинг: {len(records)} записей")
    sync_indices, _, _ = sync_video_and_tracking(video_path, records)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    smoothers = {
        v: {
            "body": PositionSmoother(SMOOTH_WINDOW),
            "lh": PositionSmoother(SMOOTH_WINDOW),
            "rh": PositionSmoother(SMOOTH_WINDOW),
        }
        for v in ["front", "side", "top"]
    }

    print(f"📸 Экспорт {num_frames} кадров...")
    cap = cv2.VideoCapture(video_path)
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        video_frame = frame[:, : width // 2].copy() if is_sbs else frame.copy()
        track_idx = sync_indices[frame_idx] if frame_idx < len(sync_indices) else 0
        positions = extract_positions(records[track_idx])

        panels = []
        for view_type in ["front", "side", "top"]:
            s_body = smoothers[view_type]["body"].smooth(positions["body"])
            s_lh = smoothers[view_type]["lh"].smooth(positions["left_hand"])
            s_rh = smoothers[view_type]["rh"].smooth(positions["right_hand"])

            panel = np.full((PANEL_SIZE, PANEL_SIZE, 3), 30, dtype=np.uint8)
            draw_skeleton_view(panel, s_body, s_lh, s_rh, view_type)
            panels.append(panel)

        video_frame = cv2.resize(video_frame, (PANEL_SIZE, PANEL_SIZE))
        combined = np.vstack([np.hstack([video_frame, panels[0]]), np.hstack([panels[1], panels[2]])])

        cv2.line(combined, (PANEL_SIZE, 0), (PANEL_SIZE, PANEL_SIZE * 2), (100, 100, 100), 2)
        cv2.line(combined, (0, PANEL_SIZE), (PANEL_SIZE * 2, PANEL_SIZE), (100, 100, 100), 2)

        cv2.imwrite(str(output_dir / f"comparison_{idx:02d}_frame{frame_idx:06d}.png"), combined)
        print(f"   ✅ {output_dir.name}/comparison_{idx:02d}_frame{frame_idx:06d}.png")
    cap.release()


def create_combined_video(video_path, jsonl_path, output_path, max_frames=None):
    print("📊 Загрузка данных...")
    records = load_tracking_data(jsonl_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_sbs = width > (height * 1.5)
    if max_frames:
        total_frames = min(total_frames, max_frames)
    cap.release()

    print(f"   Видео: {width}x{height}, {fps} FPS | Трекинг: {len(records)} записей")
    sync_indices, _, _ = sync_video_and_tracking(video_path, records)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (PANEL_SIZE * 2, PANEL_SIZE * 2))

    smoothers = {
        v: {
            "body": PositionSmoother(SMOOTH_WINDOW),
            "lh": PositionSmoother(SMOOTH_WINDOW),
            "rh": PositionSmoother(SMOOTH_WINDOW),
        }
        for v in ["front", "side", "top"]
    }

    print(f"🎬 Рендеринг {total_frames} кадров...")
    cap = cv2.VideoCapture(video_path)
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        video_frame = frame[:, : width // 2].copy() if is_sbs else frame.copy()
        video_frame = cv2.resize(video_frame, (PANEL_SIZE, PANEL_SIZE))

        track_idx = sync_indices[frame_idx] if frame_idx < len(sync_indices) else 0
        positions = extract_positions(records[track_idx])

        panels = []
        for view_type in ["front", "side", "top"]:
            s_body = smoothers[view_type]["body"].smooth(positions["body"])
            s_lh = smoothers[view_type]["lh"].smooth(positions["left_hand"])
            s_rh = smoothers[view_type]["rh"].smooth(positions["right_hand"])

            panel = np.full((PANEL_SIZE, PANEL_SIZE, 3), 30, dtype=np.uint8)
            draw_skeleton_view(panel, s_body, s_lh, s_rh, view_type)
            panels.append(panel)

        combined = np.vstack([np.hstack([video_frame, panels[0]]), np.hstack([panels[1], panels[2]])])
        cv2.line(combined, (PANEL_SIZE, 0), (PANEL_SIZE, PANEL_SIZE * 2), (100, 100, 100), 2)
        cv2.line(combined, (0, PANEL_SIZE), (PANEL_SIZE * 2, PANEL_SIZE), (100, 100, 100), 2)
        cv2.putText(
            combined,
            f"Frame {frame_idx}/{total_frames}",
            (10, PANEL_SIZE * 2 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        out.write(combined)
        if frame_idx % 50 == 0:
            print(f"   Обработано: {frame_idx} / {total_frames}")

    cap.release()
    out.release()
    print(f"✅ Видео сохранено: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XR body tracking visualizer (video + skeleton front/side/top)")
    parser.add_argument("--mode", choices=["video", "frames", "both"], default="both")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--video", default=VIDEO_PATH, help="Путь к видео (по умолчанию: record.mp4)")
    parser.add_argument("--data", default=JSONL_PATH, help="Путь к данным трекинга JSONL (по умолчанию: tracking_data.jsonl)")
    args = parser.parse_args()

    SMOOTH_WINDOW = args.smooth
    print("🎬 XenseVR Visualizer (Front / Side / Top)")
    print(f"   Сглаживание: {SMOOTH_WINDOW} | Костей в скелете: {len(XR_BODY_BONES)}")
    print(f"   Видео: {args.video}")
    print(f"   Трекинг: {args.data}")

    if args.mode in ["frames", "both"]:
        export_comparison_frames(args.video, args.data, OUTPUT_DIR / "comparison_frames", args.num_frames)
    if args.mode in ["video", "both"]:
        create_combined_video(args.video, args.data, OUTPUT_DIR / "skeleton_multiview.mp4", args.max_frames)

    print(f"\n🎉 Готово! Результаты в: {OUTPUT_DIR}")
