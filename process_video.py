import os, re
import subprocess
import sys
from pathlib import Path
import shutil

def process_video(video_path: str):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: {video_path} does not exist.")
        sys.exit(1)

    video_name = video_path.stem
    output_dir = Path(f"./video_splits/images/{video_name}")
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Directory to save resized video
    resized_video_dir = Path("./videos/resized")
    resized_video_dir.mkdir(parents=True, exist_ok=True)
    resized_video_path = resized_video_dir / f"{video_name}_resized.mp4"

    # Step 1: Resize and save the resized video
    print(f"Resizing video and saving to: {resized_video_path}")
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vf", "scale=1920:1080",
        "-c:v", "libx264",  # Use H.264 codec
        "-crf", "0",       # Quality setting: lower is better quality
        "-preset", "fast",  # Encoding speed/quality tradeoff
        str(resized_video_path)
    ], check=True)

    # Step 2: Extract frames from resized video
    print(f"Extracting frames from resized video: {resized_video_path}")
    subprocess.run([
        "ffmpeg", "-i", str(resized_video_path),
        "-vf", "fps=30",
        "-q:v", "1",
        str(tmp_dir / "frame_%05d.jpg")
    ], check=True)

    fps = 30
    print(f"Renaming frames to include timestamps")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(sorted(tmp_dir.glob("frame_*.jpg"))):
        timestamp = i / fps
        # Zero-pad whole seconds to 3 digits, keep 2 decimals after dot
        # e.g. 1.00 => 001_00, 10.50 => 010_50
        timestamp_str = f"{timestamp:06.2f}s".replace('.', '_')
        new_name = output_dir / f"frame_{timestamp_str}.jpg"
        frame.rename(new_name)


    # Remove temp dir
    shutil.rmtree(tmp_dir)

    # Step 4: Create split file (unchanged)
    split_file = Path(f"./video_splits/lists/{video_name}.txt")
    split_file.parent.mkdir(parents=True, exist_ok=True)

    def extract_seconds_from_filename(frame_path):
        match = re.search(r'frame_(\d+)_?(\d*)s\.jpg', frame_path.name)
        if match:
            whole, fractional = match.groups()
            fractional = fractional if fractional else "0"
            return float(f"{whole}.{fractional}")
        return float('inf')

    print(f"Writing frame paths to {split_file}")
    with open(split_file, "w") as f:
        for frame in sorted(output_dir.glob("frame_*.jpg"), key=extract_seconds_from_filename):
            relative_path = frame.relative_to("video_splits")
            f.write(f"{relative_path}\n")

    print("✅ Video processing complete!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_video.py <path_to_video>")
        sys.exit(1)

    process_video(sys.argv[1])
    print(f"Video processing completed successfully. Check the 'video_splits' directory for results.")
