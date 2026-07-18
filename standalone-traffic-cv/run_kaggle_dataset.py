import kagglehub
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle traffic dataset and run inference")
    parser.add_argument("--video-index", type=int, default=0, help="Index of video to process (default: first video)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without GUI")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--output", type=str, default="kaggle_metrics.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    # Step 1: Download Kaggle dataset
    print("Downloading Kaggle traffic dataset...")
    dataset_path = kagglehub.dataset_download("arshadrahmanziban/traffic-video-dataset")
    print(f"✅ Dataset downloaded to: {dataset_path}")

    # Step 2: Find all video files
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
    
    video_files = sorted(video_files)
    print(f"\nFound {len(video_files)} video files:")
    for i, video in enumerate(video_files[:15], 1):
        print(f"  {i:2d}. {os.path.basename(video)}")
    if len(video_files) > 15:
        print(f"  ... and {len(video_files) - 15} more")

    # Step 3: Run pipeline on selected video
    if len(video_files) > args.video_index:
        selected_video = video_files[args.video_index]
        print(f"\n🚀 Running pipeline on video {args.video_index + 1}: {os.path.basename(selected_video)}")
        
        # Get the directory of this script to find yolo_inference.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_script = os.path.join(script_dir, "yolo_inference.py")
        
        # Change working directory to script_dir for correct relative paths
        original_cwd = os.getcwd()
        os.chdir(script_dir)
        
        # Build the command
        cmd = f'python "{yolo_script}" --source "{selected_video}" --output "{args.output}" --imgsz {args.imgsz} --skip-frames {args.skip_frames}'
        if args.headless:
            cmd += " --headless"
        
        print(f"\nExecuting: {cmd}")
        os.system(cmd)
        
        # Restore original working directory
        os.chdir(original_cwd)
    else:
        print(f"\n❌ Video index {args.video_index} out of range (only {len(video_files)} videos found)")

if __name__ == "__main__":
    main()
