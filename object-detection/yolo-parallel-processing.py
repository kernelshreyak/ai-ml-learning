import os
import shutil
import argparse
import cv2  # pip install opencv-python
import numpy as np  # pip install numpy
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ultralytics import YOLO

# --- Configuration ---
YOLO_MODEL_PATH = 'yolov8s.pt'
SOURCE_VIDEO_PATH = 'traffic_cam.mp4'
NUM_CAMERAS = 1900
BASE_DIR = 'camera_processing_data'

# Batch size controls memory usage vs speed. 
# 32-64 is usually optimal for T4 GPU with 1080p images.
BATCH_SIZE = 64 

# --- Phase 1: File Preparation (Parallel I/O) ---
def copy_single_video(i, source_path, base_dir):
    """Copies a single video file. Used by ProcessPoolExecutor."""
    camera_id = f"camera_{i:04d}"
    folder_path = os.path.join(base_dir, camera_id)
    os.makedirs(folder_path, exist_ok=True)
    
    video_copy_path = os.path.join(folder_path, f"{camera_id}_stream.mp4")
    
    if not os.path.exists(video_copy_path):
        shutil.copy2(source_path, video_copy_path)
        if i % 50 == 0: 
            return f"Created {camera_id}"
    return None

def prepare_files(source_path: str, num_cameras: int, base_dir: str, max_workers: int):
    print(f"--- Phase 1: Preparation (Creating {num_cameras} folders) ---")
    start_time = time.time()

    if not os.path.exists(source_path):
        print(f"Error: Source video not found at {source_path}.")
        return

    os.makedirs(base_dir, exist_ok=True)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_single_video, i, source_path, base_dir) 
                   for i in range(1, num_cameras + 1)]
        for f in futures:
            res = f.result()
            if res: print(res)
    
    duration = time.time() - start_time
    print(f"--- Preparation Complete ---")
    print(f"Total time: {duration:.2f} seconds")


# --- Phase 2: Batch Processing (Optimized) ---

def read_frame_worker(task):
    """
    Reads the first frame of a video. 
    Input: (index, video_path)
    Output: (index, frame_array_or_None)
    """
    idx, video_path = task
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return idx, frame
    return idx, None

def save_result_worker(task):
    """
    Saves the bbox data to disk, handling persistence rotation.
    Input: (folder_path, bbox_data)
    """
    folder_path, bbox_data = task
    
    # Define file paths
    current_file = os.path.join(folder_path, "bboxes.npy")
    prev_file = os.path.join(folder_path, "bboxes_prev.npy")
    
    # 1. Rotate persistence: If a current file exists, move it to 'prev'
    # This enables your 'stalled vehicle' comparison (current vs prev)
    if os.path.exists(current_file):
        shutil.move(current_file, prev_file)
        
    # 2. Save new data
    np.save(current_file, bbox_data)
    return None

def process_videos_batched(num_cameras: int, base_dir: str, model_path: str):
    print(f"\n--- Phase 2: High-Performance Batch Processing ---")
    print(f"Target: {num_cameras} cameras | Batch Size: {BATCH_SIZE}")
    
    total_start_time = time.time()

    # 1. Load Model ONCE (This was the previous bottleneck)
    print("Loading YOLO model to GPU...")
    model_load_start = time.time()
    model = YOLO(model_path) 
    print(f"Model loaded in {time.time() - model_load_start:.2f} seconds")

    # 2. Collect all tasks
    all_camera_tasks = []
    for i in range(1, num_cameras + 1):
        camera_id = f"camera_{i:04d}"
        folder_path = os.path.join(base_dir, camera_id)
        video_path = os.path.join(folder_path, f"{camera_id}_stream.mp4")
        if os.path.exists(video_path):
            all_camera_tasks.append({
                'id': camera_id,
                'video': video_path,
                'folder': folder_path
            })

    total_videos = len(all_camera_tasks)
    print(f"Processing {total_videos} videos in batches...")

    # 3. Process in chunks (Batches)
    processed_count = 0
    
    # We use ThreadPool for I/O because reading frames releases the GIL
    # and we want to keep the GPU busy.
    io_pool = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4))

    for i in range(0, total_videos, BATCH_SIZE):
        batch_tasks = all_camera_tasks[i : i + BATCH_SIZE]
        batch_paths = [(idx, t['video']) for idx, t in enumerate(batch_tasks)]
        
        # A. Parallel Frame Extraction (Disk I/O)
        # We map the read function over the batch
        frame_results = list(io_pool.map(read_frame_worker, batch_paths))
        
        valid_frames = []
        valid_indices = []
        
        # Filter out failed reads
        for idx, frame in frame_results:
            if frame is not None:
                valid_frames.append(frame)
                valid_indices.append(idx)
        
        if not valid_frames:
            continue

        # B. Batch Inference (GPU)
        # Passing a list of frames to YOLO runs batch inference automatically
        # verbose=False speeds up processing by reducing print overhead
        results = model.predict(valid_frames, save=False, conf=0.25, device=0, verbose=False)

        # C. Save Results (Disk I/O)
        # Prepare save tasks
        save_tasks = []
        for j, res in enumerate(results):
            # Map back to the original task info using valid_indices
            task_info = batch_tasks[valid_indices[j]]
            
            # Extract boxes to CPU numpy array
            # .cpu().numpy() is very fast for small tensors like bbox lists
            boxes = res.boxes.data.cpu().numpy()
            
            save_tasks.append((task_info['folder'], boxes))
        
        # Run saves in parallel
        list(io_pool.map(save_result_worker, save_tasks))
        
        processed_count += len(save_tasks)
        print(f"Batch {i//BATCH_SIZE + 1} complete. Processed {processed_count}/{total_videos} videos.")

    io_pool.shutdown()

    total_duration = time.time() - total_start_time
    print(f"\n--- Batch Processing Complete ---")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print(f"Average time per camera: {total_duration/total_videos:.4f} seconds")


def main():
    parser = argparse.ArgumentParser(description="High-Performance YOLO Video Processor")
    parser.add_argument('phase', choices=['prepare', 'process'], help="Phase: 'prepare' or 'process'")
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 4, help="Worker threads/processes")
    
    args = parser.parse_args()

    if args.phase == 'prepare':
        # Cap workers at 16 for preparation to avoid file system thrashing
        workers = min(args.workers, 16)
        prepare_files(SOURCE_VIDEO_PATH, NUM_CAMERAS, BASE_DIR, workers)
        
    elif args.phase == 'process':
        # Process phase handles its own threading/batching logic
        process_videos_batched(NUM_CAMERAS, BASE_DIR, YOLO_MODEL_PATH)

if __name__ == '__main__':
    main()