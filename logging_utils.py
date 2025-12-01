import os
import glob
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Optional, Dict, Any


def load_video_to_tensor(video_path: str, max_frames: int = 1000) -> Optional[torch.Tensor]:
    """
    Load a video file and convert it to a tensor format for TensorBoard.
    Returns a tensor of shape (1, T, C, H, W) where:
    - 1 is batch size
    - T is number of frames
    - C is channels (3 for RGB)
    - H, W are height and width
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to load (default: 1000, increased to capture full episodes)
    
    Returns:
        Video tensor of shape (1, T, C, H, W) or None if loading fails
    """
    try:
        import cv2
    except ImportError:
        print("Warning: OpenCV (cv2) not available. Install with: pip install opencv-python")
        return None
    
    if not os.path.exists(video_path):
        print(f"Warning: Video file does not exist: {video_path}")
        return None
    
    # Wait a bit to ensure file is fully written (check file size stability)
    import time
    prev_size = -1
    for _ in range(10):  # Check up to 10 times
        try:
            current_size = os.path.getsize(video_path)
            if current_size == prev_size and current_size > 0:
                break
            prev_size = current_size
            time.sleep(0.1)
        except OSError:
            time.sleep(0.1)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Warning: Failed to open video file: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor and normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        # Change from (H, W, C) to (C, H, W)
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frames.append(frame_tensor)
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Stack frames: (T, C, H, W)
    video_tensor = torch.stack(frames)
    # TensorBoard add_video expects: (N, T, C, H, W) where:
    # N = batch size, T = time (frames), C = channels, H = height, W = width
    # Add batch dimension: (1, T, C, H, W)
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor


def create_writer(run_name: str, log_dir: str = "./logs"):
    """
    Create a TensorBoard SummaryWriter with a timestamped log directory.
    
    Args:
        run_name: Name of the run (used in log directory name)
        log_dir: Base directory for logs (default: "./logs")
    
    Returns:
        Tuple of (SummaryWriter instance, log_path string)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}_{timestamp}")
    os.makedirs(log_path, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_path)
    return writer, log_path


def log_hyperparameters(writer: SummaryWriter, config: Any) -> None:
    """
    Log hyperparameters to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        config: Configuration object with hyperparameters
    """
    hparams = {
        "env": config.env,
        "episodes": config.episodes,
        "buffer_size": config.buffer_size,
        "batch_size": config.batch_size,
        "seed": config.seed,
    }
    
    # Add hyperparameters if they exist in config
    if hasattr(config, 'learning_rate'):
        hparams["learning_rate"] = config.learning_rate
    if hasattr(config, 'entropy_bonus'):
        ent_bonus = config.entropy_bonus
        if ent_bonus == "None" or ent_bonus is None:
            hparams["entropy_bonus"] = "learnable"
        else:
            hparams["entropy_bonus"] = float(ent_bonus)
    if hasattr(config, 'epsilon'):
        hparams["epsilon"] = config.epsilon
    if hasattr(config, 'obs_buffer_max_len'):
        hparams["obs_buffer_max_len"] = config.obs_buffer_max_len
    
    # Log hyperparameters to TensorBoard
    # TensorBoard's add_hparams requires a metrics dict (even if empty)
    # The HPARAMS tab will appear when you have multiple runs to compare
    # For now, we'll use a placeholder metric that will be updated during training
    metrics = {
        "hparam/final_avg_reward": 0.0,  # Placeholder - will be updated at end of training
    }
    
    try:
        writer.add_hparams(hparams, metrics)
    except Exception as e:
        print(f"Warning: Failed to log hyperparameters to TensorBoard: {e}")
    
    # Also log hyperparameters as text for easy viewing (always visible)
    hparams_str = "\n".join([f"  {k}: {v}" for k, v in sorted(hparams.items())])
    writer.add_text("Hyperparameters/Config", hparams_str, 0)
    
    # Log each hyperparameter individually as scalars for easy filtering
    for key, value in hparams.items():
        # Convert non-numeric values to strings for text logging
        if isinstance(value, (int, float)):
            writer.add_scalar(f"Hyperparameters/{key}", value, 0)
        else:
            writer.add_text(f"Hyperparameters/{key}", str(value), 0)


def log_episode_metrics(
    writer: SummaryWriter,
    episode: int,
    reward: float,
    avg_reward_10: float,
    total_steps: int,
    policy_loss: float,
    alpha_loss: float,
    bellmann_error1: float,
    bellmann_error2: float,
    current_alpha: float,
    steps: int,
    buffer_size: int
) -> None:
    """
    Log episode metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        episode: Current episode number
        reward: Episode reward
        avg_reward_10: Average reward over last 10 episodes
        total_steps: Total training steps
        policy_loss: Policy loss value
        alpha_loss: Alpha loss value
        bellmann_error1: Bellmann error from Q1 network
        bellmann_error2: Bellmann error from Q2 network
        current_alpha: Current alpha (temperature) value
        steps: Current step count
        buffer_size: Current replay buffer size
    """
    writer.add_scalar("Reward/Episode", reward, episode)
    writer.add_scalar("Reward/Average10", avg_reward_10, episode)
    writer.add_scalar("Training/TotalSteps", total_steps, episode)
    writer.add_scalar("Loss/Policy", policy_loss, episode)
    writer.add_scalar("Loss/Alpha", alpha_loss, episode)
    writer.add_scalar("Loss/BellmannError1", bellmann_error1, episode)
    writer.add_scalar("Loss/BellmannError2", bellmann_error2, episode)
    writer.add_scalar("Training/Alpha", current_alpha, episode)
    writer.add_scalar("Training/Steps", steps, episode)
    writer.add_scalar("Training/BufferSize", buffer_size, episode)


def log_video(
    writer: SummaryWriter,
    episode: int,
    video_dir: str = "./video",
    fps: int = 4
) -> bool:
    """
    Log video to TensorBoard if available.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        episode: Current episode number
        video_dir: Directory containing video files (default: "./video")
        fps: Frames per second for video playback (default: 4)
    
    Returns:
        True if video was successfully logged, False otherwise
    """
    import time
    
    # RecordVideo creates subdirectories like video/rl-video-episode-0/, video/rl-video-episode-10/, etc.
    # Or files like episode-episode-1.mp4, episode-episode-11.mp4, etc.
    # Search recursively for mp4 files
    mp4list = []
    
    # First, check the base directory
    if os.path.exists(video_dir):
        # Search in subdirectories (RecordVideo creates subdirectories)
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith('.mp4'):
                    mp4list.append(os.path.join(root, file))
    
    if len(mp4list) == 0:
        # Try direct glob as fallback
        mp4list = glob.glob(os.path.join(video_dir, '*.mp4'))
        # Also try subdirectories
        mp4list.extend(glob.glob(os.path.join(video_dir, '**', '*.mp4'), recursive=True))
    
    if len(mp4list) == 0:
        print(f"Warning: No video files found in {video_dir} (episode {episode})")
        print(f"  Checked directory: {os.path.abspath(video_dir)}")
        if os.path.exists(video_dir):
            print(f"  Directory exists. Contents: {os.listdir(video_dir)}")
        return False
    
    # Try to find video matching the episode number first
    # RecordVideo creates files like: episode-episode-{episode_num}.mp4
    episode_video = None
    for mp4 in mp4list:
        # Check if filename contains the episode number
        if f"episode-{episode}" in os.path.basename(mp4) or f"-{episode}." in os.path.basename(mp4):
            episode_video = mp4
            break
    
    # If no exact match, get the most recent video (by modification time)
    if episode_video is None:
        try:
            episode_video = max(mp4list, key=os.path.getmtime)
        except (OSError, ValueError):
            # Fallback to creation time if modification time fails
            try:
                episode_video = max(mp4list, key=os.path.getctime)
            except (OSError, ValueError):
                # If both fail, just use the first one
                episode_video = mp4list[0]
    
    mp4 = episode_video
    
    # Load video and convert to tensor format
    video_tensor = load_video_to_tensor(mp4)
    if video_tensor is not None:
        try:
            # Ensure video tensor is in correct format and range
            # TensorBoard expects: (N, T, C, H, W) with values in [0, 1]
            # Clamp values to [0, 1] range to ensure they're valid
            video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
            
            # Verify tensor shape
            if len(video_tensor.shape) != 5:
                raise ValueError(f"Expected 5D tensor (N, T, C, H, W), got shape: {video_tensor.shape}")
            
            N, T, C, H, W = video_tensor.shape
            if N != 1:
                raise ValueError(f"Expected batch size 1, got: {N}")
            if C != 3:
                raise ValueError(f"Expected 3 channels (RGB), got: {C}")
            
            # Ensure we have at least 1 frame
            if T == 0:
                raise ValueError(f"Video has no frames")
            
            # Calculate actual FPS from video if available
            try:
                import cv2
                cap = cv2.VideoCapture(mp4)
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if video_fps > 0:
                    fps = int(video_fps)
            except:
                pass  # Use default fps
            
            # Log video to TensorBoard
            # PyTorch's add_video expects: (N, T, C, H, W) format
            # Videos appear in the "IMAGES" tab in TensorBoard
            # IMPORTANT: Use a simple tag name without slashes for better compatibility
            tag = f"Episode_{episode}"
            
            # Ensure FPS is reasonable (TensorBoard can have issues with very low or very high FPS)
            if fps <= 0:
                fps = 4  # Default FPS
            if fps > 30:
                fps = 30  # Cap at 30 FPS for TensorBoard compatibility
            
            # Log video - use the 'Videos' namespace in the tag
            writer.add_video(f"Videos/{tag}", video_tensor, global_step=episode, fps=fps)
            
            # Flush to ensure data is written
            writer.flush()
            
            return True
        except Exception as e:
            print(f"Warning: Failed to log video to TensorBoard: {e}")
            import traceback
            traceback.print_exc()
            writer.add_text("Videos/Episode", f"Episode {episode}: {mp4} (TensorBoard logging failed: {e})", episode)
            return False
    else:
        # Fallback: log video path as text if loading fails
        writer.add_text("Videos/Episode", f"Episode {episode}: {mp4} (failed to load video)", episode)
        print(f"Warning: Failed to load video: {mp4}")
        return False

