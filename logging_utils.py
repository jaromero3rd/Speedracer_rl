import os
import glob
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def load_video_to_tensor(video_path, max_frames=1000):
    """Load video file as tensor for tensorboard. Returns (1, T, C, H, W) or None."""
    try:
        import cv2
    except ImportError:
        print("cv2 not available")
        return None

    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
        frames.append(frame_tensor)

    cap.release()

    if len(frames) == 0:
        return None

    video = torch.stack(frames).unsqueeze(0)  # (1, T, C, H, W)
    return video


def create_writer(run_name, log_dir="./logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}_{timestamp}")
    os.makedirs(log_path, exist_ok=True)
    return SummaryWriter(log_dir=log_path), log_path


def log_hyperparameters(writer, config):
    hparams = {
        "env": config.env,
        "episodes": config.episodes,
        "buffer_size": config.buffer_size,
        "batch_size": config.batch_size,
        "seed": config.seed,
    }

    if hasattr(config, 'learning_rate'):
        hparams["learning_rate"] = config.learning_rate
    if hasattr(config, 'entropy_bonus'):
        hparams["entropy_bonus"] = str(config.entropy_bonus)
    if hasattr(config, 'epsilon'):
        hparams["epsilon"] = config.epsilon
    if hasattr(config, 'obs_buffer_max_len'):
        hparams["obs_buffer_max_len"] = config.obs_buffer_max_len

    try:
        writer.add_hparams(hparams, {"hparam/final_reward": 0.0})
    except:
        pass

    hparams_str = "\n".join([f"  {k}: {v}" for k, v in sorted(hparams.items())])
    writer.add_text("Hyperparameters", hparams_str, 0)


def log_episode_metrics(writer, episode, reward, avg_reward_10, total_steps,
                        policy_loss, alpha_loss, bellmann_error1, bellmann_error2,
                        current_alpha, steps, buffer_size):
    writer.add_scalar("Reward/Episode", reward, episode)
    writer.add_scalar("Reward/Avg10", avg_reward_10, episode)
    writer.add_scalar("Loss/Policy", policy_loss, episode)
    writer.add_scalar("Loss/Alpha", alpha_loss, episode)
    writer.add_scalar("Loss/Bellmann1", bellmann_error1, episode)
    writer.add_scalar("Loss/Bellmann2", bellmann_error2, episode)
    writer.add_scalar("Training/Alpha", current_alpha, episode)
    writer.add_scalar("Training/Steps", steps, episode)


def log_video(writer, episode, video_dir="./video", fps=4):
    """Try to log most recent video to tensorboard."""
    mp4list = glob.glob(os.path.join(video_dir, '**', '*.mp4'), recursive=True)
    if not mp4list:
        return False

    # get most recent
    mp4 = max(mp4list, key=os.path.getmtime)
    video_tensor = load_video_to_tensor(mp4)

    if video_tensor is None:
        return False

    try:
        video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
        writer.add_video(f"Episode_{episode}", video_tensor, global_step=episode, fps=fps)
        writer.flush()
        return True
    except Exception as e:
        print(f"Failed to log video: {e}")
        return False
