from DQN import DQNAgent
import numpy as np
import random
from tmrl import get_environment
from time import sleep
import os
from datetime import datetime
import data.save_utils as save_utl
import tests.test_env_connection as tec


def random_start_action():
    # Creates completely random action
    steer = np.random.uniform(-1, 1)
    return np.array([1.0, 0.0, steer])


def convert_result_to_input(input):
    # F = forward, B = break, L = left, R = right
    # Possibilities = {1:F 2:FB, 3:FL, 4:FR, 5:FBL, 6:FBR}
    input_map = [[1, 0, 0], [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, 1, -1], [1, 1, 1]]
    return np.array(input_map[input])


if __name__ == "__main__":
    # Create run directory
    run_dir = save_utl.create_run_directory(os.getcwd() +  "/data")

    # We will have to create a try method here to ensure the envi
    print("pre_env_set_up")
    env = get_environment()
    sleep(5.0)
    tec.check_env()
    obs, info = env.reset() 
    print(obs)
    print("past sleep")

    # Initialize agent with custom image size
    num_actions = 6
    image_size = 64
    agent = DQNAgent(num_actions=num_actions, image_size=image_size)
    
    print(f"Initialized DQN Agent with image size: {image_size}x{image_size}")
    print(f"CNN output features: {agent.q_network.image_feature_size}")
    
    # Save run metadata
    metadata = {
        "start_time": datetime.now().isoformat(),
        "num_actions": num_actions,
        "image_size": image_size,
        "cnn_features": agent.q_network.image_feature_size,
        "learning_rate": 1e-4,
        "gamma": agent.gamma,
        "epsilon_start": 1.0,
        "epsilon_end": agent.epsilon_end,
        "epsilon_decay": agent.epsilon_decay,
        "buffer_capacity": len(agent.replay_buffer.buffer),
        "batch_size": agent.batch_size,
        "target_update_freq": agent.target_update_freq,
    }
    save_utl.save_run_metadata(run_dir, metadata)
    
    num_episodes = 4
    max_steps = 2
    
    for episode in range(num_episodes):
        # Initialize episode tracking
        episode_data = {
            "episode": episode,
            "steps": [],
            "episode_reward": 0,
            "episode_loss": [],
            "start_time": datetime.now().isoformat(),
        }
        
        # Initialize state (example random state)
        act = random_start_action() 
        obs, rew, terminated, truncated, info = env.step(act) 
        int_features = np.array([0, 0, 0])  # Three integer features
        image = np.random.randn(1, image_size, image_size)  # Random image with specified size
        state = (int_features, image)
        
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action [will hand back an integer]
            input_idx = agent.select_action(state)
            action = convert_result_to_input(input_idx)
            
            # Take action in environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Create next state (you'll need to extract proper features from next_obs)
            next_int_features = np.array([step % 10, (step + 1) % 10, 0])  # Update based on your obs
            next_image = np.random.randn(1, image_size, image_size)  # Update based on your obs
            next_state = (next_int_features, next_image)
            
            # Store step data
            step_data = {
                "step": step,
                "state_int_features": int_features.tolist(),
                "state_image_shape": image.shape,
                "action_index": int(input_idx),
                "action_array": action.tolist(),
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "next_state_int_features": next_int_features.tolist(),
            }
            episode_data["steps"].append(step_data)
            
            # Optionally save individual step data (can generate many files)
            # save_step_data(run_dir, episode, step, step_data)
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, input_idx, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update episode data
        episode_data["episode_reward"] = float(episode_reward)
        episode_data["episode_loss"] = [float(l) for l in episode_loss]
        episode_data["avg_loss"] = float(np.mean(episode_loss)) if episode_loss else 0.0
        episode_data["num_steps"] = step + 1
        episode_data["epsilon"] = float(agent.epsilon)
        episode_data["end_time"] = datetime.now().isoformat()
        
        # Save episode data to disk
        save_utl.save_episode_data(run_dir, episode, episode_data)
        
        # Logging
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Epsilon: {agent.epsilon:.3f}")
        
        # Save training log entry
        log_entry = {
            "episode": episode,
            "reward": float(episode_reward),
            "avg_loss": float(avg_loss),
            "epsilon": float(agent.epsilon),
            "num_steps": step + 1,
            "timestamp": datetime.now().isoformat(),
        }
        save_utl.save_training_log(run_dir, log_entry)
        
        # Save checkpoint periodically
        if (episode + 1) % 10 == 0:
            checkpoint_path = os.path.join(run_dir, "checkpoints", f"dqn_checkpoint_ep{episode + 1}.pt")
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to: {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint = os.path.join(run_dir, "checkpoints", "dqn_final.pt")
    agent.save(final_checkpoint)
    
    # Update metadata with completion info
    metadata["end_time"] = datetime.now().isoformat()
    metadata["total_episodes"] = num_episodes
    metadata["status"] = "completed"
    save_utl.save_run_metadata(run_dir, metadata)
    
    print(f"\nTraining complete!")
    print(f"All data saved to: {run_dir}")