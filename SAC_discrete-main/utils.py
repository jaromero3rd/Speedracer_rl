import torch

def save(args, save_name, model, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state, info = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Combine terminated and truncated
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, info = env.reset()
