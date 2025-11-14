import numpy as np
import time
import torch


def test_inference_speed(model, input_shape=(1, 3, 224, 224), device='cuda' if torch.cuda.is_available() else 'cpu', num_samples=20):
    """
    Test the inference speed of a model by measuring forward pass time.
    
    Args:
        model: PyTorch model to test
        input_shape: Shape of input tensor (default: (1, 3, 224, 224) for images)
        device: Device to run inference on ('cuda' or 'cpu')
        num_samples: Number of data points to generate and test (default: 20)
    
    Returns:
        dict: Dictionary containing 'mean' and 'std' of inference times in milliseconds
    """
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Generate random data
    print(f"Generating {num_samples} random datapoints with shape {input_shape}...")
    data = torch.randn(num_samples, *input_shape[1:]).to(device)
    
    # Warm-up pass (important for GPU timing accuracy)
    with torch.no_grad():
        _ = model(data[0].unsqueeze(0))
    
    # Synchronize GPU if using CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure inference time for each sample
    inference_times = []
    
    print(f"Running inference on {num_samples} samples...")
    with torch.no_grad():
        for i in range(num_samples):
            # Get single sample
            sample = data[i].unsqueeze(0)
            
            # Start timing
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            # Forward pass
            _ = model(sample)
            
            # End timing
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            # Record time in milliseconds
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
    
    # Calculate statistics
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    
    print(f"\nInference Speed Results:")
    print(f"Mean: {mean_time:.4f} ms")
    print(f"Std:  {std_time:.4f} ms")
    
    return {
        'mean': mean_time,
        'std': std_time
    }
