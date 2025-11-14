import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from DQN import DQNAgent
import os
import sys


def count_parameters(model):
    """Count the total number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    
    print(f"Mean: {mean_time:.4f} ms")
    print(f"Std:  {std_time:.4f} ms")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'all_times': inference_times
    }


def test_dqn_inference_speed(image_size, num_actions=4, device='cuda' if torch.cuda.is_available() else 'cpu', num_samples=20):
    """
    Test DQN inference speed for a given image size.
    
    Args:
        image_size: Size of the square image (e.g., 64 for 64x64)
        num_actions: Number of actions for the DQN
        device: Device to run on
        num_samples: Number of samples to test
    
    Returns:
        dict: Dictionary with inference stats and model info
    """
    print(f"\n{'='*60}")
    print(f"Testing DQN with image size: {image_size}x{image_size}")
    print(f"{'='*60}")
    
    # Create agent
    agent = DQNAgent(num_actions=num_actions, image_size=image_size, device=device)
    model = agent.q_network
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:,}")
    
    # Generate test data
    int_features_batch = torch.randn(num_samples, 3).to(device)
    image_batch = torch.randn(num_samples, 1, image_size, image_size).to(device)
    
    # Warm-up
    model.eval()
    with torch.no_grad():
        _ = model(int_features_batch[0].unsqueeze(0), image_batch[0].unsqueeze(0))
    
    # Synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure inference times
    inference_times = []
    
    with torch.no_grad():
        for i in range(num_samples):
            int_feat = int_features_batch[i].unsqueeze(0)
            img = image_batch[i].unsqueeze(0)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            _ = model(int_feat, img)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
    
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    
    print(f"Mean inference time: {mean_time:.4f} ms")
    print(f"Std inference time:  {std_time:.4f} ms")
    
    return {
        'image_size': image_size,
        'num_params': num_params,
        'mean_time': mean_time,
        'std_time': std_time,
        'all_times': inference_times
    }


def run_comprehensive_test():
    """
    Run comprehensive inference speed tests across multiple image sizes
    and generate visualizations.
    """
    # Configuration
    image_sizes = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    num_actions = 6
    num_samples = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'#'*60}")
    print(f"# DQN Inference Speed Benchmark")
    print(f"# Device: {device}")
    print(f"# Image sizes: {image_sizes[0]}x{image_sizes[0]} to {image_sizes[-1]}x{image_sizes[-1]}")
    print(f"# Number of samples per test: {num_samples}")
    print(f"{'#'*60}\n")
    
    # Store results
    results = []
    
    # Test each image size
    for img_size in image_sizes:
        try:
            result = test_dqn_inference_speed(
                image_size=img_size,
                num_actions=num_actions,
                device=device,
                num_samples=num_samples
            )
            results.append(result)
        except Exception as e:
            print(f"Error testing image size {img_size}: {e}")
            continue
    
    # Extract data for plotting
    image_sizes_tested = [r['image_size'] for r in results]
    num_params_list = [r['num_params'] for r in results]
    mean_times = [r['mean_time'] for r in results]
    std_times = [r['std_time'] for r in results]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DQN Inference Speed Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Inference Time vs Image Size
    ax1 = axes[0, 0]
    ax1.errorbar(image_sizes_tested, mean_times, yerr=std_times, 
                 marker='o', linewidth=2, markersize=8, capsize=5, capthick=2)
    ax1.set_xlabel('Image Size (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Inference Time vs Image Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(image_sizes_tested)
    ax1.set_xticklabels([f'{s}x{s}' for s in image_sizes_tested], rotation=45)
    
    # Plot 2: Number of Parameters vs Image Size
    ax2 = axes[0, 1]
    ax2.plot(image_sizes_tested, num_params_list, marker='s', 
             linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Image Size (pixels)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax2.set_title('Model Parameters vs Image Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(image_sizes_tested)
    ax2.set_xticklabels([f'{s}x{s}' for s in image_sizes_tested], rotation=45)
    # Format y-axis to show parameter count in millions
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))
    
    # Plot 3: Inference Time vs Number of Parameters
    ax3 = axes[1, 0]
    ax3.errorbar(num_params_list, mean_times, yerr=std_times,
                 marker='D', linewidth=2, markersize=8, capsize=5, 
                 capthick=2, color='red')
    ax3.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Inference Time vs Model Parameters', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))
    
    # Plot 4: Log-scale comparison
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.semilogy(image_sizes_tested, mean_times, marker='o', 
                         linewidth=2, markersize=8, color='blue', label='Inference Time')
    ax4.set_xlabel('Image Size (pixels)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Inference Time (ms) [log scale]', fontsize=12, 
                   fontweight='bold', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')
    
    line2 = ax4_twin.semilogy(image_sizes_tested, num_params_list, marker='s', 
                              linewidth=2, markersize=8, color='orange', 
                              label='Num Parameters')
    ax4_twin.set_ylabel('Number of Parameters [log scale]', fontsize=12, 
                        fontweight='bold', color='orange')
    ax4_twin.tick_params(axis='y', labelcolor='orange')
    
    ax4.set_title('Log-Scale Comparison', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(image_sizes_tested)
    ax4.set_xticklabels([f'{s}x{s}' for s in image_sizes_tested], rotation=45)
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    current_path = os.getcwd()
    output_path = current_path + '/tests/inf_performance/dqn_inference_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Plot saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Image Size':<15} {'Parameters':<20} {'Mean Time (ms)':<20} {'Std Time (ms)':<20}")
    print("-"*80)
    for r in results:
        print(f"{r['image_size']}x{r['image_size']:<10} "
              f"{r['num_params']:>15,}    "
              f"{r['mean_time']:>15.4f}    "
              f"{r['std_time']:>15.4f}")
    print("="*80)
    
    # Save results to file
    results_path = current_path +'/tests/inf_performance/dqn_inference_results.txt'
    with open(results_path, 'w') as f:
        f.write("DQN Inference Speed Benchmark Results\n")
        f.write("="*80 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of samples per test: {num_samples}\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Image Size':<15} {'Parameters':<20} {'Mean Time (ms)':<20} {'Std Time (ms)':<20}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r['image_size']}x{r['image_size']:<10} "
                   f"{r['num_params']:>15,}    "
                   f"{r['mean_time']:>15.4f}    "
                   f"{r['std_time']:>15.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_test()
    print("\nBenchmark complete!")