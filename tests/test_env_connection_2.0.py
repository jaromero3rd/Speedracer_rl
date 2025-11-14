"""
OpenPlanet Connection Diagnostic Test
This script helps identify issues with OpenPlanet data connection for TrackMania
"""

import time
import sys
from datetime import datetime


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_imports():
    """Test if all required modules can be imported"""
    print_section("1. Testing Imports")
    
    try:
        import tmrl
        print("✓ tmrl imported successfully")
        print(f"  Version: {tmrl.__version__ if hasattr(tmrl, '__version__') else 'Unknown'}")
    except ImportError as e:
        print(f"✗ Failed to import tmrl: {e}")
        return False
    
    try:
        from tmrl import get_environment
        print("✓ get_environment imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import get_environment: {e}")
        return False
    
    try:
        import rtgym
        print("✓ rtgym imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import rtgym: {e}")
        return False
    
    return True


def test_openplanet_client():
    """Test direct connection to OpenPlanet client"""
    print_section("2. Testing OpenPlanet Client Connection")
    
    try:
        from tmrl.custom.tm.utils.tools import TMInterface
        print("✓ TMInterface imported successfully")
        
        print("\nAttempting to create TMInterface client...")
        client = TMInterface()
        print("✓ TMInterface client created")
        
        print("\nTesting data retrieval (10 second timeout)...")
        start_time = time.time()
        try:
            data = client.retrieve_data(timeout=10.0)
            elapsed = time.time() - start_time
            print(f"✓ Data retrieved successfully in {elapsed:.2f}s")
            print(f"  Data type: {type(data)}")
            print(f"  Data: {data}")
            return True
        except AssertionError as e:
            elapsed = time.time() - start_time
            print(f"✗ Data retrieval failed after {elapsed:.2f}s")
            print(f"  Error: {e}")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Unexpected error after {elapsed:.2f}s: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Failed to import TMInterface: {e}")
        return False
    except Exception as e:
        print(f"✗ Error creating TMInterface: {e}")
        return False


def test_continuous_data_stream(duration=30):
    """Test continuous data stream from OpenPlanet"""
    print_section("3. Testing Continuous Data Stream")
    
    try:
        from tmrl.custom.tm.utils.tools import TMInterface
        
        print(f"Testing continuous data stream for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        client = TMInterface()
        
        success_count = 0
        failure_count = 0
        timestamps = []
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                try:
                    data = client.retrieve_data(timeout=2.0)
                    success_count += 1
                    current_time = time.time() - start_time
                    timestamps.append(current_time)
                    
                    # Print progress every second
                    if success_count % 10 == 0 or success_count == 1:
                        print(f"[{current_time:6.2f}s] ✓ Successful retrieval #{success_count} - Data: {data}")
                    
                    time.sleep(0.1)  # Small delay between requests
                    
                except AssertionError as e:
                    failure_count += 1
                    current_time = time.time() - start_time
                    print(f"[{current_time:6.2f}s] ✗ Failed retrieval #{failure_count} - {e}")
                    
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
        
        total_time = time.time() - start_time
        
        print(f"\n{'─'*80}")
        print("Test Results:")
        print(f"  Duration: {total_time:.2f}s")
        print(f"  Successful retrievals: {success_count}")
        print(f"  Failed retrievals: {failure_count}")
        print(f"  Success rate: {success_count/(success_count+failure_count)*100:.1f}%")
        
        if timestamps:
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            if intervals:
                print(f"  Average interval: {sum(intervals)/len(intervals):.3f}s")
                print(f"  Min interval: {min(intervals):.3f}s")
                print(f"  Max interval: {max(intervals):.3f}s")
        
        return success_count > 0
        
    except Exception as e:
        print(f"✗ Error during continuous test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_creation():
    """Test creating the TMRL environment"""
    print_section("4. Testing Environment Creation")
    
    try:
        from tmrl import get_environment
        
        print("Attempting to create environment...")
        print("(This may take a moment...)\n")
        
        start_time = time.time()
        try:
            env = get_environment()
            elapsed = time.time() - start_time
            print(f"✓ Environment created successfully in {elapsed:.2f}s")
            print(f"  Environment type: {type(env)}")
            return env
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Failed to create environment after {elapsed:.2f}s")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except ImportError as e:
        print(f"✗ Failed to import get_environment: {e}")
        return None


def test_environment_reset(env):
    """Test environment reset"""
    print_section("5. Testing Environment Reset")
    
    if env is None:
        print("✗ No environment provided, skipping test")
        return False
    
    try:
        print("Waiting 1 second before reset...")
        time.sleep(1.0)
        
        print("Attempting environment reset...")
        start_time = time.time()
        
        try:
            obs, info = env.reset()
            elapsed = time.time() - start_time
            print(f"✓ Environment reset successful in {elapsed:.2f}s")
            print(f"  Observation type: {type(obs)}")
            print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
            print(f"  Info: {info}")
            return True
        except AssertionError as e:
            elapsed = time.time() - start_time
            print(f"✗ Environment reset failed after {elapsed:.2f}s")
            print(f"  Error: {e}")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Unexpected error during reset after {elapsed:.2f}s")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Error testing reset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_step(env):
    """Test environment step"""
    print_section("6. Testing Environment Step")
    
    if env is None:
        print("✗ No environment provided, skipping test")
        return False
    
    try:
        import numpy as np
        
        # Test with a simple action
        action = np.array([1.0, 0.0, 0.0])  # Forward, no brake, no steer
        print(f"Testing step with action: {action}")
        
        start_time = time.time()
        try:
            obs, reward, done, truncated, info = env.step(action)
            elapsed = time.time() - start_time
            print(f"✓ Environment step successful in {elapsed:.2f}s")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            print(f"  Truncated: {truncated}")
            print(f"  Info: {info}")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Environment step failed after {elapsed:.2f}s")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Error testing step: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_openplanet_status():
    """Check if OpenPlanet is running and configured"""
    print_section("OpenPlanet Status Check")
    
    print("Manual checks to perform:")
    print("  □ Is TrackMania 2020 running?")
    print("  □ Is OpenPlanet installed?")
    print("  □ Is the TMRL OpenPlanet script loaded?")
    print("  □ Can you see the TMRL overlay in-game?")
    print("  □ Are you in a map/track (not menu)?")
    print("\nIf any of the above are unchecked, that may be the issue.")


def run_full_diagnostic():
    """Run all diagnostic tests"""
    print("\n" + "█"*80)
    print("  OPENPLANET CONNECTION DIAGNOSTIC TEST")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█"*80)
    
    check_openplanet_status()
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n⚠ Import test failed. Please check your installation.")
        return results
    
    # Test 2: OpenPlanet client
    results['client'] = test_openplanet_client()
    if not results['client']:
        print("\n⚠ OpenPlanet client test failed. This is likely the main issue.")
        print("\nTroubleshooting steps:")
        print("  1. Ensure TrackMania 2020 is running")
        print("  2. Ensure OpenPlanet is installed and active")
        print("  3. Load the TMRL OpenPlanet script")
        print("  4. Make sure you're in a map (not in menu)")
        print("  5. Check if firewall is blocking connections")
        print("  6. Try restarting TrackMania and OpenPlanet")
    
    # Test 3: Continuous stream (optional)
    if results['client']:
        print("\nWould you like to test continuous data stream? (y/n)")
        try:
            response = input().strip().lower()
            if response == 'y':
                results['continuous'] = test_continuous_data_stream(duration=15)
        except:
            print("Skipping continuous stream test")
            results['continuous'] = None
    else:
        results['continuous'] = None
    
    # Test 4: Environment creation
    results['env_create'] = False
    results['env_reset'] = False
    results['env_step'] = False
    
    if results['client']:
        env = test_environment_creation()
        results['env_create'] = env is not None
        
        if env is not None:
            results['env_reset'] = test_environment_reset(env)
            
            if results['env_reset']:
                results['env_step'] = test_environment_step(env)
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    
    print("Test Results:")
    print(f"  {'Imports:':<30} {'✓ PASS' if results['imports'] else '✗ FAIL'}")
    print(f"  {'OpenPlanet Client:':<30} {'✓ PASS' if results['client'] else '✗ FAIL'}")
    if results['continuous'] is not None:
        print(f"  {'Continuous Data Stream:':<30} {'✓ PASS' if results['continuous'] else '✗ FAIL'}")
    print(f"  {'Environment Creation:':<30} {'✓ PASS' if results['env_create'] else '✗ FAIL'}")
    print(f"  {'Environment Reset:':<30} {'✓ PASS' if results['env_reset'] else '✗ FAIL'}")
    print(f"  {'Environment Step:':<30} {'✓ PASS' if results['env_step'] else '✗ FAIL'}")
    
    print("\n" + "─"*80)
    if all([v for k, v in results.items() if v is not None]):
        print("✓ All tests passed! OpenPlanet connection is working correctly.")
    else:
        print("✗ Some tests failed. See details above for troubleshooting.")
    
    return results


if __name__ == "__main__":
    try:
        results = run_full_diagnostic()
        
        print("\n" + "█"*80)
        print("  DIAGNOSTIC COMPLETE")
        print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("█"*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error during diagnostic: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)