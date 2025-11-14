"""
Quick OpenPlanet Connection Test
Simple script to quickly test if OpenPlanet is sending data
"""

import time
import sys


def quick_test():
    """Quick test of OpenPlanet connection"""
    print("="*60)
    print("QUICK OPENPLANET CONNECTION TEST")
    print("="*60)
    print("\nBefore running this test, ensure:")
    print("  1. TrackMania 2020 is running")
    print("  2. You are in a map (not menu)")
    print("  3. OpenPlanet is loaded with TMRL script")
    print("\nPress Enter to start test, or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        sys.exit(0)
    
    try:
        print("\nImporting TMInterface...")
        from tmrl.custom.tm.utils.tools import TMInterface
        print("✓ Import successful")
        
        print("\nCreating client...")
        client = TMInterface()
        print("✓ Client created")
        
        print("\nAttempting to retrieve data...")
        print("(Timeout: 10 seconds)")
        
        start = time.time()
        try:
            data = client.retrieve_data(timeout=10.0)
            elapsed = time.time() - start
            print(f"\n✓ SUCCESS! Data received in {elapsed:.2f}s")
            print(f"Data: {data}")
            return True
            
        except AssertionError as e:
            elapsed = time.time() - start
            print(f"\n✗ FAILED after {elapsed:.2f}s")
            print(f"Error: {e}")
            print("\nLikely causes:")
            print("  - TrackMania not running")
            print("  - Not in a map/track")
            print("  - OpenPlanet script not loaded")
            print("  - OpenPlanet not sending data")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def continuous_monitor(duration=60):
    """Monitor OpenPlanet connection continuously"""
    print("\n" + "="*60)
    print("CONTINUOUS MONITORING MODE")
    print("="*60)
    print(f"Monitoring for {duration} seconds...")
    print("Press Ctrl+C to stop\n")
    
    try:
        from tmrl.custom.tm.utils.tools import TMInterface
        client = TMInterface()
        
        success = 0
        failure = 0
        last_data = None
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                data = client.retrieve_data(timeout=2.0)
                success += 1
                
                # Only print if data changed
                if data != last_data:
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:6.2f}s] ✓ Data: {data}")
                    last_data = data
                
                time.sleep(0.1)
                
            except AssertionError:
                failure += 1
                elapsed = time.time() - start_time
                print(f"[{elapsed:6.2f}s] ✗ No data")
                time.sleep(1.0)  # Wait longer after failure
                
            except KeyboardInterrupt:
                break
        
        total = success + failure
        print(f"\n{'─'*60}")
        print(f"Results:")
        print(f"  Success: {success}/{total} ({success/total*100:.1f}%)")
        print(f"  Failure: {failure}/{total} ({failure/total*100:.1f}%)")
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        success = quick_test()
        
        if success:
            print("\n" + "="*60)
            print("Would you like to run continuous monitoring? (y/n)")
            try:
                response = input().strip().lower()
                if response == 'y':
                    continuous_monitor(duration=30)
            except KeyboardInterrupt:
                pass
        
        print("\nTest complete.")
        
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(0)