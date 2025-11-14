import subprocess
import time
import socket

def check_trackmania_running():
    """Check if Trackmania process is running."""
    try:
        # 'pgrep' works on Linux; use 'tasklist' on Windows or adapt accordingly
        result = subprocess.run(['pgrep', '-f', 'Trackmania'], capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print("Error checking Trackmania process:", e)
        return False

def check_openplanet_running():
    """Check if OpenPlanet plugin is active by checking log file existence and tailing recent logs."""
    import os
    log_path = "/path/to/Trackmania/Openplanet/OpenplanetHook.log"  # Adjust this path for your setup
    if not os.path.isfile(log_path):
        print("Openplanet log file not found. Openplanet may not be running.")
        return False
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()[-20:]  # Read last 20 lines
            error_lines = [line for line in lines if "error" in line.lower()]
            if error_lines:
                print("Errors found in Openplanet logs:")
                for line in error_lines:
                    print(line.strip())
                return False
    except Exception as e:
        print("Error reading Openplanet log:", e)
        return False
    return True

def check_openplanet_port(port=5000):
    """Check if port used for OpenPlanet communication is open (default port example)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    try:
        sock.connect(('127.0.0.1', port))
        print(f"Port {port} is open - connection to OpenPlanet plugin possible.")
        sock.close()
        return True
    except socket.error as e:
        print(f"Port {port} connection failed: {e}")
        return False

def check_env():
    print("Checking Trackmania process...")
    if not check_trackmania_running():
        print("Trackmania is not running. Please start the game.")
        return
    else:
        print("Trackmania is running correctly")
    print("Checking Openplanet plugin logs...")
    if not check_openplanet_running():
        print("Openplanet plugin is not running correctly or has errors.")
        return
    
    print("Checking Openplanet communication port...")
    if not check_openplanet_port():
        print("Cannot connect to Openplanet plugin port. Check plugin and networking.")
        return

    print("All basic checks passed. Plugin and game appear active and communicating.")

