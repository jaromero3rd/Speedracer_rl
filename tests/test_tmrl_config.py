"""
TMRL Configuration Checker
Checks common configuration issues that could cause OpenPlanet connection problems
"""

import os
import sys
import json


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"  ✓ {description}")
        print(f"    Path: {filepath}")
        return True
    else:
        print(f"  ✗ {description} NOT FOUND")
        print(f"    Expected path: {filepath}")
        return False


def check_tmrl_config():
    """Check TMRL configuration files"""
    print("\n" + "="*80)
    print("CHECKING TMRL CONFIGURATION")
    print("="*80 + "\n")
    
    # Try to find TMRL config directory
    possible_config_dirs = [
        os.path.expanduser("~/.tmrl"),
        os.path.join(os.getcwd(), "config"),
        os.path.join(os.getcwd(), ".tmrl"),
    ]
    
    config_dir = None
    for dir_path in possible_config_dirs:
        if os.path.exists(dir_path):
            config_dir = dir_path
            print(f"✓ Found TMRL config directory: {config_dir}\n")
            break
    
    if not config_dir:
        print("✗ Could not find TMRL config directory")
        print("\nSearched in:")
        for dir_path in possible_config_dirs:
            print(f"  - {dir_path}")
        return False
    
    # Check for config files
    config_file = os.path.join(config_dir, "config.json")
    if check_file_exists(config_file, "Main config file"):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("\n  Configuration contents:")
            for key, value in config.items():
                print(f"    {key}: {value}")
        except Exception as e:
            print(f"  ⚠ Could not read config file: {e}")
    
    print()
    return True


def check_openplanet_script():
    """Check for OpenPlanet script"""
    print("\n" + "="*80)
    print("CHECKING OPENPLANET SCRIPT")
    print("="*80 + "\n")
    
    print("OpenPlanet script should be located in:")
    print("  C:\\Users\\<YourUsername>\\OpenplanetNext\\Scripts\\")
    print("\nLook for a file named 'tmrl.op' or similar.")
    print("\nManual check required:")
    print("  □ OpenPlanet script is installed")
    print("  □ Script is enabled in OpenPlanet menu")
    print("  □ Script shows overlay in-game")


def check_network_settings():
    """Check network/firewall settings"""
    print("\n" + "="*80)
    print("CHECKING NETWORK SETTINGS")
    print("="*80 + "\n")
    
    print("TMRL uses these default ports:")
    print("  - Port 9000: OpenPlanet data communication")
    print("  - Port 8081: LIDAR data (if used)")
    print("\nManual checks:")
    print("  □ Firewall allows Python on these ports")
    print("  □ No other application using port 9000")
    print("  □ Windows Defender not blocking connection")


def check_trackmania_status():
    """Check TrackMania process"""
    print("\n" + "="*80)
    print("CHECKING TRACKMANIA STATUS")
    print("="*80 + "\n")
    
    try:
        import psutil
        
        tm_processes = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'trackmania' in proc.info['name'].lower() or 'tm' in proc.info['name'].lower():
                    tm_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if tm_processes:
            print("✓ TrackMania process(es) found:")
            for proc in tm_processes:
                print(f"  - {proc['name']} (PID: {proc['pid']})")
        else:
            print("✗ No TrackMania process found")
            print("  Please start TrackMania 2020")
        
    except ImportError:
        print("⚠ psutil not installed, cannot check process")
        print("  Install with: pip install psutil")
        print("\nManual check:")
        print("  □ TrackMania 2020 is running")


def print_troubleshooting_guide():
    """Print troubleshooting steps"""
    print("\n" + "="*80)
    print("TROUBLESHOOTING GUIDE")
    print("="*80 + "\n")
    
    print("If OpenPlanet is not sending data, try these steps IN ORDER:\n")
    
    print("1. BASIC CHECKS:")
    print("   □ TrackMania 2020 is running (not TrackMania Nations/United)")
    print("   □ You are IN A MAP (not in menus)")
    print("   □ OpenPlanet is installed and shows overlay")
    print("   □ TMRL OpenPlanet script is loaded and enabled")
    
    print("\n2. RESTART SEQUENCE:")
    print("   a. Close TrackMania completely")
    print("   b. Close any Python scripts")
    print("   c. Start TrackMania")
    print("   d. Load into a map")
    print("   e. Verify OpenPlanet overlay appears")
    print("   f. Run your Python script")
    
    print("\n3. OPENPLANET SCRIPT:")
    print("   □ Script file is in OpenPlanet Scripts folder")
    print("   □ Script is enabled in OpenPlanet menu (F3 in game)")
    print("   □ Check OpenPlanet console for errors")
    
    print("\n4. FIREWALL/ANTIVIRUS:")
    print("   □ Allow Python through Windows Firewall")
    print("   □ Allow TrackMania through Windows Firewall")
    print("   □ Temporarily disable antivirus to test")
    
    print("\n5. PORT CONFLICTS:")
    print("   □ Nothing else using port 9000")
    print("   □ Try changing port in TMRL config if needed")
    
    print("\n6. REINSTALL:")
    print("   □ Reinstall OpenPlanet")
    print("   □ Reinstall TMRL: pip install --upgrade tmrl")
    print("   □ Reinstall TMRL OpenPlanet script")
    
    print("\n7. CHECK LOGS:")
    print("   □ OpenPlanet console (F3 in game)")
    print("   □ TMRL logs (if they exist)")
    print("   □ Windows Event Viewer for crashes")


def run_all_checks():
    """Run all configuration checks"""
    print("\n" + "█"*80)
    print("  TMRL CONFIGURATION CHECKER")
    print("█"*80)
    
    check_tmrl_config()
    check_openplanet_script()
    check_network_settings()
    check_trackmania_status()
    print_troubleshooting_guide()
    
    print("\n" + "█"*80)
    print("  CHECKS COMPLETE")
    print("█"*80 + "\n")


if __name__ == "__main__":
    try:
        run_all_checks()
    except KeyboardInterrupt:
        print("\n\nChecks interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during checks: {e}")
        import traceback
        traceback.print_exc()