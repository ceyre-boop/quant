"""
Windows Task Scheduler Setup for Sovereign Daily Execution

Creates scheduled tasks for:
- Pre-market check (8:45 AM ET, Mon-Fri)
- Kill zone execution (9:50 AM ET, Mon-Fri)  
- Post-session report (4:35 PM ET, Mon-Fri)
"""

import subprocess
import sys
from pathlib import Path


def create_task_scheduler_job():
    """
    Create Windows Task Scheduler entries for Sovereign trading.
    Requires administrator privileges to run.
    """
    
    repo_path = Path.cwd()
    python_exe = sys.executable
    script_path = repo_path / "execute_daily.py"
    
    print("=" * 60)
    print("SOVEREIGN TASK SCHEDULER SETUP")
    print("=" * 60)
    print(f"Repo: {repo_path}")
    print(f"Python: {python_exe}")
    print(f"Script: {script_path}")
    print("=" * 60)
    
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    # Task 1: Pre-market Checklist (8:45 AM, Mon-Fri)
    task1_cmd = [
        'schtasks', '/create', '/f',
        '/tn', 'Sovereign_PreMarket',
        '/tr', f'"{python_exe}" "{script_path}" --phase premarket',
        '/sc', 'weekly',
        '/d', 'MON,TUE,WED,THU,FRI',
        '/st', '08:45',
        '/ru', 'SYSTEM'
    ]
    
    # Task 2: Kill Zone Execution (9:50 AM, Mon-Fri)
    task2_cmd = [
        'schtasks', '/create', '/f',
        '/tn', 'Sovereign_KillZone',
        '/tr', f'"{python_exe}" "{script_path}" --phase killzone --mode paper',
        '/sc', 'weekly',
        '/d', 'MON,TUE,WED,THU,FRI',
        '/st', '09:50',
        '/ru', 'SYSTEM'
    ]
    
    # Task 3: Post-Session Report (4:35 PM, Mon-Fri)
    task3_cmd = [
        'schtasks', '/create', '/f',
        '/tn', 'Sovereign_PostSession',
        '/tr', f'"{python_exe}" "{script_path}" --phase postsession',
        '/sc', 'weekly',
        '/d', 'MON,TUE,WED,THU,FRI',
        '/st', '16:35',
        '/ru', 'SYSTEM'
    ]
    
    tasks = [
        ('Pre-Market Checklist (8:45 AM)', task1_cmd),
        ('Kill Zone Execution (9:50 AM)', task2_cmd),
        ('Post-Session Report (4:35 PM)', task3_cmd)
    ]
    
    print("\nCreating tasks...")
    print("-" * 60)
    
    for name, cmd in tasks:
        print(f"\nCreating: {name}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                print(f"[OK] {name} created successfully")
            else:
                print(f"[FAIL] {name}")
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nTo verify tasks were created, run:")
    print("  schtasks /query /tn Sovereign_*")
    print("\nTo delete all tasks:")
    print("  schtasks /delete /tn Sovereign_PreMarket /f")
    print("  schtasks /delete /tn Sovereign_KillZone /f")
    print("  schtasks /delete /tn Sovereign_PostSession /f")
    print("=" * 60)
    
    return True


def verify_tasks():
    """Verify tasks exist in Task Scheduler."""
    print("\nVerifying scheduled tasks...")
    try:
        result = subprocess.run(
            ['schtasks', '/query', '/tn', 'Sovereign_*'],
            capture_output=True,
            text=True,
            shell=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"Error querying tasks: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Verify tasks exist')
    parser.add_argument('--remove', action='store_true', help='Remove all tasks')
    args = parser.parse_args()
    
    if args.verify:
        verify_tasks()
    elif args.remove:
        print("Removing tasks...")
        for task in ['Sovereign_PreMarket', 'Sovereign_KillZone', 'Sovereign_PostSession']:
            subprocess.run(['schtasks', '/delete', '/tn', task, '/f'], capture_output=True)
        print("[OK] Tasks removed")
    else:
        print("\n[NOTE] This script requires Administrator privileges")
        print("Right-click PowerShell → Run as Administrator")
        print("Then run: python setup_task_scheduler.py\n")
        
        response = input("Continue? (yes/no): ")
        if response.lower() == 'yes':
            create_task_scheduler_job()
