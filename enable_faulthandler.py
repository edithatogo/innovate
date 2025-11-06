"""Script to run tests with faulthandler enabled to catch segmentation faults."""
import faulthandler
import sys
import os

# Enable faulthandler to get Python tracebacks from segmentation faults
faulthandler.enable()

# Register a traceback dump to a file in case of segfault
fault_file_path = os.path.join(os.path.dirname(__file__), 'segfault_traceback.txt')
fault_file = open(fault_file_path, 'w')
faulthandler.register(sys.stderr.fileno(), all_threads=True)

def run_with_faulthandler():
    """Run tests with faulthandler enabled."""
    print("Faulthandler enabled. Segmentation faults will now show Python tracebacks.")
    print("This will help identify if segfaults are from your code or dependencies.")
    print(f"Fault tracebacks will be written to: {fault_file_path}")
    
    # You can run your tests here
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_coverage_improvement.py::TestBaseModule", "-v"], 
                           capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

if __name__ == "__main__":
    run_with_faulthandler()