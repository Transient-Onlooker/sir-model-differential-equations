import matplotlib
import os
import shutil

print("Attempting to clear the Matplotlib font cache...")
try:
    cache_dir = matplotlib.get_cachedir()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"SUCCESS: Matplotlib font cache at '{cache_dir}' has been deleted.")
        print("\nPlease try running the main simulation program again.")
    else:
        print("INFO: No Matplotlib cache directory found to delete.")
except Exception as e:
    print(f"ERROR: An error occurred: {e}")
    print("Could not delete the font cache automatically.")
