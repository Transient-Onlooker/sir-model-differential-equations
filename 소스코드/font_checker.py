import tkinter
from tkinter import font as tkFont
import matplotlib.font_manager
import platform

def check_fonts():
    print("--- Font Diagnostics ---")
    print(f"Operating System: {platform.system()}\n")

    # 1. Check fonts recognized by Tkinter
    print("1. Fonts recognized by Tkinter GUI library:")
    try:
        tk_root = tkinter.Tk()
        available_tk_fonts = sorted([f.lower() for f in tkFont.families(tk_root)])
        tk_root.destroy()
        print(f"   Found {len(available_tk_fonts)} font families.")
    except Exception as e:
        print(f"   Could not get Tkinter fonts. Error: {e}")
        available_tk_fonts = []

    # 2. Check fonts recognized by Matplotlib
    print("\n2. Fonts recognized by Matplotlib plotting library:")
    try:
        available_mpl_fonts = sorted([f.name.lower() for f in matplotlib.font_manager.fontManager.ttflist])
        print(f"   Found {len(available_mpl_fonts)} font files.")
    except Exception as e:
        print(f"   Could not get Matplotlib fonts. Error: {e}")
        available_mpl_fonts = []

    # 3. Check for specific Korean fonts
    print("\n3. Checking for specific Korean fonts...")
    korean_fonts_to_check = ["malgun gothic", "gulim", "dotum", "batang", "nanumgothic", "apple sd gothic neo", "applemyungjo"]
    
    print("   - In Tkinter:")
    found_in_tk = False
    for font in korean_fonts_to_check:
        if font in available_tk_fonts:
            print(f"     [Found] {font}")
            found_in_tk = True
    if not found_in_tk:
        print("     [Not Found] No common Korean fonts were found by Tkinter.")

    print("\n   - In Matplotlib:")
    found_in_mpl = False
    for font in korean_fonts_to_check:
        if font in available_mpl_fonts:
            print(f"     [Found] {font}")
            found_in_mpl = True
    if not found_in_mpl:
        print("     [Not Found] No common Korean fonts were found by Matplotlib.")

    print("\n--- End of Diagnostics ---")
    if not found_in_tk and not found_in_mpl:
        print("\nConclusion: No Korean fonts were detected. Please install a Korean font (like Nanum Gothic) and try again.")
    elif not found_in_tk or not found_in_mpl:
        print("\nConclusion: Korean fonts were only partially detected. There might be a library-specific issue.")
    else:
        print("\nConclusion: Korean fonts were detected. The issue might be with font caching or configuration.")

if __name__ == "__main__":
    check_fonts()
