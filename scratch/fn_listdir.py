import os, time
D = r"N:\marineai\dataset\raw\fathomnet\images\NOAA_NMFS_SEFSC"
t = time.time(); n = len(os.listdir(D)); e1 = time.time() - t
t = time.time(); n2 = len(os.listdir(D)); e2 = time.time() - t
print(f"{n:,} files")
print(f"  cold listdir {e1:.1f}s   warm {e2:.1f}s")
if n:
    print(f"  projected at 1.47M files: cold {e1*1_470_000/n:.0f}s  "
          f"warm {e2*1_470_000/n:.0f}s")
