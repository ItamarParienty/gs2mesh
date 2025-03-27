import os
import shutil

scans_root_dir = "/home/itamarp/gs2mesh/data/DTU_train"
imgs_dir = "/home/itamarp/gs2mesh/scans"
scans = os.listdir(scans_root_dir)
for scan in scans:
    src_img_path = os.path.join(scans_root_dir, scan, "images", "rect_001_max.png")
    dst_img_path = os.path.join(imgs_dir, scan + ".png")
    shutil.copy2(src_img_path, dst_img_path)
