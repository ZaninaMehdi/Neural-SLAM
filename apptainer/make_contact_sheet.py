# Puts the frames saved by physics_render_test.py side by side, with a label on each.
# The pixels themselves are not changed; RGB and depth rows alternate.
#
#   apptainer/ans.sh python apptainer/make_contact_sheet.py [frames_dir] [out.png]
#
# Defaults: frames_dir = ~/scratch/ans_runs/physics_test, out = <frames_dir>/contact_sheet.png
# Light enough to run on a login node through the container:
#   module load apptainer
#   apptainer exec --cleanenv neural-slam.sif /opt/conda/envs/ans/bin/python apptainer/make_contact_sheet.py
import os, sys
import cv2
import numpy as np

frames = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                         else os.path.expanduser("~/scratch/ans_runs/physics_test"))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(frames, "contact_sheet.png")

# (file prefix, label); the x offsets match the 5 steps in physics_render_test.py
FRAMES = [("0_baseline", "no box"), ("1_object", "box at 1.5 m"), ("2_move_0", "x=-0.8 m"),
          ("2_move_2", "x=0"), ("2_move_4", "x=+0.8 m"), ("3_removed", "box removed")]
PER_ROW = 3


def label(img, text):
    bar = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def load(name, kind):
    path = os.path.join(frames, "%s_%s.png" % (name, kind))
    img = cv2.imread(path)  # depth PNGs are greyscale; imread gives 3 channels either way
    if img is None:
        sys.exit("missing frame: %s" % path)
    return img


rows = []
for i in range(0, len(FRAMES), PER_ROW):
    chunk = FRAMES[i:i + PER_ROW]
    rows.append(np.hstack([label(load(n, "rgb"), "RGB: " + t) for n, t in chunk]))
    rows.append(np.hstack([label(load(n, "depth"), "depth: " + t) for n, t in chunk]))

cv2.imwrite(out, np.vstack(rows))
print("wrote", out)
