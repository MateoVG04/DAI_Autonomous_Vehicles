import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from pathlib import Path
p = Path("/tmp/pycharm_project_481/Data/Images/Town01/Town01_1024_320_40.024_1.118_-4.336.png")
print(p.resolve())
print("exists:", p.exists())

img_fn = str(p)
img = cv2.imread(img_fn)
# opencv (cv2) stores colors in the order blue, green, red, but we want red, green, blue
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.xlabel("$u$") # horizontal pixel coordinate
plt.ylabel("$v$") # vertical pixel coordinate
plt.show()

print("(H,W,3)=",img.shape)