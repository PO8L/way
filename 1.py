import pydicom
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\lyg\xwechat_files\wxid_qnea4ikicpvm22_75eb\msg\file\2026-03\M24557-CBCT\M24557-CBCT\M24557_CT3_image00000.DCM"

ds = pydicom.dcmread(path, force=True)

print("Rows:", ds.Rows)
print("Columns:", ds.Columns)
print("BitsAllocated:", ds.BitsAllocated)
print("PixelRepresentation:", ds.PixelRepresentation)  # 0=unsigned, 1=signed


dtype = np.uint16 if ds.PixelRepresentation == 0 else np.int16
img = np.frombuffer(ds.PixelData, dtype=dtype)

expected = ds.Rows * ds.Columns
print("Expected pixels:", expected)
print("Actual pixels:", img.size)

img = img.reshape(ds.Rows, ds.Columns)

print("min:", img.min(), "max:", img.max(), "mean:", img.mean())

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()