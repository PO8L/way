import os
import numpy as np
import pydicom
import SimpleITK as sitk


CBCT_FOLDER = r"C:\Users\lyg\xwechat_files\wxid_qnea4ikicpvm22_75eb\msg\file\2026-03\M24557-CBCT\M24557-CBCT"
PCT_FOLDER  = r"C:\Users\lyg\xwechat_files\wxid_qnea4ikicpvm22_75eb\msg\file\2026-03\M24557-CBCT\M24557-pCT"

OUT_DIR = r"D:\lyg\pythonProject1\affine_registration_result"
os.makedirs(OUT_DIR, exist_ok=True)



def decode_dicom_pixel(ds):
    rows = int(ds.Rows)
    cols = int(ds.Columns)
    bits_allocated = int(ds.BitsAllocated)
    pixel_repr = int(getattr(ds, "PixelRepresentation", 0))

    if bits_allocated == 8:
        dtype = np.uint8 if pixel_repr == 0 else np.int8
    elif bits_allocated == 16:
        dtype = np.uint16 if pixel_repr == 0 else np.int16
    else:
        raise ValueError(f"Unsupported BitsAllocated={bits_allocated}")

    img = np.frombuffer(ds.PixelData, dtype=dtype)
    expected = rows * cols
    if img.size != expected:
        raise ValueError(
            f"像素数量不匹配: expected={expected}, actual={img.size}, rows={rows}, cols={cols}"
        )

    img = img.reshape(rows, cols)
    return img



def load_series_as_sitk(folder):
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(".dcm")
    ]

    if not files:
        raise RuntimeError(f"目录里没有 DICOM 文件: {folder}")

    records = []
    for f in files:
        path = os.path.join(folder, f)
        ds = pydicom.dcmread(path, force=True)

        if not hasattr(ds, "ImagePositionPatient"):
            raise ValueError(f"{f} 缺少 ImagePositionPatient，没法稳妥排序")

        z = float(ds.ImagePositionPatient[2])
        img = decode_dicom_pixel(ds)
        records.append((z, f, ds, img))

    records.sort(key=lambda x: x[0])

    volume = np.stack([r[3] for r in records], axis=0)  # [z, y, x]
    first_ds = records[0][2]

    pixel_spacing = [float(v) for v in first_ds.PixelSpacing]
    spacing_x = pixel_spacing[0]
    spacing_y = pixel_spacing[1]

    if len(records) > 1:
        z_positions = [r[0] for r in records]
        dz = np.diff(z_positions)
        spacing_z = float(np.median(np.abs(dz)))
    else:
        spacing_z = float(getattr(first_ds, "SliceThickness", 1.0))

    origin = [float(v) for v in first_ds.ImagePositionPatient]


    direction = (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )

    sitk_img = sitk.GetImageFromArray(volume.astype(np.float32))  # [z,y,x]
    sitk_img.SetSpacing((spacing_x, spacing_y, spacing_z))
    sitk_img.SetOrigin(origin)
    sitk_img.SetDirection(direction)

    return sitk_img, records



def resample_with_initial_geometry_transform(fixed_img, moving_img):
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    resampled = sitk.Resample(
        moving_img,
        fixed_img,
        initial_transform,
        sitk.sitkLinear,
        0.0,
        moving_img.GetPixelID()
    )

    return resampled, initial_transform



def make_body_mask(img, threshold=1.0):
    arr = sitk.GetArrayFromImage(img)
    max_val = float(arr.max())

    if max_val <= threshold:
        raise RuntimeError(
            f"body mask 为空：整幅图最大值只有 {max_val:.3f}，通常说明图像几乎全空或几何没对上"
        )

    mask = sitk.BinaryThreshold(
        img,
        lowerThreshold=threshold,
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0
    )

    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 1])

    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    if stats.GetNumberOfLabels() == 0:
        raise RuntimeError("body mask 为空，阈值可能不合适，或者图像几乎没重叠")

    largest_label = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    largest = sitk.BinaryThreshold(
        cc,
        lowerThreshold=largest_label,
        upperThreshold=largest_label,
        insideValue=1,
        outsideValue=0
    )

    return largest



def get_bbox_with_margin(mask, margin=(20, 30, 30)):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)

    labels = stats.GetLabels()
    if len(labels) == 0:
        raise RuntimeError("mask 中没有标签，无法获取 bbox")

    bbox = list(stats.GetBoundingBox(labels[0]))  # [x, y, z, sx, sy, sz]
    x, y, z, sx, sy, sz = bbox
    mx, my, mz = margin

    img_size = mask.GetSize()

    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    z0 = max(0, z - mz)

    x1 = min(img_size[0], x + sx + mx)
    y1 = min(img_size[1], y + sy + my)
    z1 = min(img_size[2], z + sz + mz)

    return (x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)


def crop_with_bbox(img, bbox):
    x, y, z, sx, sy, sz = bbox
    return sitk.RegionOfInterest(img, size=[sx, sy, sz], index=[x, y, z])



def register_affine_roi(fixed_img, moving_img):
    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Affine 变换：12自由度
    initial_transform = sitk.AffineTransform(3)

    fixed_center = np.array(
        fixed_img.TransformContinuousIndexToPhysicalPoint(
            np.array(fixed_img.GetSize()) / 2.0
        )
    )
    initial_transform.SetCenter(tuple(fixed_center.tolist()))

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.5,
        numberOfIterations=400,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(fixed_img, moving_img)

    print("\nROI affine 配准完成")
    print("Final metric value:", registration_method.GetMetricValue())
    print("Optimizer stop condition:", registration_method.GetOptimizerStopConditionDescription())

    return final_transform



def resample_to_fixed(fixed_img, moving_img, transform):
    return sitk.Resample(
        moving_img,
        fixed_img,
        transform,
        sitk.sitkLinear,
        0.0,
        moving_img.GetPixelID()
    )



if __name__ == "__main__":
    print("读取 pCT ...")
    pct_img, pct_records = load_series_as_sitk(PCT_FOLDER)

    print("读取 CBCT ...")
    cbct_img, cbct_records = load_series_as_sitk(CBCT_FOLDER)

    print("pCT size:", pct_img.GetSize(), "spacing:", pct_img.GetSpacing(), "origin:", pct_img.GetOrigin())
    print("CBCT size:", cbct_img.GetSize(), "spacing:", cbct_img.GetSpacing(), "origin:", cbct_img.GetOrigin())

    print("\n先用几何中心初始化，把 CBCT 粗略搬到 pCT 空间 ...")
    cbct_in_pct_space, initial_transform = resample_with_initial_geometry_transform(pct_img, cbct_img)

    print("生成 pCT body mask ...")
    pct_mask = make_body_mask(pct_img, threshold=1.0)

    print("生成粗对齐后 CBCT 的 body mask ...")
    cbct_mask = make_body_mask(cbct_in_pct_space, threshold=1.0)

    print("计算两个 mask 的交集，作为重叠 ROI ...")
    overlap_mask = pct_mask & cbct_mask

    overlap_stats = sitk.LabelShapeStatisticsImageFilter()
    overlap_stats.Execute(overlap_mask)

    if overlap_stats.GetNumberOfLabels() == 0:
        print("警告：mask 交集为空，退回使用 pCT 的 body ROI")
        roi_bbox = get_bbox_with_margin(pct_mask, margin=(20, 30, 30))
    else:
        roi_bbox = get_bbox_with_margin(overlap_mask, margin=(20, 30, 30))

    print("ROI bbox:", roi_bbox)

    print("\n裁剪 ROI ...")
    pct_roi = crop_with_bbox(pct_img, roi_bbox)
    cbct_roi = crop_with_bbox(cbct_in_pct_space, roi_bbox)

    print("pct_roi size:", pct_roi.GetSize(), "spacing:", pct_roi.GetSpacing(), "origin:", pct_roi.GetOrigin())
    print("cbct_roi size:", cbct_roi.GetSize(), "spacing:", cbct_roi.GetSpacing(), "origin:", cbct_roi.GetOrigin())

    print("\n开始 ROI affine registration ...")
    roi_affine_transform = register_affine_roi(fixed_img=pct_roi, moving_img=cbct_roi)

    print("\n组合 初始几何变换 + ROI affine 变换 ...")
    composite_transform = sitk.CompositeTransform(3)
    composite_transform.AddTransform(initial_transform)
    composite_transform.AddTransform(roi_affine_transform)

    print("\n把总变换应用回原始 CBCT，并重采样到完整 pCT 空间 ...")
    cbct_registered_full = resample_to_fixed(
        fixed_img=pct_img,
        moving_img=cbct_img,
        transform=composite_transform
    )


    pct_save = os.path.join(OUT_DIR, "pct_fixed_affine_ref.nii.gz")
    cbct_save = os.path.join(OUT_DIR, "cbct_moving_original.nii.gz")
    cbct_reg_save = os.path.join(OUT_DIR, "cbct_registered_to_pct_affine.nii.gz")
    pct_roi_save = os.path.join(OUT_DIR, "pct_roi_affine.nii.gz")
    cbct_roi_save = os.path.join(OUT_DIR, "cbct_roi_before_affine_reg.nii.gz")
    overlap_save = os.path.join(OUT_DIR, "overlap_mask_affine.nii.gz")
    cbct_coarse_save = os.path.join(OUT_DIR, "cbct_coarse_in_pct_space_affine.nii.gz")

    sitk.WriteImage(pct_img, pct_save)
    sitk.WriteImage(cbct_img, cbct_save)
    sitk.WriteImage(cbct_registered_full, cbct_reg_save)
    sitk.WriteImage(pct_roi, pct_roi_save)
    sitk.WriteImage(cbct_roi, cbct_roi_save)
    sitk.WriteImage(sitk.Cast(overlap_mask, sitk.sitkUInt8), overlap_save)
    sitk.WriteImage(cbct_in_pct_space, cbct_coarse_save)

    print("\n已保存：")
    print(pct_save)
    print(cbct_save)
    print(cbct_reg_save)
    print(pct_roi_save)
    print(cbct_roi_save)
    print(overlap_save)
    print(cbct_coarse_save)