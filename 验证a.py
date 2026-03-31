import numpy as np
import SimpleITK as sitk


FIXED_PATH = r"D:\lyg\pythonProject1\pct_fixed.nii.gz"
MOVING_PATH = r"D:\lyg\pythonProject1\cbct_moving.nii.gz"
REGISTERED_OLD_PATH = r"D:\lyg\pythonProject1\cbct_registered_to_pct.nii.gz"

REGISTERED_ROI_PATH = r"D:\lyg\pythonProject1\roi_registration_result\cbct_registered_to_pct_roi.nii.gz"

FIXED_AFFINE_PATH = r"D:\lyg\pythonProject1\affine_registration_result\pct_fixed_affine_ref.nii.gz"
REGISTERED_AFFINE_PATH = r"D:\lyg\pythonProject1\affine_registration_result\cbct_registered_to_pct_affine.nii.gz"

DIFF_BEFORE_PATH = r"D:\lyg\pythonProject1\diff_before_again.nii.gz"
DIFF_OLD_PATH = r"D:\lyg\pythonProject1\diff_old_registration.nii.gz"
DIFF_ROI_PATH = r"D:\lyg\pythonProject1\diff_roi_registration_again.nii.gz"
DIFF_AFFINE_PATH = r"D:\lyg\pythonProject1\diff_affine_registration.nii.gz"



def load_image(path: str) -> sitk.Image:
    return sitk.ReadImage(path)


def check_same_geometry(img1: sitk.Image, img2: sitk.Image, name1="img1", name2="img2") -> None:
    if img1.GetSize() != img2.GetSize():
        raise ValueError(f"{name1} 和 {name2} 的 Size 不一致: {img1.GetSize()} vs {img2.GetSize()}")
    if img1.GetSpacing() != img2.GetSpacing():
        raise ValueError(f"{name1} 和 {name2} 的 Spacing 不一致: {img1.GetSpacing()} vs {img2.GetSpacing()}")
    if img1.GetOrigin() != img2.GetOrigin():
        raise ValueError(f"{name1} 和 {name2} 的 Origin 不一致: {img1.GetOrigin()} vs {img2.GetOrigin()}")
    if img1.GetDirection() != img2.GetDirection():
        raise ValueError(f"{name1} 和 {name2} 的 Direction 不一致: {img1.GetDirection()} vs {img2.GetDirection()}")


def to_array(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]


def resample_to_fixed_identity(fixed_img: sitk.Image, moving_img: sitk.Image) -> sitk.Image:
    identity = sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(
        moving_img,
        fixed_img,
        identity,
        sitk.sitkLinear,
        0.0,
        moving_img.GetPixelID()
    )



def build_valid_mask(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return (arr1 > 0) | (arr2 > 0)


def normalized_cross_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))

    if denominator == 0:
        return float("nan")

    return float(numerator / denominator)


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def dice_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.sum(mask1 & mask2)
    denom = np.sum(mask1) + np.sum(mask2)
    if denom == 0:
        return float("nan")
    return float(2.0 * intersection / denom)


def evaluate_pair(
    fixed_img: sitk.Image,
    other_img: sitk.Image,
    diff_save_path: str,
    bone_threshold_fixed: float = 300.0,
    bone_threshold_other: float = 300.0,
):
    check_same_geometry(fixed_img, other_img, "fixed", "other")

    fixed = to_array(fixed_img)
    other = to_array(other_img)

    valid_mask = build_valid_mask(fixed, other)

    fixed_valid = fixed[valid_mask]
    other_valid = other[valid_mask]

    ncc_value = normalized_cross_correlation(fixed_valid, other_valid)
    mse_value = mse(fixed_valid, other_valid)
    mae_value = mae(fixed_valid, other_valid)

    bone_fixed = (fixed > bone_threshold_fixed) & valid_mask
    bone_other = (other > bone_threshold_other) & valid_mask
    dice_bone = dice_score(bone_fixed, bone_other)

    diff = np.abs(fixed - other)
    diff_img = sitk.GetImageFromArray(diff.astype(np.float32))
    diff_img.CopyInformation(fixed_img)
    sitk.WriteImage(diff_img, diff_save_path)

    return {
        "valid_voxels": int(np.sum(valid_mask)),
        "ncc": ncc_value,
        "mse": mse_value,
        "mae": mae_value,
        "bone_dice": dice_bone,
        "diff_path": diff_save_path,
    }


def percent_improvement_higher_better(before: float, after: float) -> float:
    if np.isnan(before) or np.isnan(after):
        return float("nan")
    if abs(before) < 1e-12:
        return float("nan")
    return (after - before) / abs(before) * 100.0


def percent_improvement_lower_better(before: float, after: float) -> float:
    if np.isnan(before) or np.isnan(after):
        return float("nan")
    if abs(before) < 1e-12:
        return float("nan")
    return (before - after) / abs(before) * 100.0


def print_result_block(title: str, result: dict):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"有效体素数: {result['valid_voxels']}")
    print("-" * 72)
    print(f"NCC       : {result['ncc']:.6f}   （越接近 1 越好）")
    print(f"MSE       : {result['mse']:.6f}   （越小越好）")
    print(f"MAE       : {result['mae']:.6f}   （越小越好）")
    print(f"Bone Dice : {result['bone_dice']:.6f}   （越接近 1 越好，粗指标）")
    print("-" * 72)
    print(f"差值图已保存: {result['diff_path']}")



if __name__ == "__main__":
    print("读取图像 ...")
    fixed_img = load_image(FIXED_PATH)
    moving_img = load_image(MOVING_PATH)
    registered_old_img = load_image(REGISTERED_OLD_PATH)
    registered_roi_img = load_image(REGISTERED_ROI_PATH)

    fixed_affine_img = load_image(FIXED_AFFINE_PATH)
    registered_affine_img = load_image(REGISTERED_AFFINE_PATH)

    print("把配准前 CBCT 用恒等变换重采样到 pCT 空间 ...")
    moving_resampled = resample_to_fixed_identity(fixed_img, moving_img)

    print("评估：配准前 ...")
    before_result = evaluate_pair(
        fixed_img=fixed_img,
        other_img=moving_resampled,
        diff_save_path=DIFF_BEFORE_PATH,
    )

    print("评估：旧版整图 rigid 配准后 ...")
    old_result = evaluate_pair(
        fixed_img=fixed_img,
        other_img=registered_old_img,
        diff_save_path=DIFF_OLD_PATH,
    )

    print("评估：ROI rigid 配准后 ...")
    roi_result = evaluate_pair(
        fixed_img=fixed_img,
        other_img=registered_roi_img,
        diff_save_path=DIFF_ROI_PATH,
    )

    print("评估：ROI affine 配准后 ...")
    affine_result = evaluate_pair(
        fixed_img=fixed_affine_img,
        other_img=registered_affine_img,
        diff_save_path=DIFF_AFFINE_PATH,
    )

    print()
    print_result_block("配准前（pct_fixed vs cbct_moving）", before_result)
    print()
    print_result_block("旧版整图 rigid 后", old_result)
    print()
    print_result_block("ROI rigid 后", roi_result)
    print()
    print_result_block("ROI affine 后", affine_result)

    print("\n" + "=" * 72)
    print("相对“配准前”的改善幅度（正数表示更好）")
    print("=" * 72)

    print("\n--- 旧版整图 rigid 相对配准前 ---")
    print(f"NCC 改善       : {percent_improvement_higher_better(before_result['ncc'], old_result['ncc']):.2f}%")
    print(f"MSE 改善       : {percent_improvement_lower_better(before_result['mse'], old_result['mse']):.2f}%")
    print(f"MAE 改善       : {percent_improvement_lower_better(before_result['mae'], old_result['mae']):.2f}%")
    print(f"Bone Dice 改善 : {percent_improvement_higher_better(before_result['bone_dice'], old_result['bone_dice']):.2f}%")

    print("\n--- ROI rigid 相对配准前 ---")
    print(f"NCC 改善       : {percent_improvement_higher_better(before_result['ncc'], roi_result['ncc']):.2f}%")
    print(f"MSE 改善       : {percent_improvement_lower_better(before_result['mse'], roi_result['mse']):.2f}%")
    print(f"MAE 改善       : {percent_improvement_lower_better(before_result['mae'], roi_result['mae']):.2f}%")
    print(f"Bone Dice 改善 : {percent_improvement_higher_better(before_result['bone_dice'], roi_result['bone_dice']):.2f}%")

    print("\n--- ROI affine 相对配准前 ---")
    print(f"NCC 改善       : {percent_improvement_higher_better(before_result['ncc'], affine_result['ncc']):.2f}%")
    print(f"MSE 改善       : {percent_improvement_lower_better(before_result['mse'], affine_result['mse']):.2f}%")
    print(f"MAE 改善       : {percent_improvement_lower_better(before_result['mae'], affine_result['mae']):.2f}%")
    print(f"Bone Dice 改善 : {percent_improvement_higher_better(before_result['bone_dice'], affine_result['bone_dice']):.2f}%")

    print("\n" + "=" * 72)
    print("ROI affine 相对 ROI rigid 的进一步改善（正数表示更好）")
    print("=" * 72)
    print(f"NCC 改善       : {percent_improvement_higher_better(roi_result['ncc'], affine_result['ncc']):.2f}%")
    print(f"MSE 改善       : {percent_improvement_lower_better(roi_result['mse'], affine_result['mse']):.2f}%")
    print(f"MAE 改善       : {percent_improvement_lower_better(roi_result['mae'], affine_result['mae']):.2f}%")
    print(f"Bone Dice 改善 : {percent_improvement_higher_better(roi_result['bone_dice'], affine_result['bone_dice']):.2f}%")

    print("\n提示：")
    print("1. pCT vs CBCT 的绝对数值不会像同模态那样漂亮。")
    print("2. 真正重要的是：ROI affine 是否比 ROI rigid 再进一步改善。")
    print("3. 最后还要结合 Slicer 看骨性结构是否更贴合。")