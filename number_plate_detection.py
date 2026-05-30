import cv2
import numpy as np
import os

# -----------------------------------
# Automatic Number Plate Detection
# -----------------------------------

IMAGE_PATH = "car.jpg"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Read image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot find image: {IMAGE_PATH}")

original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Improve contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)

# 3. Reduce noise while preserving edges
blur = cv2.bilateralFilter(gray_eq, 11, 17, 17)

# 4. Blackhat morphology
# Highlights dark characters / dark regions on bright plate background
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rect_kernel)

# 5. Compute horizontal gradient
grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
grad_x = np.absolute(grad_x)

min_val, max_val = grad_x.min(), grad_x.max()
grad_x = ((grad_x - min_val) / (max_val - min_val + 1e-6) * 255).astype("uint8")

# 6. Smooth gradient image and threshold
grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
_, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 7. Morphological closing to connect plate-like structures
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
morph = cv2.erode(morph, None, iterations=2)
morph = cv2.dilate(morph, None, iterations=2)

# 8. Restrict search area
# For front vehicle images, plate usually appears in lower-middle region
h, w = gray.shape
roi_mask = np.zeros_like(morph)
cv2.rectangle(
    roi_mask,
    (int(w * 0.15), int(h * 0.45)),
    (int(w * 0.85), int(h * 0.90)),
    255,
    -1
)
roi_result = cv2.bitwise_and(morph, roi_mask)

# 9. Find contours
contours, _ = cv2.findContours(roi_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best_box = None
best_score = -1

for cnt in contours:
    x, y, bw, bh = cv2.boundingRect(cnt)
    area = bw * bh
    aspect_ratio = bw / float(bh)
    contour_area = cv2.contourArea(cnt)
    fill_ratio = contour_area / float(area + 1e-6)

    center_x = x + bw / 2
    center_y = y + bh / 2

    # Basic filtering
    if area < 1000:
        continue
    if not (2.0 <= aspect_ratio <= 6.5):
        continue
    if y < h * 0.45 or y > h * 0.88:
        continue

    # Score candidate
    score = area * fill_ratio

    # Prefer regions close to lower-middle area
    score *= (1.0 - abs(center_x - w / 2) / (w / 2) * 0.4)
    score *= (1.0 - abs(center_y - h * 0.72) / h * 0.8)

    if score > best_score:
        best_score = score
        best_box = (x, y, bw, bh)

# 10. Draw result
candidate_image = original.copy()
final_image = original.copy()

if best_box is not None:
    x, y, bw, bh = best_box

    # Slight padding for final rectangle
    pad_x = int(bw * 0.08)
    pad_y = int(bh * 0.15)

    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + bw + pad_x, w)
    y2 = min(y + bh + pad_y, h)

    cv2.rectangle(candidate_image, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
    cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    plate_crop = original[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "plate_crop.jpg"), plate_crop)

    print("Plate detected at:", (x1, y1, x2, y2))
else:
    print("No plate detected.")

# 11. Save all intermediate outputs
cv2.imwrite(os.path.join(OUTPUT_DIR, "01_original.jpg"), original)
cv2.imwrite(os.path.join(OUTPUT_DIR, "02_gray.jpg"), gray)
cv2.imwrite(os.path.join(OUTPUT_DIR, "03_clahe.jpg"), gray_eq)
cv2.imwrite(os.path.join(OUTPUT_DIR, "04_blackhat.jpg"), blackhat)
cv2.imwrite(os.path.join(OUTPUT_DIR, "05_gradx.jpg"), grad_x)
cv2.imwrite(os.path.join(OUTPUT_DIR, "06_threshold.jpg"), thresh)
cv2.imwrite(os.path.join(OUTPUT_DIR, "07_morphology.jpg"), morph)
cv2.imwrite(os.path.join(OUTPUT_DIR, "08_roi_result.jpg"), roi_result)
cv2.imwrite(os.path.join(OUTPUT_DIR, "09_candidate.jpg"), candidate_image)
cv2.imwrite(os.path.join(OUTPUT_DIR, "10_final_detection.jpg"), final_image)

# 12. Display results
cv2.imshow("Original", original)
cv2.imshow("Gray", gray)
cv2.imshow("CLAHE", gray_eq)
cv2.imshow("Blackhat", blackhat)
cv2.imshow("Gradient X", grad_x)
cv2.imshow("Threshold", thresh)
cv2.imshow("Morphology", morph)
cv2.imshow("ROI Result", roi_result)
cv2.imshow("Candidate Region", candidate_image)
cv2.imshow("Final Detection", final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()