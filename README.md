# Automatic-Number-Plate-Detection-Using-Image-Processing
==========================================================================

This repository contains my Practical 3 submission for INTE 41312 – Image Processing and Computer Graphics. The project implements a basic automatic number plate detection system using Python and OpenCV, focusing on classical image processing techniques rather than deep learning.

**Project overview**
In many real-world scenarios such as traffic monitoring, smart parking systems, toll gate automation, and security/surveillance, vehicles must be identified quickly and accurately. Manual number plate reading is slow, error-prone, and not scalable when there are large numbers of vehicles.

This project demonstrates how a carefully designed image processing pipeline can automatically locate the license plate region from a vehicle image, even in the presence of background clutter, lighting variations, shadows, and noise.

**Image processing pipeline**
The detection pipeline consists of the following main steps:

**1. Input vehicle image**
**2. Grayscale conversion** – simplifies the image and reduces computation.
**3. CLAHE contrast enhancement** – improves local contrast under uneven lighting.
**4. Bilateral filtering** – reduces noise while preserving strong edges.
**5. Blackhat morphology** – highlights dark plate characters on a brighter background.
**6. Horizontal gradient (Sobel)** – emphasizes plate-like horizontal transitions.
**7. Thresholding** – converts the gradient image to a binary image.
**8. Morphological closing** – connects fragmented plate-like regions.
**9. ROI restriction** – limits the search to a plausible plate area.
**10. Contour analysis and scoring** – evaluates candidates using area, aspect ratio, fill ratio, and position.
**11. Final detection** – selects the best candidate and draws a bounding box around the detected number plate.

Intermediate results from each major step are saved as output images to clearly show how the plate region is isolated.

**Implementation details**
Language: Python
Libraries:
    OpenCV (cv2) for image processing
    NumPy for numerical operations
    os for file and output management

**Key OpenCV functions used include:**
cv2.imread, cv2.cvtColor
cv2.createCLAHE, cv2.bilateralFilter
cv2.morphologyEx, cv2.Sobel, cv2.threshold
cv2.findContours, cv2.rectangle

The script:
Reads the input image.
Applies the full preprocessing pipeline.
Restricts the region of interest.
Scores candidate contours.
Draws bounding boxes for the candidate and final detection.
Saves all intermediate outputs for analysis and visualization.

**Expected output**
For a given input vehicle image, the system:
Detects the approximate number plate region.
Draws a bounding box around the detected plate on the original image.
Produces a cropped plate image that can be used as input to an OCR module in future work.

**Applications and extensions**
This classical image processing approach is suitable for:

Traffic law enforcement
Smart parking and access control systems
Toll gate automation
Security and surveillance scenarios

As future work, the detected plate crop can be connected to an OCR engine (e.g., Tesseract or a deep learning text recognizer) to build a complete Automatic Number Plate Recognition (ANPR) system.


