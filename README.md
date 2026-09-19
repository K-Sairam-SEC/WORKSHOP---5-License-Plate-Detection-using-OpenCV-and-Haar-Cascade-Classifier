# WORKSHOP---5-License-Plate-Detection-using-OpenCV-and-Haar-Cascade-Classifier

## Aim

To detect a license plate in an image using a Haar Cascade classifier and blur the detected license plate instead of simply drawing a rectangle around it.

## Requirements

- Anaconda
- Jupyter Notebook
- Python
- OpenCV
- Matplotlib
- NumPy
- Haar Cascade XML file

## Steps

1. Import the required libraries.
2. Read the `car_plate.jpg` image.
3. Create a function to display the image with correct coloring.
4. Load the `haarcascade_russian_plate_number.xml` classifier.
5. Detect the license plate using the Haar Cascade classifier.
6. Convert the detected `(x, y, w, h)` values into image indexing positions.
7. Extract the detected license plate as a Region of Interest (ROI).
8. Apply `cv2.medianBlur()` to blur the license plate.
9. Replace the original license plate region with the blurred ROI.
10. Display the final image with the license plate blurred.

## Algorithm

```text
Input Image
     ↓
Load Haar Cascade Classifier
     ↓
Detect License Plate
     ↓
Get (x, y, w, h)
     ↓
Extract License Plate ROI
     ↓
Apply Median Blur
     ↓
Replace Original ROI
     ↓
Display Blurred Image
```

## Program

```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image
img = cv2.imread('car_plate.jpg')


# Function to display image correctly in Matplotlib
def display(img):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Load Haar Cascade for license plate detection
plate_cascade = cv2.CascadeClassifier(
    'haarcascade_russian_plate_number.xml'
)


# Function to detect license plate
def detect_plate(img):
    plate_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in plates:
        cv2.rectangle(
            plate_img,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

    return plate_img


# Detect plate
result = detect_plate(img)

# Display detected plate
display(result)


# Function to detect and blur license plate
def detect_and_blur_plate(img):

    result = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in plates:

        # Extract license plate ROI
        roi = result[y:y+h, x:x+w]

        # Blur the license plate
        blurred = cv2.medianBlur(roi, 25)

        # Put blurred ROI back into original image
        result[y:y+h, x:x+w] = blurred

    return result


# Detect and blur plate
result = detect_and_blur_plate(img)

# Display final result
display(result)
```

## Output


<img width="596" height="321" alt="image" src="https://github.com/user-attachments/assets/5d27c958-6388-428d-8db1-0e61c95e3e8e" />

<img width="606" height="342" alt="image" src="https://github.com/user-attachments/assets/d228ffe9-9b75-40ea-8df3-2d0881912b86" />

<img width="611" height="339" alt="image" src="https://github.com/user-attachments/assets/141c20e3-043d-44c5-b9fe-c3d16fd3b114" />
