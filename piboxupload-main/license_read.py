import cv2
import pytesseract
import os
import datetime
import RPi.GPIO as GPIO
import time
import numpy as np
import imutils


# GPIO setup
TRIGGER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Directory setup
BASE_DIR = os.path.expanduser("~/Desktop/project_files")
PROCESSED = os.path.join(BASE_DIR, "processed")
FAILED = os.path.join(BASE_DIR, "failed")

os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(FAILED, exist_ok=True)

# Format timestamped filename
def generate_filename(folder):
    now = datetime.datetime.now()
    date_str = now.strftime("%d-%b-%Y_%H:%M")
    existing = [f for f in os.listdir(folder) if f.startswith(date_str)]
    count = len(existing) + 1
    return f"{date_str}_{count}.jpg"

# Digital 2x zoom function
def digital_zoom(frame, zoom_factor=2):
    height, width = frame.shape[:2]
    new_w, new_h = int(width / zoom_factor), int(height / zoom_factor)
    x1 = (width - new_w) // 2
    y1 = (height - new_h) // 2
    cropped = frame[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (width, height))

# Detect white plate-like rectangles
def find_plate_region(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 6 and cv2.contourArea(cnt) > 1000:
            return (x, y, w, h), frame[y:y+h, x:x+w]
    return None, None

# Main processing loop
def watch_for_plate():
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    print("🔍 Watching for license plate (zoomed, with live preview)...")
    found_plate = False
    plate_text = ""
    frame_to_save = None
    start_time = time.time()

    while time.time() - start_time < 60:
        ret, frame = cap.read()
        if not ret:
            continue

        img = frame.copy()
        img = cv2.resize(img, (620, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(gray, 30, 200)

        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        display_frame = img.copy()
        if screenCnt is not None:
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx+1, topy:bottomy+1]

            text = pytesseract.image_to_string(Cropped, config='--psm 11')
            print("Detected license plate Number is:", text)

            if text.strip():
                cv2.drawContours(display_frame, [screenCnt], -1, (0, 255, 0), 3)
                cv2.putText(display_frame, text.strip(), (topx, topy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                found_plate = True
                plate_text = ''.join(filter(str.isalnum, text))
                frame_to_save = display_frame.copy()
                cv2.imshow("Cropped Plate", Cropped)
                cv2.imshow("Live Feed", display_frame)
                cv2.waitKey(1000)
                break

        cv2.imshow("Live Feed", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    now = datetime.datetime.now()
    if found_plate:
        print(f"✅ License Plate Detected: {plate_text}")
        filename = generate_filename(PROCESSED)
        filepath = os.path.join(PROCESSED, filename)
        cv2.imwrite(filepath, frame_to_save)
        with open("plate.txt", "w") as f:
            f.write(plate_text)
    else:
        print("❌ License plate not recognized.")
        filename = generate_filename(FAILED)
        filepath = os.path.join(FAILED, filename)
        cv2.imwrite(filepath, frame)


# Main trigger loop
def main():
    print("🔁 Waiting for trigger (GPIO 17)... Press CTRL+C to exit.")
    try:
        while True:
            if GPIO.input(TRIGGER_PIN) == GPIO.LOW:
                watch_for_plate()
                time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()
        print("🛑 Program exited.")

if __name__ == "__main__":
    main()
