# ---------------------------------------------
# رادار طويق - قراءة اللوحات + لون السيارة من فيديو
# ---------------------------------------------

from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import csv
import os

# ============================
# 1) عدّل هذه المتغيرات حسب جهازك
# ============================

WEIGHTS_PATH = r"C:\Users\nawaf\Downloads\NawafJHHack\best.pt"  # مسار best.pt
VIDEO_PATH   = r"C:\Users\nawaf\Downloads\NawafJHHack\mycarplatee.MP4"  # فيديو الاختبار (أو 0 للكاميرا)
OUTPUT_VIDEO = r"C:\Users\nawaf\Downloads\NawafJHHack\output_with_plate_info.mp4"  # الفيديو الناتج
CSV_PATH = r"C:\Users\nawaf\Downloads\NawafJHHack\detections2_log.csv"
     # ملف CSV لحفظ النتائج

CONF_PLATE = 0.5   # أقل ثقة لقبول اللوحة
CONF_CAR   = 0.4   # أقل ثقة لقبول السيارة

# أسماء الكلاسات من data.yaml تبعك
CLASS_NAMES = [
    "arabic-number",   # 0
    "arabic-text",     # 1
    "car",             # 2
    "english-number",  # 3
    "english-text",    # 4
    "license-plate",   # 5
    "truck"            # 6
]

PLATE_CLASS_ID = CLASS_NAMES.index("license-plate")
CAR_CLASS_ID   = CLASS_NAMES.index("car")

# ============================
# 2) دوال مساعدة
# ============================

# دالة تقريبية لتخمين لون السيارة من الـ BGR
def estimate_car_color(bgr_img):
    if bgr_img.size == 0:
        return "غير معروف"

    # نصغر الصورة عشان الحساب يكون أسرع
    small = cv2.resize(bgr_img, (50, 50))
    b, g, r = np.mean(small.reshape(-1, 3), axis=0)

    brightness = (r + g + b) / 3.0

    if brightness > 200:
        return "أبيض"
    if brightness < 50:
        return "أسود"

    # نحدد أي قناة غالبية
    if r > g and r > b:
        if r - max(g, b) < 30:
            return "رمادي / فضي"
        return "أحمر / عنابي"
    if g > r and g > b:
        return "أخضر"
    if b > r and b > g:
        return "أزرق / كحلي"

    return "رمادي / فضي"


# دالة قراءة نص اللوحة من صورة باستخدام EasyOCR
def read_plate_text(plate_img, reader):
    if plate_img.size == 0:
        return ""

    # EasyOCR يقبل صور BGR مباشرة كـ numpy array
    results = reader.readtext(plate_img, detail=0)  # detail=0 يرجع نص فقط
    if not results:
        return ""
    # نجمع النصوص المتقطعة في سطر واحد
    text = " ".join(results)
    # نشيل المسافات الزائدة
    text = text.replace("\n", " ").strip()
    return text


# دالة لإيجاد أقرب سيارة إلى اللوحة (بالاعتماد على مراكز الصناديق)
def find_nearest_car(plate_box, car_boxes):
    if not car_boxes:
        return None

    px1, py1, px2, py2 = plate_box
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2

    min_dist = None
    best_car = None

    for car_box in car_boxes:
        cx1, cy1, cx2, cy2 = car_box
        ccx = (cx1 + cx2) / 2
        ccy = (cy1 + cy2) / 2
        dist = (pcx - ccx) ** 2 + (pcy - ccy) ** 2  # ما نحتاج الجذر

        if (min_dist is None) or (dist < min_dist):
            min_dist = dist
            best_car = car_box

    return best_car


# ============================
# 3) تحميل النموذج و OCR
# ============================

print("[INFO] تحميل نموذج YOLO...")
model = YOLO(WEIGHTS_PATH)

print("[INFO] تحميل EasyOCR (عربي + إنجليزي)... قد يأخذ نصف دقيقة أول مرة.")
reader = easyocr.Reader(['ar', 'en'], gpu=False)

# ============================
# 4) فتح الفيديو / الكاميرا
# ============================

if VIDEO_PATH == "0":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("لم أستطع فتح مصدر الفيديو، تأكد من المسار أو الكاميرا.")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# تجهيز كاتب الفيديو الناتج
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps if fps > 0 else 25, (width, height))

# تجهيز ملف CSV
csv_file = open(CSV_PATH, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "time_sec", "plate_text", "car_color", "confidence", "x1", "y1", "x2", "y2"])

# ============================
# 5) حلقة المعالجة الرئيسية
# ============================

frame_idx = 0
print("[INFO] بدء المعالجة... اضغط Q في نافذة الفيديو للإيقاف.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # تشغيل YOLO على الفريم
    results = model(frame, verbose=False)[0]

    boxes_xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
    confs      = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
    classes    = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []

    plate_boxes = []
    plate_confs = []
    car_boxes   = []

    # نفصل اللوحات عن السيارات
    for box, conf, cls_id in zip(boxes_xyxy, confs, classes):
        x1, y1, x2, y2 = box
        if cls_id == PLATE_CLASS_ID and conf >= CONF_PLATE:
            plate_boxes.append((x1, y1, x2, y2))
            plate_confs.append(float(conf))
        elif cls_id == CAR_CLASS_ID and conf >= CONF_CAR:
            car_boxes.append((x1, y1, x2, y2))

    # نعالج كل لوحة
    for (box, plate_conf) in zip(plate_boxes, plate_confs):
        x1, y1, x2, y2 = box
        ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])

        # قص اللوحة
        plate_crop = frame[iy1:iy2, ix1:ix2].copy()
        plate_text = read_plate_text(plate_crop, reader)

        # نربط اللوحة بأقرب سيارة
        nearest_car = find_nearest_car(box, car_boxes)
        car_color = "غير معروف"
        if nearest_car is not None:
            cx1, cy1, cx2, cy2 = nearest_car
            icx1, icy1, icx2, icy2 = map(int, [cx1, cy1, cx2, cy2])
            car_crop = frame[icy1:icy2, icx1:icx2].copy()
            car_color = estimate_car_color(car_crop)

            # نرسم مستطيل السيارة
            cv2.rectangle(frame, (icx1, icy1), (icx2, icy2), (255, 255, 0), 2)

        # نرسم مستطيل حول اللوحة
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

        # نص نعرضه فوق اللوحة
        label = f"{plate_text} | {car_color} | {plate_conf:.2f}"
        cv2.putText(
            frame,
            label,
            (ix1, max(0, iy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # نحفظ في CSV
        csv_writer.writerow([
            frame_idx,
            f"{time_sec:.2f}",
            plate_text,
            car_color,
            f"{plate_conf:.3f}",
            ix1, iy1, ix2, iy2
        ])

    # نكتب الفريم في الفيديو الناتج
    out_writer.write(frame)

    # نعرض الفريم (اختياري)
    cv2.imshow("Tuwaiq Radar - Plate & Color", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

# ============================
# 6) تنظيف الموارد
# ============================

cap.release()
out_writer.release()
csv_file.close()
cv2.destroyAllWindows()

print("[INFO] انتهت المعالجة ✅")
print(f"[INFO] تم حفظ الفيديو في: {OUTPUT_VIDEO}")
print(f"[INFO] تم حفظ ملف CSV في: {CSV_PATH}")
