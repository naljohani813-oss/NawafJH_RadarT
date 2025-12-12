# run_pipeline.py
# ========================================
# تشغيل YOLO + OCR على فيديو وعرضه لايف
# مع تسجيل النتائج في CSV
# ========================================

from ultralytics import YOLO
import cv2
import easyocr
import csv
import os
import numpy as np

# =========================
# 1) عدّل هذه المسارات
# =========================
WEIGHTS_PATH = r"C:\Users\nawaf\OneDrive\Desktop\Nawaf12\best.pt"
VIDEO_PATH   = r"C:\Users\nawaf\OneDrive\Desktop\Nawaf12\mycarplatee.mp4"
CSV_PATH     = r"C:\Users\nawaf\OneDrive\Desktop\Nawaf12\detectionss_log.csv"

# لو تبي كاميرا بدال الفيديو:
# cap = cv2.VideoCapture(0)

# =====================================
# 2) تحميل موديل YOLO و OCR Reader
# =====================================
print("[INFO] تحميل نموذج YOLO...")
model = YOLO(WEIGHTS_PATH)

print("[INFO] تجهيز EasyOCR (ar + en)...")
reader = easyocr.Reader(['ar', 'en'], gpu=False)  # خلي gpu=True لو عندك كرت شاشة قوي

# كلاس اللوحة من data.yaml:
# names: ['arabic-number','arabic-text','car','english-number','english-text','license-plate','truck']
PLATE_CLASS_ID = 5   # حسب ترتيب الأسماء في data.yaml
CAR_CLASS_ID   = 2   # car

# =====================================
# 3) دوال مساعدة
# =====================================

def preprocess_plate(plate_bgr):
    """تحسين صورة اللوحة قبل إرسالها لـ OCR"""
    if plate_bgr is None or plate_bgr.size == 0:
        return None

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    # تكبير اللوحة عشان الحروف تكون أوضح
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # تقليل النويز
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # ثريشولد أوتسو
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def clean_text(text):
    """تنظيف النص من المسافات والرموز الغريبة"""
    if not text:
        return ""

    # حذف مسافات زائدة
    text = text.replace(" ", "")
    # حذف بعض الرموز اللي تطلع بالغلط
    bad_chars = ['"', "'", ',', ';', ':', '|', '،', 'ـ']
    for ch in bad_chars:
        text = text.replace(ch, "")

    return text

def detect_color(car_bgr):
    """تخمين لون السيارة بشكل تقريبي جداً من متوسط اللون"""
    if car_bgr is None or car_bgr.size == 0:
        return "غير معروف"

    # متوسط BGR
    b, g, r = np.mean(car_bgr.reshape(-1, 3), axis=0)

    # تحويل تقريبي إلى أسماء ألوان
    if r < 80 and g < 80 and b < 80:
        return "أسود"
    if r > 180 and g > 180 and b > 180:
        return "أبيض"
    if r > g and r > b:
        return "أحمر/برتقالي"
    if g > r and g > b:
        return "أخضر"
    if b > r and b > g:
        return "أزرق"
    return "رمادي"

def approx_model_name():
    """مكان مخصص لموديل السيارة (حالياً ما عندنا موديل مدرب)"""
    # هنا لاحقاً تقدر تربط موديل تصنيف car model أو تستخدم Roboflow آخر
    return "موديل غير معروف"

# =====================================
# 4) فتح الفيديو وتحضير CSV
# =====================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] ما قدر يفتح الفيديو: {VIDEO_PATH}")
    exit()

# فتح ملف CSV
csv_file = open(CSV_PATH, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame",
    "confidence",
    "plate_text",
    "car_color",
    "car_model",
    "x1", "y1", "x2", "y2"
])

print("[INFO] بدأ المعالجة... اضغط q للإغلاق")

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] انتهى الفيديو.")
        break

    frame_idx += 1

    # تشغيل YOLO على الفريم
    results = model(frame, imgsz=800, conf=0.5, verbose=False)[0]

    # نخزن أفضل لوحة في الفريم (أعلى ثقة)
    best_plate = None
    best_conf = 0.0
    best_box  = None
    best_car_box = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        if cls_id == PLATE_CLASS_ID:
            if conf > best_conf:
                best_conf = conf
                best_box  = (x1, y1, x2, y2)

        # نحتفظ بأول سيارة (قرب اللوحة عادة)
        if cls_id == CAR_CLASS_ID and best_car_box is None:
            best_car_box = (x1, y1, x2, y2)

    plate_text = ""
    car_color  = "غير معروف"
    car_model  = approx_model_name()

    # لو لقينا لوحة
    if best_box is not None:
        x1, y1, x2, y2 = best_box

        # تأكد من الحدود
        h, w, _ = frame.shape
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        plate_crop = frame[y1:y2, x1:x2]
        plate_proc = preprocess_plate(plate_crop)

        if plate_proc is not None:
            # نحدد الأحرف المسموح بها (عربي + إنجليزي + أرقام)
            allowlist = "ابتثجحخدذرزسشصضطظعغفقكلمنهويآأإ0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ocr_result = reader.readtext(
                plate_proc,
                detail=0,          # نبغى النص بس
                paragraph=True,
                allowlist=allowlist
            )

            if ocr_result:
                plate_text = clean_text("".join(ocr_result))

        # نرسم مستطيل حول اللوحة
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # لو لقينا سيارة: نستخدمها لتخمين اللون
    if best_car_box is not None:
        cx1, cy1, cx2, cy2 = best_car_box
        h, w, _ = frame.shape
        cx1 = max(0, cx1); cy1 = max(0, cy1)
        cx2 = min(w, cx2); cy2 = min(h, cy2)
        car_crop = frame[cy1:cy2, cx1:cx2]
        car_color = detect_color(car_crop)

    # نكتب النص على الفريم (لو عندنا لوحة)
    if best_box is not None:
        label = f"لوحة: {plate_text or 'غير مقروءة'} | لون: {car_color} | conf: {best_conf:.2f}"
        # نختار مكان النص فوق اللوحة
        tx, ty = best_box[0], max(20, best_box[1] - 10)
        cv2.putText(frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # نكتب في CSV
        csv_writer.writerow([
            frame_idx,
            round(best_conf, 3),
            plate_text,
            car_color,
            car_model,
            best_box[0], best_box[1], best_box[2], best_box[3]
        ])

    # عرض الفريم لايف
    cv2.imshow("Smart City Radar - Live", frame)

    # اضغط q للخروج
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تنظيف
cap.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"[INFO] تم حفظ اللوق في: {CSV_PATH}")
