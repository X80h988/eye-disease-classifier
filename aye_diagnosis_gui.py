import tkinter as tk# لبناء الواجهة الرسومية
from tkinter import filedialog, messagebox# لاختيار الملفات وإظهار الرسائل
import cv2  # مكتبة OpenCV لمعالجة الصور
import numpy as np  # مكتبة للحسابات العددية والمصفوفات
from sklearn.ensemble import RandomForestClassifier# نموذج الغابة العشوائية
import joblib # لحفظ وتحميل النماذج
from PIL import Image, ImageTk# للتعامل مع الصور وعرضها في Tkinter
import os  # للتعامل مع الملفات والمسارات
import warnings # لإخفاء التحذيرات الغير ضرورية

warnings.filterwarnings("ignore", category=UserWarning)
def train_robust_model():
    X, y = [], []
    for _ in range(1200):
        label = np.random.randint(0, 2)
        if label == 1: # لو مصاب (العين فيها غشاوة)
            white_pixel_ratio = np.random.normal(0.6, 0.1) # نسبة البياض عالية
            avg_brightness = np.random.normal(180, 20) # السطوع مرتفع
            std_dev = np.random.normal(40, 10)# التباين متوسط
        else:
            white_pixel_ratio = np.random.normal(0.1, 0.05)
            avg_brightness = np.random.normal(40, 10)
            std_dev = np.random.normal(15, 5)
        X.append([white_pixel_ratio, avg_brightness, std_dev])
        y.append(label)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y) # ندرّب النموذج على البيانات X و y
    joblib.dump(model, "eye_model_final.pkl") # نحفظ النموذج في ملف حتى نستخدمه لاحقًا
    return model
# -------------------------------------------------
# 🖥️ كلاس الواجهة الرسومية الرئيسية (Tkinter)
# -------------------------------------------------
class EyeApp:
    def __init__(self, root):
        self.root = root # نحفظ نافذة البرنامج داخل الكائن
        self.root.title("نظام فحص إعتام عدسة العين الذكي")
        self.root.geometry("720x850")
        self.root.configure(bg="#f5f6fa")

        try:
            if not os.path.exists("eye_model_final.pkl"):
                self.model = train_robust_model()
            else:
                self.model = joblib.load("eye_model_final.pkl")
        except:
            self.model = train_robust_model()
        # ------------------------------
        # 🏷️ رأس التطبيق (العنوان)
        # ------------------------------
        header = tk.Frame(root, bg="#2f3640", height=60)
        header.pack(fill="x")#يوسعه عرضيا على الكامل النافذه
        tk.Label(header, text="نظام تشخيص إعتام عدسة العين", font=("Arial", 20, "bold"),
                 bg="#2f3640", fg="white").pack(pady=10)
        # ------------------------------
        # 🖼️ منطقة عرض الصورة
        # ------------------------------
        self.display_frame = tk.Frame(root, width=500, height=400, bg="#dcdde1", relief="sunken", bd=2)
        self.display_frame.pack(pady=20)
        self.display_frame.pack_propagate(False)# تثبيت الحجم
        self.display = tk.Label(self.display_frame, text="يرجى تحميل صورة عين واضحة", bg="#dcdde1", font=("Arial", 13))
        self.display.pack(fill="both", expand=True)

        btn_frame = tk.Frame(root, bg="#f5f6fa")
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="📁 تحميل صورة", command=self.load_img, font=("Arial", 12, "bold"),
                  bg="#0097e6", fg="white", padx=25, pady=10).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="🔍 تحليل الصورة", command=self.analyze, font=("Arial", 12, "bold"),
                  bg="#44bd32", fg="white", padx=25, pady=10).grid(row=0, column=1, padx=10)

        self.res_box = tk.Frame(root, bg="white", bd=2, relief="groove")
        self.res_box.pack(pady=20, padx=50, fill="x")
        self.res_lbl = tk.Label(self.res_box, text="النتيجة: بانتظار التحليل", font=("Arial", 18, "bold"), bg="white", fg="#353b48")
        self.res_lbl.pack(pady=8)
        self.conf_lbl = tk.Label(self.res_box, text="", font=("Arial", 14), bg="white", fg="#7f8c8d")
        self.conf_lbl.pack(pady=5)

        info_box = tk.Frame(root, bg="#ecf0f1", bd=1, relief="ridge")
        info_box.pack(pady=10, padx=40, fill="x")
        info_text = (
            "🔹 ما هو إعتام عدسة العين (Cataract):\n"
            "هو حالة طبية تُسبب غشاوة في عدسة العين، مما يؤدي إلى ضعف تدريجي في الرؤية.\n"
            "يمكن اكتشافها مبكرًا عبر الفحص الطبي. هذا النظام يقدم فحصًا مبدئيًا فقط "
            "ولا يغني عن استشارة الطبيب المختص."
        )
        tk.Label(info_box, text=info_text, font=("Arial", 12), bg="#ecf0f1", fg="#2f3640", justify="right", wraplength=600).pack(pady=10)

        self.img_path = None

    def load_img(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.img_path = path
            try:
                img = Image.open(path)
                try:
                    # نستخدم فلتر مناسب لتصغير الصورة بجودة عالية
                    resample_filter = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_filter = Image.LANCZOS       # نستخدم فلتر مناسب لتصغير الصورة بجودة عالية حسب اصدار pill
                img = img.resize((500, 400), resample_filter)
                tk_img = ImageTk.PhotoImage(img)#عشان واجهه tikتقدر تتعامل معاه نحوله ل object
                self.display.config(image=tk_img, text="")
                self.display.image = tk_img# نحتفظ بنسخة لتبقى ظاهرة
                self.res_lbl.config(text="جاهز للتحليل", fg="#353b48")   # نحدث النص ليظهر أن الصورة جاهزة للتحليل
                self.conf_lbl.config(text="")
            except Exception as e:#في حال خطا تضهر رساله مع سبب الخطا
                messagebox.showerror("خطأ", f"فشل تحميل الصورة: {e}")

    # -------------------------------------------------
    # 🔍 دالة تحليل الصورة
    # -------------------------------------------------
    def analyze(self):
        if not self.img_path:
            messagebox.showwarning("تنبيه", "يرجى تحميل صورة أولاً")
            return

        try:
            with open(self.img_path, "rb") as f:#binتقرا وتحولها ل
                nparr = np.frombuffer(f.read(), np.uint8)#نحول لمصفوفه بابتات
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)#نحول لمصفوفه صيغه bgr

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#نحول  لصوره رماديه عشان يكون سهل في الحساب
            h, w = gray.shape
            center_roi = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
            # نصغر الصور للوسط عشان يكون التحليل سريع

            # نستخدم فلتر تنعيم لتقليل التشويش
            blurred = cv2.medianBlur(gray, 5)
            # نحاول نكتشف دائرة العين (القزحية)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                                       param1=100, param2=30, minRadius=30, maxRadius=int(min(h, w) / 3))

            # نحول الصورة الثنائية (أسود/أبيض)
            _, thresh = cv2.threshold(center_roi, 150, 255, cv2.THRESH_BINARY)
            # نحسب نسبة البياض
            white_ratio = np.sum(thresh == 255) / thresh.size
            # نحسب السطوع المتوسط والتباين
            avg_brightness = np.mean(center_roi)
            std_dev = np.std(center_roi)
            # نجمع الميزات في مصفوفة لتحليلها
            features = np.array([[white_ratio, avg_brightness, std_dev]])
            # نحصل على التنبؤ والاحتمالية
            prediction = self.model.predict(features)[0]
            #يحول الئ نسبه مئويع
            confidence = self.model.predict_proba(features)[0][prediction] * 100
            # إذا ما اكتشف دوائر → الصورة مو عين
            if circles is None or len(circles[0]) == 0:
                messagebox.showwarning("⚠️ غير عين", "لم يتم اكتشاف شكل يشبه العين في الصورة.")
                self.res_lbl.config(text=f"النتيجة: غير صالحة (نسبة الاشتباه {confidence:.1f}%)", fg="#e1b12c")
                return
            # لو الصورة مضيئة أكثر من اللازم → غير مناسبة للتحليل
            mean_color = np.mean(img, axis=(0, 1))
            if mean_color[2] > 180 and mean_color[1] > 180 and mean_color[0] > 180:
                messagebox.showwarning("صورة غير مناسبة", "الصورة شديدة الإضاءة أو لا تحتوي على تفاصيل عين واضحة.")
                self.res_lbl.config(text=f"النتيجة: غير صالحة (نسبة الاشتباه {confidence:.1f}%)", fg="#e1b12c")
                return
            #لولو التنبؤ = 1 ونسبة البياض كبيرة → اشتباه بإصابة

            if prediction == 1 and white_ratio > 0.15:
                self.res_lbl.config(text="النتيجة: اشتباه بإصابة (Cataract)", fg="#e84118")
            else:
                self.res_lbl.config(text="النتيجة: العين سليمة", fg="#44bd32")

            self.conf_lbl.config(text=f"نسبة التأكد: {confidence:.1f}%")

        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء التحليل:\n{e}")

 # -------------------------------------------------
        # 🚀 تشغيل البرنامج
        # -------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeApp(root)
    #يرسم الواجهه داخليا
    root.mainloop()
    #loop عشان تبقى الواجهه شغاله