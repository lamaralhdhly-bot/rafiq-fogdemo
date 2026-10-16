import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path

# ==========================================
# 1. إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="رفيق | Rafiq - متابعة الحركة",
    page_icon="💙",
    layout="wide"
)

st.title("💙 رفيق | Rafiq")
st.caption("نموذج بحثي أولي لمتابعة اضطرابات الحركة (Freezing of Gait) لمريض باركنسون")

st.info("💡 **حالة النظام:** التطبيق يدعم الآن المراقبة الحية عبر السوار الذكي (IoT) بالإضافة إلى تحليل السجلات السابقة.")

# ==========================================
# 2. تحميل النموذج
# ==========================================
MODEL_PATH = Path(__file__).parent / "fog_model.joblib"

@st.cache_resource
def load_model():
    package = joblib.load(MODEL_PATH)
    if isinstance(package, dict) and "model" in package:
        return package["model"], package.get("sensor_cols", ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    else:
        return package, ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

try:
    model, sensor_cols = load_model()
except Exception as e:
    st.error(f"❌ فشل في تحميل النموذج. تفاصيل الخطأ: {e}")
    st.stop()

st.divider()

# ==========================================
# 3. واجهة التبويبات
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔴 المراقبة الحية (Live IoT)", "📁 تحليل السجلات (CSV)", "⌚ التصميم الهندسي (3D)"])

# ------------------------------------------
# التبويب الأول: المراقبة الحية
# ------------------------------------------
with tab1:
    st.subheader("الاتصال المباشر بالسوار الذكي 📡")
    
    CHANNEL_ID = "3539820"
    READ_API_KEY = "YDMYLPQD1HEO76JB"
    THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"

    run_live = st.checkbox("بدء الاتصال المباشر بالسوار 📡")
    placeholder = st.empty()

    if run_live:
        while run_live:
            try:
                headers = {'Cache-Control': 'no-cache', 'Pragma': 'no-cache'}
                response = requests.get(THINGSPEAK_URL, headers=headers, timeout=5)
                data = response.json()
                
                if 'feeds' in data and len(data['feeds']) > 0:
                    latest_feed = data['feeds'][0]
                    
                    acc_x = float(latest_feed.get('field1') or 0.0)
                    acc_y = float(latest_feed.get('field2') or 0.0)
                    acc_z = float(latest_feed.get('field3') or 0.0)
                    gyro_x = float(latest_feed.get('field4') or 0.0)
                    gyro_y = float(latest_feed.get('field5') or 0.0)
                    gyro_z = float(latest_feed.get('field6') or 0.0)
                    
                    bpm_str = latest_feed.get('field7')
                    bpm = int(float(bpm_str)) if bpm_str else 0
                    
                    temp_data = pd.DataFrame([[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]], 
                                             columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
                    
                    for col in sensor_cols:
                        if col not in temp_data.columns:
                            temp_data[col] = 0.0
                            
                    features = temp_data[sensor_cols]
                    prediction = model.predict(features)[0]
                    
                    with placeholder.container():
                        st.write(f"آخر تحديث للبيانات: {latest_feed['created_at']}")
                        col1, col2 = st.columns(2)
                        col1.metric("معدل النبض ❤️", f"{bpm} BPM")
                        
                        if prediction == 1:
                            col2.error("⚠️ تنبيه: تم رصد احتمالية تجمد (FoG)!")
                        else:
                            col2.success("الحركة طبيعية ✅")
                else:
                    with placeholder.container():
                        st.warning("لا توجد بيانات في القناة حتى الآن.")
                        
            except Exception as e:
                with placeholder.container():
                    st.error(f"توقف العرض بسبب خطأ تقني: {e}")
            
            time.sleep(2)

# ------------------------------------------
# التبويب الثاني: تحليل السجلات
# ------------------------------------------
with tab2:
    st.subheader("تحليل قراءات الحركة المسجلة 📊")
    st.write("ارفعي ملف البيانات (CSV) لمراجعة السجل التفصيلي للمريض.")

    uploaded_file = st.file_uploader("اختاري ملف بيانات الحركة", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            X = data[sensor_cols].replace([np.inf, -np.inf], np.nan)
            probabilities = model.predict_proba(X)[:, 1]

            threshold = st.slider("عتبة التنبيه (حساسية النظام)", 0.1, 0.9, 0.5, 0.05)
            predictions = (probabilities >= threshold).astype(int)
            fog_detected = int(predictions.sum())

            col1, col2 = st.columns(2)
            col1.metric("إجمالي القراءات المفحوصة", f"{len(data)}")
            
            if fog_detected > 0:
                col2.metric("تنبيهات التجمد (FoG)", f"{fog_detected} حالة", delta="تم رصد اضطراب", delta_color="inverse")
                st.error(f"⚠️ انتباه: تم رصد {fog_detected} حالة تجمد محتملة في البيانات المرفوعة.")
            else:
                col2.metric("تنبيهات التجمد (FoG)", "0 حالة", delta="الحركة طبيعية المستقرة", delta_color="normal")
                st.success("✅ الحركة طبيعية، لم يتم رصد أي حالات تجمد.")

            chart_data = pd.DataFrame(probabilities, columns=["احتمالية التجمد"])
            st.line_chart(chart_data, color="#1f77b4")

        except Exception as e:
            st.error(f"حدث خطأ أثناء المعالجة: {e}")

# ------------------------------------------
# التبويب الثالث: التصميم الهندسي 3D
# ------------------------------------------
with tab3:
    st.subheader("التصميم الهندسي الداخلي للسوار (3D Model) ⚙️")
    st.write("هذا النموذج ثلاثي الأبعاد يوضح طريقة ترتيب القطع الإلكترونية (لوحة ESP32 والحساسات) داخل هيكل السوار. **يمكنك تحريك المجسم بلمس الشاشة أو الماوس لرؤيته من جميع الزوايا!**")
    
    # دمج تصميم Tinkercad مباشرة باستخدام iframe
    tinkercad_url = "https://www.tinkercad.com/embed/iQfew60fuLV"
    components.iframe(tinkercad_url, width=800, height=500)
    
    st.divider()
    st.caption("ملاحظة: هذا المجسم يمثل الهيكل الداخلي الأولي (Functional Case) لاحتواء القطع التقنية، وهو جزء من متطلبات النموذج الأولي لمشروع رفيق.")
