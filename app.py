import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path

# ==========================================
# 1. إعدادات الصفحة (تُكتب مرة واحدة فقط)
# ==========================================
st.set_page_config(
    page_title="رفيق | Rafiq - متابعة الحركة",
    page_icon="💙",
    layout="wide"
)

# واجهة رفيق
st.title("💙 رفيق | Rafiq")
st.caption("نموذج بحثي أولي لمتابعة اضطرابات الحركة (Freezing of Gait) لمريض باركنسون")

# تحديث رسالة النظام
st.info("💡 **حالة النظام:** التطبيق يدعم الآن المراقبة الحية عبر السوار الذكي (IoT) بالإضافة إلى تحليل السجلات السابقة.")

# ==========================================
# 2. تحميل نموذج الذكاء الاصطناعي
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
# 3. واجهة التبويبات (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["🔴 المراقبة الحية (Live IoT)", "📁 تحليل السجلات (ملفات CSV)"])

# ------------------------------------------
# التبويب الأول: المراقبة الحية (ThingSpeak)
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
                response = requests.get(THINGSPEAK_URL)
                data = response.json()
                
                if 'feeds' in data and len(data['feeds']) > 0:
                    latest_feed = data['feeds'][0]
                    
                    acc_x, acc_y, acc_z = float(latest_feed['field1']), float(latest_feed['field2']), float(latest_feed['field3'])
                    gyro_x, gyro_y, gyro_z = float(latest_feed['field4']), float(latest_feed['field5']), float(latest_feed['field6'])
                    bpm = int(float(latest_feed['field7']))
                    
                    # تجهيز البيانات للنموذج
                    features = pd.DataFrame([[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]], 
                                            columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
                    
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
                    st.info("جاري انتظار البيانات من السوار...")
            time.sleep(2)

# ------------------------------------------
# التبويب الثاني: مراجعة السجلات (كودك الأصلي)
# ------------------------------------------
with tab2:
    st.subheader("تحليل قراءات الحركة المسجلة 📊")
    st.write("ارفعي ملف البيانات (CSV) لمراجعة السجل التفصيلي للمريض.")

    uploaded_file = st.file_uploader("اختاري ملف بيانات الحركة", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            missing = [c for c in sensor_cols if c not in data.columns]

            if missing:
                st.error("الملف المرفوع لا يحتوي على جميع أعمدة المستشعرات المطلوبة.")
            else:
                X = data[sensor_cols].replace([np.inf, -np.inf], np.nan)
                probabilities = model.predict_proba(X)[:, 1]

                st.markdown("### إعدادات النظام للمرافق ⚙️")
                threshold = st.slider(
                    "عتبة التنبيه (حساسية النظام)",
                    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                    help="قيمة أقل = تنبيهات أسرع وأكثر حساسية، قيمة أعلى = تنبيهات للحالات المؤكدة فقط."
                )

                predictions = (probabilities >= threshold).astype(int)
                fog_detected = int(predictions.sum())

                # عرض المؤشرات البصرية
                st.markdown("### حالة المريض الحالية 📋")
                col1, col2 = st.columns(2)
                col1.metric("إجمالي القراءات المفحوصة", f"{len(data)}")
                
                if fog_detected > 0:
                    col2.metric("تنبيهات التجمد (FoG)", f"{fog_detected} حالة", delta="تم رصد اضطراب", delta_color="inverse")
                    st.error(f"⚠️ انتباه: تم رصد {fog_detected} حالة تجمد محتملة في البيانات المرفوعة.")
                else:
                    col2.metric("تنبيهات التجمد (FoG)", "0 حالة", delta="الحركة طبيعية المستقرة", delta_color="normal")
                    st.success("✅ الحركة طبيعية، لم يتم رصد أي حالات تجمد.")

                results = data.copy()
                results["fog_probability"] = probabilities
                results["predicted_fog"] = predictions

                # إضافة الرسم البياني التفاعلي
                st.markdown("### التدرج الزمني لاحتمالية التجمد 📈")
                st.write("يوضح الرسم متى اقتربت حالة المريض من التجمد (كلما ارتفع الخط زادت الاحتمالية).")
                chart_data = pd.DataFrame(probabilities, columns=["احتمالية التجمد"])
                st.line_chart(chart_data, color="#1f77b4")

                # إخفاء التفاصيل التقنية في قائمة قابلة للطي
                with st.expander("عرض البيانات التقنية المفصلة (للباحثين) 🔍"):
                    st.dataframe(results[["predicted_fog", "fog_probability"] + sensor_cols].head(100), use_container_width=True)

                st.download_button(
                    label="📥 تنزيل تقرير الحالة الكامل (CSV)",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="rafiq_caregiver_report.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                st.caption("تنويه: هذا مشروع بحثي لمسابقة إبداع وليس جهازاً طبياً معتمداً.")

        except Exception as e:
            st.error(f"حدث خطأ أثناء المعالجة: {e}")
    else:
        st.info("في انتظار رفع ملف البيانات...")
