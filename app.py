import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# إعدادات الصفحة
st.set_page_config(
    page_title="رفيق | Rafiq - متابعة الحركة",
    page_icon="💙",
    layout="centered"
)

# واجهة رفيق
st.title("💙 رفيق | Rafiq")
st.caption("نموذج بحثي أولي لمتابعة اضطرابات الحركة (Freezing of Gait)")

st.info("💡 **حالة النظام:** التطبيق يحلل بيانات حركة مسجلة مسبقاً لاختبار دقة النموذج، ولا يتصل حالياً بسوار ذكي حقيقي.")

MODEL_PATH = Path(__file__).parent / "fog_model.joblib"

# تحميل النموذج بدون الأكواد المعقدة
@st.cache_resource
def load_model():
    package = joblib.load(MODEL_PATH)
    return package["model"], package["sensor_cols"]

try:
    model, sensor_cols = load_model()
    st.success("✅ تم تحميل نموذج الذكاء الاصطناعي بنجاح.")
except Exception as e:
    st.error(f"❌ فشل في تحميل النموذج. تفاصيل الخطأ: {e}")
    st.stop()

st.divider()

st.subheader("محاكاة قراءات المستشعرات 📊")
st.write("ارفعي ملف البيانات (CSV) لاختبار أداء النموذج.")

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

            st.markdown("### إعدادات حساسية التنبيه")
            threshold = st.slider(
                "عتبة التنبؤ (Threshold)",
                min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                help="قيمة أقل = تنبيهات أسرع، قيمة أعلى = تنبيهات متأخرة لكن أكيدة."
            )

            predictions = (probabilities >= threshold).astype(int)
            fog_detected = int(predictions.sum())

            col1, col2 = st.columns(2)
            col1.metric("القراءات المفحوصة", f"{len(data)} قراءة")
            col2.metric("حالات التجمد (FoG) المكتشفة", f"{fog_detected} حالة")

            results = data.copy()
            results["fog_probability"] = probabilities
            results["predicted_fog"] = predictions

            st.dataframe(results[["predicted_fog", "fog_probability"] + sensor_cols].head(50))

            st.download_button(
                label="📥 تنزيل النتائج (CSV)",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="rafiq_results.csv",
                mime="text/csv"
            )
            
            st.warning("هذا مشروع بحثي لمسابقة إبداع وليس جهازاً طبياً معتمداً.")

    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
else:
    st.info("في انتظار رفع ملف البيانات...")
