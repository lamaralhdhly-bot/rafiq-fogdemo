import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="رفيق | متابعة الحركة",
    page_icon="💙",
    layout="centered"
)

st.title("💙 رفيق")
st.caption("نموذج بحثي تجريبي لمتابعة الحركة")

MODEL_PATH = Path(__file__).parent / "fog_model.joblib"

@st.cache_resource
def load_model():
    package = joblib.load(MODEL_PATH)
    return package["model"], package["sensor_cols"]

try:
    model, sensor_cols = load_model()
    st.success("تم تحميل النموذج بنجاح")
except Exception as e:
    st.error("تعذر تحميل النموذج. تحققي من ملف النموذج والمكتبات.")
    st.stop()

st.divider()
st.subheader("اختبار النموذج")

st.write(
    "ارفعي ملف CSV يحتوي على أعمدة مستشعرات الحركة "
    "المستخدمة في تدريب النموذج."
)

uploaded_file = st.file_uploader(
    "اختاري ملف بيانات الحركة CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        missing = [c for c in sensor_cols if c not in data.columns]

        if missing:
            st.error(
                "الملف لا يحتوي على جميع أعمدة المستشعرات المطلوبة."
            )
            st.write("الأعمدة الناقصة:", missing)
        else:
            X = data[sensor_cols].replace(
                [np.inf, -np.inf], np.nan
            )

            probabilities = model.predict_proba(X)[:, 1]
            threshold = st.slider(
                "عتبة التنبؤ التجريبية",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05
            )

            predictions = (probabilities >= threshold).astype(int)

            st.metric("عدد الصفوف المفحوصة", len(data))
            st.metric(
                "الصفوف التي صنّفها النموذج إيجابية",
                int(predictions.sum())
            )

            results = data.copy()
            results["fog_probability"] = probabilities
            results["predicted_fog"] = predictions

            st.dataframe(results.head(100))

            st.download_button(
                "تنزيل نتائج الاختبار",
                results.to_csv(index=False).encode("utf-8"),
                file_name="rafiq_test_results.csv",
                mime="text/csv"
            )

            st.warning(
                "هذه نتائج بحثية تجريبية على بيانات مسجّلة، "
                "وليست تنبيهات طبية أو اكتشافًا مباشرًا من ساعة."
            )

    except Exception:
        st.error(
            "تعذر تحليل الملف. تحققي من صيغة CSV وأعمدة البيانات."
        )

else:
    st.info("ارفعي ملف بيانات الحركة لبدء الاختبار.")
