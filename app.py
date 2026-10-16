import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="رفيق | متابعة المشي",
    page_icon="💙",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #101820;
    color: #F4F7FA;
}
h1, h2, h3, p, label {
    color: #F4F7FA !important;
}
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 45px;
}
</style>
""", unsafe_allow_html=True)

st.title("💙 رفيق")
st.caption("نظام تجريبي لمتابعة الحركة وتنبيه المرافق")

st.divider()

st.subheader("حالة المتابعة")
st.info("هذه واجهة تجريبية — لم يتم الاتصال بساعة فعلية.")

st.metric("حالة اتصال السوار", "غير متصل")
st.caption("آخر تحديث: " + datetime.now().strftime("%H:%M"))

st.divider()

st.subheader("تجربة التنبيه")
st.write(
    "استخدمي الزر لمحاكاة تنبيه فقط. "
    "هذا لا يعني أن النظام اكتشف حالة حقيقية."
)

if st.button("🔔 تجربة تنبيه للمرافق"):
    st.warning(
        "تنبيه تجريبي: يُرجى التحقق من حالة المريض "
        "والتواصل معه عند الحاجة."
    )
else:
    st.success("لا توجد تنبيهات تجريبية نشطة.")

st.divider()

st.caption(
    "هذا نموذج أولي بحثي وليس جهازًا طبيًا. "
    "لم يُربط بعد بنموذج التنبؤ أو بمستشعر حركة حقيقي."
)
