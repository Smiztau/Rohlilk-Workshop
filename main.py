import streamlit as st
import subprocess
import os

def run_ui():

    calculate = st.checkbox("Calculate Data")
    predict = st.checkbox("Predict Availability")
    test = st.checkbox("Train model and predict test")

    test_mode = st.radio("Test Mode:", ["With Calculated Availability", "Without Availability"])

    def run_command_live(label, command):
        st.write(f"▶️ **{label}**")
        full_log = ""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        with st.status(f"Running `{label}`...", expanded=True) as status_box:
            log_output = st.empty()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, env=env)

            for line in process.stdout:
                full_log += line
                log_output.code(full_log, language="bash")

            process.wait()

            if process.returncode == 0:
                status_box.update(label=f"{label} completed ✅", state="complete")
            else:
                status_box.update(label=f"{label} failed ❌ (exit code {process.returncode})", state="error")

    if st.button("Run Selected Scripts"):
        if calculate:
            run_command_live("Calculate Data", ["python", "data.py"])

        if predict:
            run_command_live("Predict Availability", ["python", "availability_by_warehouse.py"])

        if test:
            flag = "--use-availability"
            value = "true" if test_mode == "With Calculated Availability" else "false"
            run_command_live("Train model and predict test", ["python", "train_test_by_warehouse.py", flag, value])

        st.success("All selected steps completed ✅")


st.logo("logos/logo.png", size="large")
st.title("Data Processing & Predictions")

if "page" not in st.session_state:
    st.session_state.page = "About"

# === Navigation buttons ===
if st.sidebar.button("📘 About"):
    st.session_state.page = "About"
if st.sidebar.button("⚙️ Run Scripts"):
    st.session_state.page = "Scripts"

# === Page content ===
if st.session_state.page == "About":
    st.title("📘 Project Overview")
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            readme_text = f.read()
        st.markdown(readme_text)
    else:
        st.warning("README.md file not found!")

elif st.session_state.page == "Scripts":
    run_ui()