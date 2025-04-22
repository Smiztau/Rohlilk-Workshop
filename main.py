import streamlit as st
import subprocess
import os

def run_ui():
    calculate = st.checkbox("Calculate Data")

    # Only show if Calculate is checked
    use_tsfresh = False
    if calculate:
        st.markdown("## Calculate Data Configuration")
        use_tsfresh = st.checkbox("With TSFresh")

    predict = st.checkbox("Predict Availability")
    test = st.checkbox("Train model and predict test")

    # Only show options if "test" is checked
    if test:
        st.markdown("## Train & Test Configuration")
        test_mode = st.checkbox("With Calculated Availability")
        test_by = st.radio("Split Data By", ["Without Split", "Test by Warehouse", "Test by Unique_id"])
    else:
        test_mode = None
        test_by = None

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

            if use_tsfresh:
                run_command_live("Generate TSFresh Features", ["python", "feature_extraction.py"])

        if predict:
            run_command_live("Predict Availability", ["python", "calculate_availability.py"])

        if test:
            flag = "--use-availability"
            value = "true" if test_mode else "false"

            if test_by == "Test by Warehouse":
                script_path = "train_and_test/train_test_by_warehouse.py"
            elif test_by == "Test by Unique_id":
                script_path = "train_and_test/train_test_by_id.py"
            else:
                script_path = "train_and_test/train_test.py"

            run_command_live("Train model and predict test", ["python", script_path, flag, value])

        st.success("All selected steps completed ✅")

# === Logo and Title ===
st.logo("logos/logo.png", size="large")
st.title("Data Processing & Predictions")

# === Navigation buttons ===
if "page" not in st.session_state:
    st.session_state.page = "About"

if st.sidebar.button("📘 About"):
    st.session_state.page = "About"
if st.sidebar.button("⚙️ Run Scripts"):
    st.session_state.page = "Scripts"

# === Page content ===
if st.session_state.page == "About":
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            readme_text = f.read()
        st.markdown(readme_text)
    else:
        st.warning("README.md file not found!")

elif st.session_state.page == "Scripts":
    run_ui()
