import streamlit as st
import os
import sys
import subprocess

st.set_page_config(page_title="DravyaDock Diagnostics", layout="wide")
st.title("🛠️ DravyaDock Cloud Diagnostic Tool")

st.markdown("""
If you are seeing this page, the app successfully booted, but we need to check why RDKit is failing to install in the background.
""")

st.markdown("---")

# 1. Try to import RDKit safely
try:
    from rdkit import Chem
    st.success("✅ SUCCESS! RDKit is now installed and working perfectly. You can put your original app code back!")
except ModuleNotFoundError as e:
    st.error(f"❌ Failed to load RDKit: {e}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 1. Files seen by the Server")
        st.write("If you do not see `requirements.txt` below, the server is in the wrong folder, or the file is named incorrectly.")
        files = os.listdir(".")
        st.write(files)
        
        st.subheader("📄 2. Reading your requirements.txt")
        if "requirements.txt" in files:
            st.success("requirements.txt found! Here is what it contains:")
            with open("requirements.txt", "r") as f:
                st.code(f.read())
        elif "requirements.txt.txt" in files:
            st.error("⚠️ TYPO DETECTED: Your file is named `requirements.txt.txt`. You must rename it on GitHub!")
        else:
            st.error("⚠️ requirements.txt is MISSING from this folder.")

    with col2:
        st.subheader("📦 3. What the Server Actually Installed")
        st.write("This is the list of packages actually installed on the Streamlit Linux machine right now:")
        try:
            reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
            st.code(reqs.decode('utf-8'))
        except Exception as ex:
            st.write(f"Could not run pip freeze: {ex}")
