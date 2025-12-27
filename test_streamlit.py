import streamlit as st
import json

st.title("DEBUG TEST")

with open("class_names.json") as f:
    class_dict = json.load(f)

index_to_class = {v: k for k, v in class_dict.items()}

st.write("CLASS DICTIONARY:")
st.write(index_to_class)
