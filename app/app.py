import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import base64
from tinydb import TinyDB, Query
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
# ICON_PATH = "app/img/mdc.png"  # No longer used
MODEL_PATH = "src/models/model_Xception_ft.hdf5"
DB_PATH = "app/patients_db.json"
TARGET_SIZE = (224, 224)
CLASS_NAMES = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]

# --- SETUP ---
st.set_page_config(page_title="Severity Analysis of Arthrosis in the Knee")
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- DATABASE ---
db = TinyDB(DB_PATH)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    grad_model = tf.keras.models.clone_model(model)
    grad_model.set_weights(model.get_weights())
    grad_model.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=[grad_model.inputs],
        outputs=[
            grad_model.get_layer("global_average_pooling2d_1").input,
            grad_model.output,
        ],
    )
    return model, grad_model

model, grad_model = load_model()

# --- UTILS ---
def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

def generate_patient_id():
    all_patients = db.all()
    return f"BRO{len(all_patients)+1:03d}"

def get_gemini_analysis(name, age, gender, severity, probability):
    prompt = (
        f"You are an expert AI radiologist. Analyze the following knee X-ray diagnosis:\n"
        f"Patient Name: {name}\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n"
        f"Severity Grade: {severity}\n"
        f"Model Confidence: {probability:.2f}%\n"
        f"Please provide a brief, professional diagnostic summary and possible next steps."
    )
    # Use a supported Gemini model name!
    model = genai.GenerativeModel('models/gemini-2.5-flash-preview-05-20')
    response = model.generate_content(prompt)
    return response.text

def reset_state():
    for key in st.session_state.keys():
        del st.session_state[key]

# --- APP LOGIC ---
def main():
    st.title("Severity Analysis of Arthrosis in the Knee")

    # Navigation: Home / Patient / Doctor
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        st.header("Who are you?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("I am a Patient"):
                st.session_state.page = "patient"
                st.experimental_rerun()
        with col2:
            if st.button("I am a Doctor"):
                st.session_state.page = "doctor"
                st.experimental_rerun()

    elif st.session_state.page == "patient":
        st.header("Patient Registration & X-ray Upload")
        with st.form("patient_form"):
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=1, max_value=120)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            uploaded_file = st.file_uploader("Upload Knee X-ray", type=["jpg", "jpeg", "png"])
            submitted = st.form_submit_button("Submit")
        if submitted and uploaded_file:
            patient_id = generate_patient_id()
            image_bytes = uploaded_file.read()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            db.insert({
                "patient_id": patient_id,
                "name": name,
                "age": int(age),
                "gender": gender,
                "xray": image_b64
            })
            st.success(f"Registration successful! Your Patient ID is: {patient_id}")
            st.info("Please save this Patient ID for your doctor.")
            if st.button("Go to Home Page"):
                st.session_state.page = "home"
                st.experimental_rerun()

    elif st.session_state.page == "doctor":
        st.header("Search for Diagnosis")
        patient_id = st.text_input("Enter Patient ID")
        if st.button("Search"):
            Patient = Query()
            patient = db.get(Patient.patient_id == patient_id)
            if patient:
                st.session_state.found_patient = patient
            else:
                st.session_state.found_patient = None
                st.error("Patient ID not found. Please check and try again.")

        if st.session_state.get("found_patient"):
            patient = st.session_state.found_patient
            st.subheader("Patient Information")
            st.write(f"**Name:** {patient['name']}")
            st.write(f"**Age:** {patient['age']}")
            st.write(f"**Gender:** {patient['gender']}")
            st.write(f"**Patient ID:** {patient['patient_id']}")
            st.divider()
            st.subheader("X-ray Image")
            image_bytes = base64.b64decode(patient['xray'])
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, use_column_width=True)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img = image.resize(TARGET_SIZE)
            img = np.array(img)
            img_aux = img.copy()
            img_array = np.expand_dims(img_aux, axis=0)
            img_array = np.float32(img_array)
            img_array = tf.keras.applications.xception.preprocess_input(img_array)

            y_pred = model.predict(img_array)
            y_pred = 100 * y_pred[0]
            probability = np.amax(y_pred)
            number = np.where(y_pred == np.amax(y_pred))
            grade = str(CLASS_NAMES[np.amax(number)])

            st.subheader(":white_check_mark: Severity Grade")
            st.metric(
                label="Severity Grade:",
                value=f"{CLASS_NAMES[np.amax(number)]} - {probability:.2f}%",
            )

            st.subheader(":mag: Heatmap (Explainability)")
            heatmap = make_gradcam_heatmap(grad_model, img_array)
            gradcam_img = save_and_display_gradcam(img, heatmap)
            st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)

            st.subheader(":bar_chart: Model Confidence Analysis")
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(CLASS_NAMES, y_pred, height=0.55, align="center")
            for i, (c, p) in enumerate(zip(CLASS_NAMES, y_pred)):
                ax.text(p + 2, i - 0.2, f"{p:.2f}%")
            ax.grid(axis="x")
            ax.set_xlim([0, 120])
            ax.set_xticks(range(0, 101, 20))
            fig.tight_layout()
            st.pyplot(fig)

            st.subheader(":robot_face: Analysis from AI Radiologist")
            if st.button("Generate AI Analysis"):
                with st.spinner("AI radiologist is analyzing..."):
                    analysis = get_gemini_analysis(
                        patient['name'], patient['age'], patient['gender'], grade, probability
                    )
                    st.write(analysis)

        if st.button("Go to Home Page"):
            st.session_state.page = "home"
            st.session_state.found_patient = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()
