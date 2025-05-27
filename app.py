import streamlit as st 
import pickle 
import os
from streamlit_option_menu import option_menu
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
import io

st.set_page_config(page_title="Mulitple Disease Prediction",layout="wide", page_icon="👨‍🦰🤶")


working_dir = os.path.dirname(os.path.abspath(__file__))
scaler1 = pickle.load(open(f'{working_dir}/saved_models/scaler1.pkl','rb'))
lr = pickle.load(open(f'{working_dir}/saved_models/lr.pkl','rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart.pkl','rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
anemia_model=pickle.load(open(f'{working_dir}/saved_models/anemia.pkl','rb'))
model=load_model(f'{working_dir}/saved_models/effnet.keras')
#skin_model = load_model(f'{working_dir}/saved_models/skin.keras')


classes = {
    4: ('nv', 'melanocytic nevi'),
    6: ('mel', 'melanoma'),
    2: ('bkl', 'benign keratosis-like lesions'),
    1: ('bcc', 'basal cell carcinoma'),
    5: ('vasc', 'pyogenic granulomas and hemorrhage'),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'dermatofibroma')
}


def skin_pred(uploaded_image):
    image = Image.open(uploaded_image)

    image = image.resize((28, 28))  # Resize to match the input size of the model
    image = np.array(image)  # Convert image to numpy array
    
    if image.shape[-1] != 3:  # Ensure the image is RGB
        image = np.stack([image] * 3, axis=-1)  # Stack channels if the image is grayscale
    
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.reshape(1, 28, 28, 3)  # Reshape for model prediction

    probabilities = skin_model.predict(image)[0]

    predicted_class = np.argmax(probabilities)
    confidence_score = probabilities[predicted_class]
    predicted_class_name = classes[predicted_class][1]

    return f"Prediction: {predicted_class_name} with confidence: {confidence_score:.2f}" 

def img_pred(uploaded_image):
    img = Image.open(uploaded_image)
    
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Resize the image to match the input shape of the model
    img = cv2.resize(opencvImage, (150, 150)) 
    img = img.reshape(1, 150, 150, 3) 

    # Predict using the model
    prediction = model.predict(img)
    p_class = np.argmax(prediction, axis=1)[0]

    # Map predictions to class labels
    if p_class == 0:
        p = 'Glioma Tumor'
    elif p_class == 1:
        p = 'No Tumor'
    elif p_class == 2:
        p = 'Meningioma Tumor'
    else:
        p = 'Pituitary Tumor'
    
    return p


@st.cache_resource
def load_lung_cancer_model():
    return tf.keras.models.load_model(f'{working_dir}/saved_models/trained_lung_cancer_model.h5')

lung_cancer_model = load_lung_cancer_model()


LUNG_CANCER_CLASSES = ['adenocarcinoma', 'large cell carcinoma', 'normal', 'squamous cell carcinoma']
IMAGE_SIZE = (350, 350)


with st.sidebar:
    selected = option_menu("Mulitple Disease Prediction", 
                ['Diabetes Prediction',
                 'Heart Disease Prediction',
                 'Parkinsons Prediction',
                 'Skin Cancer Prediction',
                 'Lungs Cancer Predictin',
                 'Brain Tumor Prediction',
                 'Anemia Prediction'],
                 menu_icon='hospital-fill',
                 icons=['activity','heart','person','gender-female','lungs','person'],
                 default_index=0)
    
    if selected == "Breast Cancer Prediction":
        st.query_params.page = "breast_cancer_predictor"

if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction")

    # Create columns for inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120, step=1)
    with col2:
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80, step=1)
        Insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=100, step=1)
    with col3:
        BMIs = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
    
    # Another row for the Diabetes Pedigree Function
    col4 = st.columns(1)[0]
    with col4:
        DiabetesPedigreeFunction = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01
        )

    # Predict button
    if st.button("Predict"):
        # Prepare the input array
        temp_arr = [Glucose, BloodPressure, SkinThickness, Insulin, BMIs, DiabetesPedigreeFunction, Age]
        data = np.array([temp_arr])
        
        # Transform data using scaler
        temp_sc = scaler1.transform(data)
        
        # Make prediction
        pred = lr.predict(temp_sc)[0]
        
        # Map prediction to result
        if pred == 0:
            res = "does not indicate"
        else:
            res = "indicates"
        
        # Display results
        st.write("Input Features:", temp_arr)
        st.write("Transformed Features:", temp_sc)
        st.success(f"The prediction {res} diabetes.")

if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction Using Machine Learning")

    # Set up input columns
    col1, col2, col3  = st.columns(3)

    # Inputs with number and dropdowns for better guidance
    with col1:
        age = st.number_input("Age", min_value=0, step=1, help="Enter your age in years")
    with col2:
        sex = st.selectbox("Sex", options=["Male", "Female"], index=0, help="Select 0 for Male, 1 for Female")
    with col3:
        cp = st.selectbox("Chest Pain Type", options=[
            "0: Typical angina", 
            "1: Atypical angina", 
            "2: Non-anginal pain", 
            "3: Asymptomatic"], index=0)

    with col1:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, help="Resting blood pressure in mm Hg")
    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, help="Serum cholesterol level in mg/dL")
    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=["True", "False"], index=1, help="Select True if fasting blood sugar is > 120 mg/dL")

    with col1:
        restecg = st.selectbox("Resting Electrocardiographic Results", options=[
            "0: Normal", 
            "1: Having ST-T wave abnormality", 
            "2: Left ventricular hypertrophy"], index=0)

    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, help="Enter the maximum heart rate achieved")
    with col3:
        exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"], index=1)

    with col1:
        oldpeak = st.number_input("ST Depression Induced by Exercise", format="%.1f", help="Enter the value for ST depression induced by exercise")
    with col2:
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
            "0: Upsloping", 
            "1: Flat", 
            "2: Downsloping"], index=0)

    with col3:
        ca = st.slider("Major Vessels Colored by Flourosopy", min_value=0, max_value=4, step=1, help="Select the number of major vessels (0-4)")
    
    with col1:
        thal = st.selectbox("Thalassemia", options=[
            "0: Normal", 
            "1: Fixed defect", 
            "2: Reversible defect"], index=0)

    heart_disease_result = ""

    # Trigger prediction on button click
    if st.button("Heart Disease Test Result"):
        sex = 0 if sex == "Male" else 1
        fbs = 1 if fbs == "True" else 0
        exang = 1 if exang == "Yes" else 0

        user_input = [
            age, sex, int(cp[0]), trestbps, chol, fbs, int(restecg[0]), thalach, 
            exang, oldpeak, int(slope[0]), ca, int(thal[0])
        ]
        
        # Predict and display result
        prediction = heart_disease_model.predict([user_input])
        if prediction[0] == 1:
            heart_disease_result = "This person is likely to have heart disease."
        else:
            heart_disease_result = "This person is unlikely to have heart disease."
    
    st.success(heart_disease_result)

if selected == "Parkinsons Prediction":
    # Page title
    st.title("Parkinson's Disease Prediction using Machine Learning")

    # Grouped columns for better organization and tooltips for context
    st.subheader("Voice and Vocal Characteristics")
    st.subheader("Amplitude and Harmonics Features")
    st.subheader("Nonlinear Dynamics and Spread Measures")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', help="Average vocal fundamental frequency")
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', help="Maximum vocal fundamental frequency")
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', help="Minimum vocal fundamental frequency")
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', help="Frequency variation in voice (percentage)")
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', help="Absolute frequency variation in voice")

    with col1:
        RAP = st.number_input('MDVP:RAP', help="Relative amplitude perturbation")
    with col2:
        PPQ = st.number_input('MDVP:PPQ', help="Five-point period perturbation quotient")
    with col3:
        DDP = st.number_input('Jitter:DDP', help="Average absolute difference of period pairs")
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', help="Amplitude variation in voice")
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', help="Amplitude variation in voice (dB)")

    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', help="Three-point amplitude perturbation quotient")
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', help="Five-point amplitude perturbation quotient")
    with col3:
        APQ = st.number_input('MDVP:APQ', help="Amplitude perturbation quotient")
    with col4:
        DDA = st.number_input('Shimmer:DDA', help="Average absolute difference of amplitude pairs")
    with col5:
        NHR = st.number_input('NHR', help="Noise-to-harmonics ratio")

    with col1:
        HNR = st.number_input('HNR', help="Harmonics-to-noise ratio")

    with col2:
        RPDE = st.number_input('RPDE', help="Recurrence period density entropy")
    with col3:
        DFA = st.number_input('DFA', help="Signal fractal scaling exponent")
    with col4:
        spread1 = st.number_input('Spread1', help="Nonlinear measure of fundamental frequency variation")
    with col5:
        spread2 = st.number_input('Spread2', help="Nonlinear measure of fundamental frequency variation")

    with col1:
        D2 = st.number_input('D2', help="Correlation dimension")
    with col2:
        PPE = st.number_input('PPE', help="Pitch period entropy")

    # Code for Prediction
    parkinsons_diagnosis = ''

    # Creating a button for Prediction    
    if st.button("Get Parkinson's Disease Prediction"):
        # Prepare user input
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        # Convert inputs to float values
        user_input = [float(x) for x in user_input]

        # Make prediction
        parkinsons_prediction = parkinsons_model.predict([user_input])

        # Display prediction result
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person is likely to have Parkinson's disease."
        else:
            parkinsons_diagnosis = "The person is unlikely to have Parkinson's disease."
    
    st.success(parkinsons_diagnosis)

if selected == "Anemia Prediction":
    st.title("Anemia Prediction Using Machine Learning")

    # Input fields with enhanced usability
    st.subheader("Enter Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox("Gender", ("Male", "Female"), help="Select the patient's gender")
    with col2:
        Hemoglobin = st.number_input("Hemoglobin Level (g/dL)", min_value=0.0, max_value=25.0, step=0.1, help="Normal range: 13.8-17.2 g/dL for males, 12.1-15.1 g/dL for females")
    with col3:
        MCH = st.number_input("Mean Corpuscular Hemoglobin (MCH)", min_value=0.0, max_value=50.0, step=0.1, help="Normal range: 27-32 pg/cell")
    with col1:
        MCHC = st.number_input("Mean Corpuscular Hemoglobin Concentration (MCHC)", min_value=0.0, max_value=40.0, step=0.1, help="Normal range: 32-36 g/dL")
    with col2:
        MCV = st.number_input("Mean Corpuscular Volume (MCV)", min_value=0.0, max_value=120.0, step=0.1, help="Normal range: 80-100 fL")

    anemia_result = ""

    # Predict button
    if st.button("Get Anemia Test Result"):
        # Gender encoding
        gender_encoded = 0 if Gender == "Male" else 1

        # Collecting user inputs
        user_input = [gender_encoded, Hemoglobin, MCH, MCHC, MCV]
        
        # Making prediction
        prediction = anemia_model.predict([user_input])  # Assuming `anemia_model` is preloaded

        # Displaying the result
        anemia_result = "The person has anemia" if prediction[0] == 1 else "The person does not have anemia"
        st.success(anemia_result)

if selected == 'Breast Cancer Prediction':
    import pages.breast_cancer_predictor

if selected == "Lungs Cancer Predictin":
    
    st.title("Lung Cancer Classification Using Machine Learning")

    
    uploaded_file = st.file_uploader("Upload a lung scan image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        
        image_data = Image.open(uploaded_file)
        # Slider for image width with default value 300
        image_width = st.slider("Adjust image size", min_value=100, max_value=800, value=300)

        # Display the image with the selected width
        st.image(image_data, caption="Uploaded Image", width=image_width)

        img = image_data.resize(IMAGE_SIZE)
        img=img.convert('RGB')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array/255.0

        predictions = lung_cancer_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = LUNG_CANCER_CLASSES[predicted_class]

        st.write(f"### Predicted Class: {predicted_label}")
        st.write(f"#### Confidence: {predictions[0][predicted_class]:.2f}")

if selected == "Brain Tumor Prediction":

    st.title("Brain Tumor Prediction")

    st.write("Upload an image of a brain scan to predict if it contains a tumor.")
    uploaded_image = st.file_uploader("Choose a brain scan image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        # Slider for image width with default value 300
        image_width = st.slider("Adjust image size", min_value=100, max_value=800, value=300)

        # Display the image with the selected width
        st.image(uploaded_image, caption="Uploaded Image", width=image_width)
        # Predict the class
        prediction = img_pred(uploaded_image)

        st.write(f"Prediction: {prediction}")

if selected == "Skin Cancer Prediction":
    st.title("Skin Cancer Prediction")

    st.write("Upload an image of a skin lesion to predict the type of skin cancer.")
    uploaded_image = st.file_uploader("Choose a skin scan image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image_width = st.slider("Adjust image size", min_value=100, max_value=800, value=300)
        st.image(uploaded_image, caption="Uploaded Image", width=image_width)

        # Make prediction using the uploaded image
        prediction = skin_pred(uploaded_image)

        st.write(prediction)
 

