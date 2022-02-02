import streamlit as st
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
# Deployment purposes
from custom_transformer import column_transformer
import tensorflow as tf
import numpy as np
from PIL import Image


# Sidebar widget
st.sidebar.title("CNN Classification")
st.sidebar.header('Milestone 2 Phase 2')
st.sidebar.markdown("Dashboard CNN Classification of Sport Image")
# loading our model
@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('model_base_MobileNetV2.h5')
	return model

def main():
    page = st.sidebar.selectbox(
        "Select a page", ["Homepage", "Model" ,"Prediction"])

    if page == "Homepage":
        homepage_screen()
    elif page == "Model":
        model_screen()
    elif page == "Prediction":
        model_predict()


@st.cache()
def load_data():
    data = pd.read_csv('class_dict.csv', delimiter=",")
    return data


df = load_data()


def homepage_screen():
    st.image("imclass.png",use_column_width=True)
    st.title('SPORT IMAGE CLASSIFICATION')
    #st.header("Dataset Information")
    st.write("""  
        **About Dataset**  
        Collection of sports images covering 100 different sports.. Images are 224,224,3 jpg format. 
        Data is separated into train, test and valid directories.
        (https://www.kaggle.com/blastchar/telco-customer-churn).
    """)

    df = pd.read_csv('class_dict.csv', delimiter=",")
    
        # Load data
    if st.checkbox("Show Dataset"): #checkbox untuk menampilkan dataset
            number=st.number_input("Number of Rows to view",5,15)
            st.dataframe(df.head(number))
            st.success("Data loaded successfully") 
            
            
            data_dim= st.radio("Shape of the dataset:", ("Number of Rows","Number of Columns")) #radio button widget
            if st.button("Show Dimension"):
                if data_dim== 'Number of Rows':
                    st.write(df.shape[0])
                elif data_dim== 'Number of Columns':
                    st.write(df.shape[1])
                else:
                    st.write(df.shape)

            Info =['Display the dataset summary','Check for missing values in the dataset']
            options=st.selectbox("Pilihan - pilihan terhadap dataset",Info)
            
            if options=='Check for missing values in the dataset': 
                 st.write(df.isnull().sum(axis=0)) #cekk null values
                 if st.button("Drop Null Values"):
                     df=df.dropna() #drop null values
                     st.success("Null values droped successfully")
            
            if options=='Display the dataset summary':
                 st.write(df.describe().T)    

def model_screen():
    st.image("CNN.png",use_column_width=True)
    st.title("Model")
    st.write(""" 
             CNN Model Evaluation
             """)
    model_selected = st.selectbox("Select Evaluation Type: ", ['Train Base Model','Train Base Model + Improve',
    'PRE-TRAINED MODEL (MobileNetV2)','PRE-TRAINED MODEL (MobileNetV2) + Fine Tuning' ])
   
    if model_selected ==  'Train Base Model':
        st.image("seq.png",use_column_width=True)
        st.write(f"""
                👉 Validation's loss value: 10.1962
                👉 Validation's accuracy value: 0.2460
                """)

        
    if model_selected ==  'Train Base Model + Improve':
        st.image("seqs.png",use_column_width=True)
        st.write(f"""
                👉 Validation's loss value: 2.2369
                👉 Validation's accuracy value: 0.4520
                """)
        st.info('Model Improving with Data Augmentation and Dropout')  
        st.info('Model Accuracy = 46,99%')
        st.info('In Notebook The result of model inference is, image most likely belongs to bull riding with a 62.48 percent confidence.')      
           
    if model_selected ==  'PRE-TRAINED MODEL (MobileNetV2)':
        st.image("pre.png",use_column_width=True)
        st.write(f"""
                👉 Validation's loss value: 0.2344
                👉 Validation's accuracy value: 0.9419
                """)
        st.write(f"With the pre-trained MobileNetV2 model, the results obtained are much better than the previous model. With just 20 epochs, the results were already very good. The resulting graph also looks good between the increase in accuracy and the decrease in loss from training data and validation data.")        
        st.info('In Notebook The result of model inference is, image most likely belongs to axe throwing with a 99.99 percent confidence.') 
          
    if model_selected ==  'PRE-TRAINED MODEL (MobileNetV2) + Fine Tuning':
        st.image("prefine.png",use_column_width=True)
        st.write(f"""
                👉 Validation's loss value: 0.1393
                👉 Validation's accuracy value: 0.9639
                """)
        st.write(f"The results of fine tuning experiments from the previous model can be seen from the plot. Previously the accuracy obtained by the model was good at 95.7%. and after fine tuning rose to 96.99%.")        
        st.info('In Notebook The result of model inference is, image most likely belongs to axe throwing with a 100 percent confidence.')                 
        st.info('For the inference model itself which previously got a confidence percentage prediction score of 99.99% this is very good then after fine tuning the score rises to 100% this is a very perfect score.')

# def validation(matrix):
#     fig, ax = plt.subplots()
#     sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
#     ax.set_xlabel('PREDICTED')
#     ax.set_ylabel('ACTUAL')
#     ax.set_title('Confusion Matrix')
#     st.write(fig)

def model_predict():
    st.image("predict.png",use_column_width=True)
    st.title("Sport Image Prediction")
    st.write("### Predict image JPG (224 x 224 pixel. JPG) !")
    @st.cache(allow_output_mutation=True)

    def load_model():
        model = tf.keras.models.load_model('model_base_MobileNetV2.h5')
        return model

    def predict_class(image, model):

        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [224, 224])

        image = np.expand_dims(image, axis = 0)

        prediction = model.predict(image)

        return prediction

    model = load_model()

    file = st.file_uploader("Upload an image", type=["jpg", "png"])


    if file is None:
        st.text('Waiting for upload....')

    else:
        slot = st.empty()
        slot.text('Running inference....')

        test_image = Image.open(file)

        st.image(test_image, caption="Input Image", width = 400)

        pred = predict_class(np.asarray(test_image), model)

        class_names = ['air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing', 'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling', 'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding', 'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket', 'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men', 'figure skating pairs', 'figure skating women', 'fly fishing', 'football', 'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 'golf', 'hammer throw', 'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping', 'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing', 'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling', 'luge', 'motorcycle racing', 'mushing', 'nascar racing', 'olympic wrestling', 'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo', 'pommel horse', 'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby', 'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing', 'ski jumping', 'sky surfing', 'skydiving', 'snow boarding', 'snowmobile racing', 'speed skating', 'steer wrestling', 'sumo wrestling', 'surfing', 'swimming', 'table tennis', 'tennis', 'track bicycle', 'trapeze', 'tug of war', 'ultimate', 'uneven bars', 'volleyball', 'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball', 'wheelchair racing', 'wingsuit flying']

        result = class_names[np.argmax(pred)]

        output = 'The image is a ' + result

        slot.text('Done')

        st.success(output)


main()
