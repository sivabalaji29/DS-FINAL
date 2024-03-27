# impoting library  
import streamlit as st 
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
# In this image processing
from PIL import Image,ImageFilter,ImageEnhance
from PIL import ImageOps
import plotly.express as px
import easyocr
# In this Text processing
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
# In this library using Recommendation 
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# streamlit webpage design
def set_page_config():
    st.set_page_config(
        page_title="",
        page_icon="https://uxwing.com/wp-content/themes/uxwing/download/business-professional-services/column-chart-line-icon.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={'About': """# This OCR app is created by *SIVABALAJI*!"""}
    )

set_page_config()
# This is Application background image._.
def setting_bg():
    st.markdown(
        """
        <style>
            .stApp {
                background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQn3ayvZFQIM9arN8ll_szWh_4dW6-FU_iYuVcVZnYEl_F9ntEOwF9hHAUT8OfMRmkY-uE&usqp=CAU");
                background-size: 100% 100vh;
                background-repeat: no-repeat;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

setting_bg()
#  hide the streamlit main and footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

with st.sidebar:
    # To create a selecting option menu..
    selected = option_menu(None, ["HOME", "CUSTOMER BEHAVIOR PREDICTION", "IMAGE-PROCESSING", "TEXT-PROCESSING", "PRODUCT RECOMMENDATION-SYSTEM", "PROFILE"],
                                        icons=["house", "pie-chart", "image", "file", "body-text", "person-circle"],
                                        default_index=0,
                                        styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "0px",
                                                            "--hover-color": "#6495ED"},
                                                "icon": {"font-size": "20px"},
                                                "container": {"max-width": "300px"},
                                                "nav-link-selected": {"background-color": "#93cbf2"}})

    text_process = st.expander("Text Processing", expanded=False)

if selected == "HOME":
    st.markdown("<h1 style='text-align: center; color: #f72323;'>Customer Insights and Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: justify; font-size: 30px; font-weight: bold;">
        <p style="font-size: 25px; text-align: justify;">
            <span style="color: black;">Classification Prediction:</span> In the classification prediction model, we aim to analyze customer behavior using the following algorithms: Decision Tree, Logistic Regression, and Random Forest.<br>
            <span style="color: black;">Image Processing:</span> In this module, we process images using techniques such as EasyOCR (Optical Character Recognition) to extract text from images, and the Python Imaging Library (PIL) to identify and extract objects from images. Additionally, PIL can be used to modify images by changing formats, rotating, and manipulating pixel sizes.<br>
            <span style="color: black;">Text Processing:</span> In this module, we provide sentiment analysis for text based on user input, utilizing text processing techniques such as NLTK (Natural Language Toolkit).<br>
            <span style="color: black;">Product Recommendation System:</span> Build a recommendation system for product selection using NLTK techniques.
        </p>
    </div>
""", unsafe_allow_html=True)
    col1,col2 = st.columns([2,2])
    with col1:
        st.write('### :red[TECHNOLOGY USED]')
        st.write('- PYTHON   (PANDAS, NUMPY)')
        st.write('- SCIKIT-LEARN')
        st.write('- DATA PREPROCESSING')
        st.write('- EXPLORATORY DATA ANALYSIS')
        st.write('- OCR')
        st.write('- NLTK')
        st.write('- WorldCould')
        st.write('- STREAMLIT')
    with col2:
        st.write("### :red[MACHINE LEARNING MODEL]")
        st.write('#### :red[CLASSIFICATION] - ***:red[RANDOMFOREST CLASSIFIER,LOGISTIC REGRESSION, DECISION TREE]***')
        st.write('- The RandomForestClassifier is an ensemble learning method that combines multiple decision trees to create a robust and accurate classification model.')
        st.write('- Logistic regression estimates the probability of an event occurring, such as voted or didnt vote, based on a given dataset of independent variables.')
        st.write('- A decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks.')
  #--Classification Prediction --#
if selected == "CUSTOMER BEHAVIOR PREDICTION":
    st.markdown("<h1 style='text-align: center; color: #f72323;'>Classification Prediction</h1>", unsafe_allow_html=True)
    selected = option_menu(None, ["Algorithms","Prediction"],
                       icons=["clipboard-data","graph-down"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "35px", "text-align": "center", "margin": "0px",
                                           "--hover-color": "#6495ED"},
                               "icon": {"font-size": "35px"},
                               "container": {"max-width": "6000px"},
                               "nav-link-selected": {"background-color": "#93cbf2"}})

    text_process = st.expander("Text Processing", expanded=False)
    if selected == "Algorithms":
        df = pd.DataFrame({
        "Algorithm Names":["Decision Tree","Logistic Regression ","Random Forest"],
        "Accuracy":[94,78,96],
        "Precision":[94,82,95],
        "Recall":[93,60,95],
        "F1_score":[93,69,95]
    })
        
        st.table(df)
    if selected == "Prediction":
        with st.form(key='my_form'):
            c1,c2 = st.columns(2)
            with c1:
                st.subheader(":red[Min & Max given for reference, you can enter any value]")
                # st.write( f'<h5 style="color:red;">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                transactionRevenue = st.number_input("Enter transactionRevenue (Min:0.0 & Max:307221222.5)",value=None)
                num_interactions = st.number_input("Enter num_interactions (Min:20.0 & Max:25911.0)",value=None)
                count_hit = st.number_input("Enter count_hit (Min:2, Max:7085.0)",value=None)
                st.markdown("")
                historic_session_page = st.number_input("historic_session_page (Min:0.0, Max:5021.25)",value=None)
                for _ in range(2): 
                    st.markdown(" ")     
            with c2:
                time_on_site = st.number_input("time_on_site (Min:0.0, Max:26652.75)",value=None)
                avg_session_time = st.number_input("avg_session_time (Min:2.0, Max:1109.1536494755242)",value=None)
                avg_session_time_page = st.number_input("avg_session_time_page (Min:0.0, Max:339.53914141414145)",value=None)
                historic_session = st.number_input("historic_session (Min:2.0, Max:23271.5)",value=None)
                visits_per_day = st.number_input("visits_per_day (Min:0.9230769230769232, Max:304.8440708626405)",value=None)
                submit_button = st.form_submit_button(label="PREDICT STATUS")
                st.markdown("""
                            <style>
                            div.stButton > button:first-child {
                                background-color: #009999;
                                color: white;
                                width: 100%;
                            }
                            </style>
                        """, unsafe_allow_html=True)
            
            for i in ["transactionRevenue","num_interactions","count_hit","historic_session_page","time_on_site","avg_session_time","avg_session_time_page","historic_session","visits_per_day"]:
                if submit_button :
                    with open(r"C:\Users\SIVABALAJI S\Desktop\E customer\Model.pkl", 'rb') as file:
                        loaded_model = pickle.load(file)
                    with open(r"C:\Users\SIVABALAJI S\Desktop\E customer\Scaler.pkl", 'rb') as f:
                        scaler_loaded = pickle.load(f)
                    with open(r"C:\Users\SIVABALAJI S\Desktop\E customer\ct.pkl", 'rb') as f:
                        t_loaded = pickle.load(f)

                # Predict the has_converted for a new sample
                    new_sample = np.array([[float(transactionRevenue),float(num_interactions),float(count_hit),float(historic_session_page),float(time_on_site),float(avg_session_time),float(avg_session_time_page),float(historic_session),float(visits_per_day)]])
                    try:
                        new_sample = np.array((new_sample[:, [0,1,2, 3, 4, 5, 6,7,8]]))
                        new_sample = scaler_loaded.transform(new_sample)
                        new_pred = loaded_model.predict(new_sample)
                        if new_pred== 1:
                            st.write('## :green[The Status is Converted] ')
                            break
                        else:
                            st.write('## :red[The Status is Not Converted] ')
                            break
                    except ValueError as e:
                        st.write(f"Error: {e}")
                        st.write("Please make sure all input values are valid numbers.")
                        break
    uploaded_file=st.sidebar.file_uploader(label="Upload your csv or excel file.max(200mb)",type=["csv","xlsx"])

    if uploaded_file is not None:
        print(uploaded_file)
        
        try:
            df=pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            df=pd.read_excel(uploaded_file)
    try:
        st.write(df)
        numeric_columns= list(df.select_dtypes(["float","int"]).columns)
        categorical_column=list(df.select_dtypes("object").columns)
    except Exception as e:
        print(e)
        st.write("Please upload file to the application.")
    #add a select widget tot the  sidebar
    chart_select=st.sidebar.selectbox(label="select the chart type",
                                    options=["Scatterplots","Barcharts","Boxplot","Histogram"])

    if chart_select=="Scatterplots":
        st.sidebar.subheader("Scatterplot settings")
        try:
            x_values= st.sidebar.selectbox("X axis",options=numeric_columns)
            y_values= st.sidebar.selectbox("Y axis",options=numeric_columns)
            plot=px.scatter(data_frame=df,x=x_values,y=y_values)
            #display the chart 
            st.plotly_chart(plot)
        except Exception as e:
            print(e)


    if chart_select=="Barcharts":
        st.sidebar.subheader("Barcharts settings")
        try:
            x_values= st.sidebar.selectbox("X axis",categorical_column)
            y_values= st.sidebar.selectbox("Y axis",options=numeric_columns)
            plot=px.bar(data_frame=df,x=x_values,y=y_values)
            #display the chart 
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select=="Boxplot":
        st.sidebar.subheader("Boxplot settings")
        try:
            x_values= st.sidebar.selectbox("X axis",categorical_column)
            y_values= st.sidebar.selectbox("Y axis",options=numeric_columns)
            plot=px.box(data_frame=df,x=x_values,y=y_values)
            #display the chart 
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select=="Histogram":
        st.sidebar.subheader("Barcharts settings")
        try:
            y_values= st.sidebar.selectbox("Y axis",options=numeric_columns)
            plot=px.histogram(data_frame=df,x=y_values,nbins=20)
            #display the chart 
            st.plotly_chart(plot)
        except Exception as e:
            print(e)                
  
  #-- Image Processing --#  
if selected == "IMAGE-PROCESSING":
    st.markdown("<h1 style='text-align: center; color: #f72323;'>Image Processing</h1>", unsafe_allow_html=True)
    uploaded_Image_file = st.file_uploader(label="Upload your jpg or jpeg file. Max size: 200mb", type=["jpg", "jpeg"])
    if uploaded_Image_file is not None:
        # Open the original image view
        original_image = Image.open(uploaded_Image_file)
        st.image(original_image, caption="Original Image")
        resized_image = original_image.resize((1000, 500))
        
        # Perform OCR on the resized image
        np_img = np.array(resized_image) 
        reader = easyocr.Reader(['en'])  
        ocr_result = reader.readtext(np_img)  
        # Display OCR results
        ocr_text = [result[1] for result in ocr_result] if ocr_result else []
        if ocr_text:
            st.success("OCR Text: " + "\n".join(ocr_text))
        else:
            st.error("No text found : 🥺")

        try: 
            # Buttons for image processing arranged horizontally
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                if st.button("Convert to Grayscale"):
                    gray_image = resized_image.convert("L")
                    st.image(gray_image, caption="Grayscale Image")
            with col2:
                if st.button("Apply Gaussian Blur"):
                    blurred_image = resized_image.filter(ImageFilter.GaussianBlur(radius=10))
                    st.image(blurred_image, caption="Blurred Image")
            with col3:
                if st.button("Enhance Contrast"):
                    blurred_image = resized_image.filter(ImageFilter.GaussianBlur(radius=10))
                    contrast_enhanced_image = ImageEnhance.Contrast(blurred_image)
                    st.image(contrast_enhanced_image.enhance(20), caption="Contrast Enhanced Image")
            with col4:
                if st.button("Rotated Image"):
                    st.image(resized_image.rotate(90))
                    st.image(resized_image.rotate(180))
                    st.image(resized_image.rotate(270))
                    st.image(resized_image.rotate(360))
            with col5:
                if st.button("Mirror Image"):
                    st.image(ImageOps.mirror(resized_image), caption="Mirror Image")
            with col6:
                if st.button("Brightened Image"):
                    bright = ImageEnhance.Brightness(resized_image)
                    bright_1 = bright.enhance(3)
                    st.image(bright_1, caption="Brightened Image")
            with col7:
                if st.button("Negative & Edge Detected Image"):
                    neg_image = ImageOps.invert(resized_image)
                    st.image(neg_image)
                    Edge_det_image = resized_image.filter(ImageFilter.FIND_EDGES)
                    st.image(Edge_det_image)
                if st.button("Sharpened & Framed Image"):
                    sharped_image = ImageEnhance.Sharpness(resized_image)
                    st.image(sharped_image.enhance(10))
                    Framed_img = ImageOps.expand(resized_image, 10, "black")
                    st.image(Framed_img)
        except Exception as e:
            st.error("Error: " + str(e))    
  
  # --This is Text Processing --#       
if selected == "TEXT-PROCESSING":
    st.markdown("<h1 style='text-align: center; color: #f72323;'>Text Processing</h1>", unsafe_allow_html=True)
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('vader_lexicon')
    input_text = st.text_input("Enter the Text")
    # NLP Pre-processing
    if input_text:
        # Tokenization
        words = word_tokenize(input_text)
        sentences = sent_tokenize(input_text)

        # Stopword removal
        stop_words = set(stopwords.words('english'))
        filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

        # Display pre-processing results
        st.header("NLP Pre-processing")
        st.subheader("Tokenization")
        st.write(words)

        st.subheader("Stopword Removal")
        st.write(filtered_words)

        st.subheader("Sentence Tokenization")
        st.write(sentences)

        # Keyword extraction
        st.header("Keyword Extraction")
        word_freq = Counter(filtered_words)
        keywords = word_freq.most_common(5)  # Display top 5 keywords
        st.write("Keywords:", [word[0] for word in keywords])

        # Sentiment Analysis
        st.header("Sentiment Analysis")
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(input_text)
        st.write("Sentiment Score:", sentiment_score)

        # Display overall sentiment
        if sentiment_score['compound'] >= 0.05:
            sentiment_label = "Positive"
        elif sentiment_score['compound'] <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        st.write("Sentiment:", sentiment_label)
        
        col1, col2 = st.columns([2, 2], gap="medium")
        with col1:
            WC = WordCloud(width=4000, height=3250).generate(input_text)
            plt.figure(1, figsize=(10, 10))
            plt.imshow(WC)
            plt.axis('off')  # Turn off axis labels
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        with col2:
            color_palette = {"neg": "red", "neu": "blue", "pos": "green", "compound": "purple"}

            # Create a DataFrame for visualization
            data = {'sentiment_score': sentiment_score.values()}
            df = pd.DataFrame(data, index=sentiment_score.keys())
            df = df.reset_index().rename(columns={'index': 'sentiment'})  # Add sentiment labels as a column

            # Plot the bar chart with correct color palette
            fig, ax = plt.subplots()
            sns.barplot(x='sentiment', y='sentiment_score', data=df, palette=[color_palette[label] for label in df['sentiment']], ax=ax)

            ax.set_title('Sentiment Analysis BarPlot', fontsize=20)
            st.pyplot(fig)
            
  # --Recommendation System-- #
if selected == "PRODUCT RECOMMENDATION-SYSTEM":
    st.markdown("<h1 style='text-align: center; color: #f72323;'>Product Recommendation System</h1>", unsafe_allow_html=True)
    data = pd.read_csv(r"C:\Users\SIVABALAJI S\Desktop\E customer\amazon_product.csv")
    # Remove unnecessary columns
    data = data.drop('id', axis=1)

    # Define tokenizer and stemmer
    stemmer = SnowballStemmer('english')
    def tokenize_and_stem(text):
        tokens = nltk.word_tokenize(text.lower())
        stems = [stemmer.stem(t) for t in tokens]
        return stems

    # Create stemmed tokens column
    data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

    # Define TF-IDF vectorizer and cosine similarity function
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
    def cosine_sim(text1, text2):
        # tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
        text1_concatenated = ' '.join(text1)
        text2_concatenated = ' '.join(text2)
        tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
        return cosine_similarity(tfidf_matrix)[0][1]

    # Define search function
    def search_products(query):
        query_stemmed = tokenize_and_stem(query)
        data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
        results = data.sort_values(by=['similarity'], ascending=False).head(10)[['Title', 'Description', 'Category']]
        return results

    # web app
    st.title(':red[Search Engine and Product Recommendation System ON Am Data]')
    query = st.text_input("Enter Product Name")
    sumbit = st.button('Search')
    if sumbit:
        res = search_products(query)
        st.write(res)
    

  # --MY progile-- #   
if selected == "PROFILE":
    st.subheader(":red[DATA SCIENCE FINAL PROJECT]",divider='rainbow')
    st.subheader(":red[The objective of this project is to:]")
    st.markdown("""
                <div style="text-align: justify; font-size: 30px;">
                    <p style="font-size: 25px; text-align: justify;">
                        The objective of this project is to perform a comprehensive analysis and implement various tasks, including Exploratory Data Analysis (EDA) on an e-commerce dataset, image processing, Natural Language Processing (NLP), and the development of a recommendation system. It is important to note that executing these steps will necessitate a sound understanding of the relevant tools and libraries.
                    </p></div>""", unsafe_allow_html=True)
    col1,col2 = st.columns([3,3],gap="medium")
    with col1:
        for _ in range(5):
            st.write(" ")
        # Create additional vertical space
        for _ in range(3):
            st.write(" ")
        st.markdown("### :orange[Name:  ] :blue[SIVABALAJI]")
        st.markdown("### :orange[GitHub] ⬇️")
        github_url = "https://github.com/sivabalaji29"
        button_color = "#781734"
        # Create a button with a hyperlink
        button_html = f'<a href="{github_url}" target="_blank"><button style="font-size: 16px; background-color: {button_color}; color: #fff; padding: 8px 16px; border: none; border-radius: 4px;">GitHub</button></a>'
        st.markdown(button_html, unsafe_allow_html=True)
    
    with col2:
        # Create vertical space using empty containers
        for _ in range(5):
            st.write(" ")
        # Create additional vertical space
        for _ in range(3):
            st.write(" ")
        st.markdown("### :orange[Email: sivabalaji10000@gmail.com] ")
        st.markdown("### :orange[LinkedIn] ⬇️")
        linkedin_url = "https://www.linkedin.com/in/sivabalaji-s-a92979251/"
        button_color = "#781734"
        button_html = f'<a href="{linkedin_url}" target="_blank"><button style="font-size: 16px; background-color: {button_color}; color: #fff; padding: 8px 16px; border: none; border-radius: 4px;">My LinkedIn profile</button></a>'
        st.markdown(button_html, unsafe_allow_html=True)