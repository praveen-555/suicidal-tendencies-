import streamlit as st
import json
import os
import re
import string
# from streamlit_webrtc import webrtc_streamer
from st_audiorec import st_audiorec 
# import soundfile as sf
# import speech_recognition as sr
import io
import whisper
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from pydub import AudioSegment
import pickle


session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0
with open('logistic.pik', 'rb') as f:
    model1 = pickle.load(f)

with open('tfidf_vectorizer.pik', 'rb') as f:
    vectorizer = pickle.load(f)
    
def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None
def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")
        
def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,

        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None


def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")

    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
    
def preprocess(text): 
    def remove_url(text):
        return re.sub(r"http\S+", "", text) 
    exclude =string.punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans("", "" , exclude))
    
    def remove_stopwords(text):
        stopword = stopwords.words('english')
        new_text = []
        for word in text.split():
            if word in stopword:
                new_text.append('')
            else:
                new_text.append(word)
        x = new_text[:]
        new_text.clear()
        return " ".join(x)
    def lemmatize_text(text):
        words = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)
    text=remove_url(text)
    text=remove_punctuation(text)
    text=remove_stopwords(text)
    text=lemmatize_text(text)
    return text
def predict_function(user_input):
    text = preprocess(user_input)
    tfidf_vector = vectorizer.transform([text])
    pred_1 = model1.predict(tfidf_vector)
    first_prediction = pred_1[0]
    
    st.write('**Tendencies:**', first_prediction)
    
    st.write("**Resources:**")

    if(first_prediction=='depression'):
        st.write("Here are some resources to help cope with depression:")
        st.markdown("- [How to cope up with depression](https://www.youtube.com/watch?v=8Su5VtKeXU8)")
        st.markdown("- [Ways to cope with depression](https://www.medicalnewstoday.com/articles/327018)")
        st.markdown("- [Depression Treatment](https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/depression-treatment-and-management)")
        st.markdown("- [Don’t suffer from Depression in Silence](https://www.ted.com/talks/nikki_webber_allen_don_t_suffer_from_your_depression_in_silence?hasSummary=true&language=en)")
        
    elif(first_prediction=='suicide'):
        st.write("Some resources for controlling suicidal tendencies:")
        st.markdown("- [Are you feeling Suicidal?](https://www.helpguide.org/articles/suicide-prevention/are-you-feeling-suicidal.htm)")
        st.markdown("- [Suicide and Suicidal Thoughts](https://www.mayoclinic.org/diseases-conditions/suicide/symptoms-causes/syc-20378048)")
        st.markdown("- [Suicide Prevention](https://www.ted.com/talks/ashleigh_husbands_suicide_prevention?hasSummary=true)")
        st.markdown("- [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/)")
        st.markdown("- [Crisis Text Line](https://www.crisistextline.org/)")
        st.markdown("- [International Association for Suicide Prevention](https://www.iasp.info/resources/Crisis_Centres/)")
                    
    else : 
        st.write("Great job! Keep up the positivity and stay clear of negative thinking.")
                                                
def transcribe_audio_from_data(file_data):
    with open("temp.mp3", "wb") as f:
        f.write(file_data)
    model = whisper.load_model("base")
    result = model.transcribe("temp.mp3")
    os.remove("temp.mp3")
    return result['text']

def main(json_file_path="data.json"):
    st.sidebar.title("Tendency Text")
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "Dashboard", "Text Classification"),
        key="comment_classifying",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Text Classification":
        if session_state.get("logged_in"):
            st.title("Text Classifiation")
            
              # Upload audio file
            selection = st.radio(
            "Choose an option:",
            ("Text", "Audio")
        )
            if selection== "Text":
                user_input = st.text_area("Enter your Comment here:", height=200)  
                if st.button("Submit"):
                    predict_function(user_input)

    
            else:
                options = ["Record", "Upload"]
                choice = st.radio("Choose an option", options)
                 # Check the choice and display the corresponding widget
                if choice == "Record":
                    st.write("Click the button below to start recording:")
                    audio = st_audiorec()
                    if audio is not None and st.button('Submit'):
                       user_input = transcribe_audio_from_data(audio)
                       predict_function(user_input)
                       
                
                elif choice == "Upload":
                    audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
                    if audio is not None:
                        st.audio(audio, format="audio/wav")
                        if st.button("Submit"):
                            user_input= transcribe_audio_from_data(audio.read())
                            predict_function(user_input)
            # if st.button("Submit"):
            #     text = preprocess(user_input)
            #     tfidf_vector = vectorizer.transform([text])
            #     pred_1 = model1.predict(tfidf_vector)
            #     first_prediction = pred_1[0]
                
            #     st.write('**Tendencies:**', first_prediction)
                
            #     st.write("**Resources:**")

            #     if(first_prediction=='depression'):
            #         st.write("Here are some resources to help cope with depression:")
            #         st.markdown("- [How to cope up with depression](https://www.youtube.com/watch?v=8Su5VtKeXU8)")
            #         st.markdown("- [Ways to cope with depression](https://www.medicalnewstoday.com/articles/327018)")
            #         st.markdown("- [Depression Treatment](https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/depression-treatment-and-management)")
            #         st.markdown("- [Don’t suffer from Depression in Silence](https://www.ted.com/talks/nikki_webber_allen_don_t_suffer_from_your_depression_in_silence?hasSummary=true&language=en)")
                    
            #     elif(first_prediction=='suicide'):
            #         st.write("Some resources for controlling suicidal tendencies:")
            #         st.markdown("- [Are you feeling Suicidal?](https://www.helpguide.org/articles/suicide-prevention/are-you-feeling-suicidal.htm)")
            #         st.markdown("- [Suicide and Suicidal Thoughts](https://www.mayoclinic.org/diseases-conditions/suicide/symptoms-causes/syc-20378048)")
            #         st.markdown("- [Suicide Prevention](https://www.ted.com/talks/ashleigh_husbands_suicide_prevention?hasSummary=true)")
            #         st.markdown("- [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/)")
            #         st.markdown("- [Crisis Text Line](https://www.crisistextline.org/)")
            #         st.markdown("- [International Association for Suicide Prevention](https://www.iasp.info/resources/Crisis_Centres/)")
                                
            #     else : 
            #         st.write("Great job! Keep up the positivity and stay clear of negative thinking.")
                                                

    else:
        st.warning("Please login/signup to use the app")
            
if __name__ == "__main__":
    initialize_database()
    main()
