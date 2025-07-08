import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Updated title and description
st.title("✍️ Arabic Handwriting Extraction")
st.write(
    "Upload an image containing Arabic handwriting – Gemini will extract the text! "
    "Get your Google Gemini API key [here](https://aistudio.google.com/app/apikey)."
)

# Get Gemini API key from user
gemini_api_key = st.text_input("Gemini API Key", type="password")
if not gemini_api_key:
    st.info("Please add your Gemini API key to continue.", icon="🗝️")
    st.stop()  # Stop execution if no key

# Configure Gemini with user-provided key
genai.configure(api_key=gemini_api_key)

# Image uploader (accepts PNG/JPEG)
uploaded_file = st.file_uploader(
    "Upload Handwriting Image",
    type=["png", "jpg", "jpeg"],
    help="Take a clear photo of handwritten Arabic text"
)

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Process image when user clicks button
    if st.button("Extract Text"):
        # Read image bytes
        image_bytes = uploaded_file.getvalue()
        
        # Initialize the vision model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create prompt for Arabic handwriting extraction
        prompt = "Extract the Arabic handwriting text EXACTLY as written. Return only the raw text without any translations, explanations, or formatting."
        
        # Generate content
        with st.spinner("Extracting handwriting..."):
            try:
                response = model.generate_content(
                    contents=[prompt, {"mime_type": uploaded_file.type, "data": image_bytes}]
                )
                
                # Display extracted text
                st.subheader("Extracted Text")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Ensure your API key is valid and supports Gemini Vision")
