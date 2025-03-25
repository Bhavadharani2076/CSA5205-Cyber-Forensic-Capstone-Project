import os
from flask import Flask, render_template, send_file, jsonify, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import spacy
from wordcloud import WordCloud
from textblob import TextBlob
from fpdf import FPDF
import logging
import tempfile
from PIL import Image
from PIL.ExifTags import TAGS

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Download stopwords
nltk.download('stopwords')

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.debug(f"Created uploads folder: {UPLOAD_FOLDER}")

def extract_exif_metadata(image_path):
    """
    Extracts EXIF metadata from an image file.
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data:
            exif_metadata = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_metadata[tag] = value
            return exif_metadata
        else:
            return None
    except Exception as e:
        logging.error(f"Error extracting EXIF metadata: {e}")
        return None

def get_unique_filename(file_path):
    """
    Generates a unique filename by appending a number if the file already exists.
    """
    base_name, ext = os.path.splitext(file_path)
    counter = 1
    while os.path.exists(file_path):
        file_path = f"{base_name}_{counter}{ext}"
        counter += 1
    return file_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    """
    Handles file uploads and processes the uploaded file for forensic analysis.
    """
    logging.debug("Upload route called")
    
    # Check if a file is uploaded
    if 'file' not in request.files:
        logging.error("No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # Check if a file is selected
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({"error": "No file selected"}), 400

    logging.debug(f"File received: {file.filename}")

    # Generate a unique file path
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file_path = get_unique_filename(file_path)
    file.save(file_path)
    logging.debug(f"File saved to: {file_path}")

    # Process the file and generate the report
    try:
        # Check file extension
        if file.filename.endswith('.xlsx'):
            logging.debug("Loading Excel file...")
            sheets = pd.read_excel(file_path, sheet_name=None)
            combined_df = pd.concat(sheets.values(), ignore_index=True)
            logging.debug("Excel file loaded successfully.")
        elif file.filename.endswith('.txt'):
            logging.debug("Loading text file...")
            with open(file_path, 'r') as f:
                text_data = f.read()
            combined_df = pd.DataFrame({ 'Message (Plaintext)': [text_data] })
            logging.debug("Text file loaded successfully.")
        else:
            logging.error("Unsupported file format")
            return jsonify({"error": "Unsupported file format"}), 400

        # Check if the required text column exists
        text_column = 'Message (Plaintext)'
        if text_column not in combined_df.columns:
            logging.warning(f"Column '{text_column}' not found in the uploaded file. Checking for image files...")
            exif_analysis = True
        else:
            exif_analysis = False

        # If plain text column is found, perform text analysis
        if not exif_analysis:
            # Drop rows with missing text data
            combined_df.dropna(subset=[text_column], inplace=True)

            # Text Preprocessing
            logging.debug("Preprocessing text...")
            nlp = spacy.load('en_core_web_sm')
            stop_words = set(stopwords.words('english'))

            def preprocess_text(text):
                if pd.isna(text) or text.strip() == "":
                    return ""
                doc = nlp(text)
                tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in stop_words]
                return ' '.join(tokens)

            combined_df['processed_text'] = combined_df[text_column].apply(preprocess_text)
            combined_df = combined_df[combined_df['processed_text'].str.strip() != ""]

            # Check if DataFrame is empty after preprocessing
            if combined_df.empty:
                logging.error("No valid text data found after preprocessing.")
                return jsonify({"error": "No valid text data found after preprocessing."}), 400

            # Topic Modeling
            logging.debug("Performing topic modeling...")
            tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(combined_df['processed_text'])
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(tfidf_matrix)

            def display_topics(model, feature_names, no_top_words):
                topics = []
                for topic_idx, topic in enumerate(model.components_):
                    topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
                return topics

            no_top_words = 10
            topics = display_topics(lda, tfidf.get_feature_names_out(), no_top_words)

            # Word Cloud
            logging.debug("Generating word cloud...")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(combined_df['processed_text']))
            wordcloud_path = os.path.join(tempfile.gettempdir(), 'wordcloud.png')
            wordcloud.to_file(wordcloud_path)

            # Sentiment Analysis
            logging.debug("Performing sentiment analysis...")
            combined_df['sentiment'] = combined_df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
            sentiment_path = os.path.join(tempfile.gettempdir(), 'sentiment_distribution.png')
            plt.figure(figsize=(8, 6))
            sns.histplot(combined_df['sentiment'], bins=30, kde=True)
            plt.title('Sentiment Distribution')
            plt.xlabel('Sentiment Polarity')
            plt.ylabel('Frequency')
            plt.savefig(sentiment_path, bbox_inches='tight')
            plt.close()

            # Suspicious Activity Detection
            logging.debug("Detecting suspicious activity...")
            suspicious_keywords = ["hack", "steal", "delete", "malware", "virus", "phishing", "fraud", "intrusion"]
            combined_df['suspicious_activity'] = combined_df[text_column].apply(
                lambda x: any(keyword in x.lower() for keyword in suspicious_keywords)
            )

            # Deleted Files Detection
            logging.debug("Detecting deleted files...")
            deleted_keywords = ["deleted", "removed", "erased", "wiped"]
            combined_df['deleted_files'] = combined_df[text_column].apply(
                lambda x: any(keyword in x.lower() for keyword in deleted_keywords)
            )

            # Visualization of Suspicious Activity and Deleted Files
            suspicious_activity_path = os.path.join(tempfile.gettempdir(), 'suspicious_activity.png')
            plt.figure(figsize=(10, 5))
            sns.countplot(x='suspicious_activity', data=combined_df)
            plt.title('Suspicious Activity Detection')
            plt.xlabel('Suspicious Activity')
            plt.ylabel('Count')
            plt.savefig(suspicious_activity_path, bbox_inches='tight')
            plt.close()

            deleted_files_path = os.path.join(tempfile.gettempdir(), 'deleted_files.png')
            plt.figure(figsize=(10, 5))
            sns.countplot(x='deleted_files', data=combined_df)
            plt.title('Deleted Files Detection')
            plt.xlabel('Deleted Files')
            plt.ylabel('Count')
            plt.savefig(deleted_files_path, bbox_inches='tight')
            plt.close()

        # If plain text column is not found, check for image files and extract EXIF metadata
        else:
            logging.debug("Examining EXIF metadata...")
            exif_results = []
            for index, row in combined_df.iterrows():
                if 'Image Path' in row:
                    image_path = row['Image Path']
                    if os.path.exists(image_path):
                        exif_metadata = extract_exif_metadata(image_path)
                        if exif_metadata:
                            exif_results.append(exif_metadata)
                        else:
                            exif_results.append({"error": "No EXIF metadata found"})
                    else:
                        exif_results.append({"error": "Image file not found"})
                else:
                    exif_results.append({"error": "No image path provided"})

            # Save EXIF results to a DataFrame
            exif_df = pd.DataFrame(exif_results)
            exif_csv_path = os.path.join(tempfile.gettempdir(), 'exif_metadata.csv')
            exif_df.to_csv(exif_csv_path, index=False)

        # Generate PDF Report
        logging.debug("Generating PDF report...")
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'Forensic Analysis Report', 0, 1, 'C')

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Forensic Analysis Report', 0, 1, 'C')
        pdf.ln(10)

        # Add content to PDF
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Summary of Findings', 0, 1)
        pdf.set_font('Arial', '', 12)
        if not exif_analysis:
            pdf.multi_cell(0, 10, f"""
            This report summarizes the forensic analysis of the provided dataset. Key findings include:
            - **Suspicious Activity Detected**: {combined_df['suspicious_activity'].sum()} rows flagged.
            - **Deleted Files Detected**: {combined_df['deleted_files'].sum()} rows flagged.
            - **Sentiment Analysis**: Overall sentiment distribution is shown below.
            - **Topic Modeling**: Key topics identified in the text data.
            """)
            pdf.ln(10)

            # Add Word Cloud
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Word Cloud', 0, 1)
            pdf.image(wordcloud_path, x=10, y=None, w=180)
            pdf.ln(10)

            # Add Sentiment Distribution
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Sentiment Distribution', 0, 1)
            pdf.image(sentiment_path, x=10, y=None, w=180)
            pdf.ln(10)

            # Add Suspicious Activity Plot
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Suspicious Activity Detection', 0, 1)
            pdf.image(suspicious_activity_path, x=10, y=None, w=180)
            pdf.ln(10)

            # Add Deleted Files Plot
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Deleted Files Detection', 0, 1)
            pdf.image(deleted_files_path, x=10, y=None, w=180)
            pdf.ln(10)

            # Add Topics
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Key Topics Identified', 0, 1)
            pdf.set_font('Arial', '', 12)
            for i, topic in enumerate(topics):
                pdf.multi_cell(0, 10, f"Topic {i}: {topic}")
            pdf.ln(10)
        else:
            pdf.multi_cell(0, 10, """
            This report summarizes the forensic analysis of the provided dataset. Key findings include:
            - **EXIF Metadata Analysis**: Extracted metadata from image files.
            """)
            pdf.ln(10)

            # Add EXIF Metadata
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'EXIF Metadata', 0, 1)
            pdf.set_font('Arial', '', 12)
            for index, row in exif_df.iterrows():
                pdf.multi_cell(0, 10, f"Image {index + 1}: {row.to_string()}")
            pdf.ln(10)

        # Save PDF
        pdf_output_path = os.path.join(tempfile.gettempdir(), 'Forensic_Analysis_Report.pdf')
        pdf.output(pdf_output_path)

        # Clean up temporary files
        logging.debug("Cleaning up temporary files...")
        if not exif_analysis:
            os.remove(wordcloud_path)
            os.remove(sentiment_path)
            os.remove(suspicious_activity_path)
            os.remove(deleted_files_path)
        else:
            os.remove(exif_csv_path)

        logging.debug("Report generated successfully.")
        return send_file(pdf_output_path, as_attachment=True)

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)