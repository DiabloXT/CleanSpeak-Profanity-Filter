import uuid
import zipfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class TextProfanityFilter:
    def __init__(self):
        try:
            self.loaded_model = joblib.load("model/text_profanity_filter_rfc_model.joblib")
            self.vectorizer = joblib.load("model/tfidf_embeddings.joblib")

            genai.configure(api_key=st.secrets["gemini_api_key"])
            self.model = genai.GenerativeModel("gemini-1.5-pro")
            
        except Exception as err:
            st.exception(err)

    def detect_toxicity(self, text):
        text_tfidf = self.vectorizer.transform([text])
        pred_proba = self.loaded_model.predict_proba(text_tfidf)

        toxicity_likelihood = [pred_proba[i][0][1] for i in range(6)]
        toxicity_scores = {
            "toxic": toxicity_likelihood[0],
            "severely toxic": toxicity_likelihood[1],
            "obscene": toxicity_likelihood[2],
            "threatening": toxicity_likelihood[3],
            "insulting": toxicity_likelihood[4],
            "discriminatory": toxicity_likelihood[5],
        }

        print(toxicity_scores)

        detected_toxicity = []

        for profanity_type, threshold in toxicity_scores.items():
            if threshold >= 0.4:
                detected_toxicity.append(profanity_type)

        if "severely toxic" in detected_toxicity:
            detected_toxicity.remove("toxic")

        if len(detected_toxicity) > 0:
            safety_configurations = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            prompt = f"""
            The following text has been flagged for profanity.
            Input Text: "{text}"

            Please provide an alternative message that conveys the same essence as the original text but without using any profane language.
            IMPORTANT: Regardless of the level of profanity, always return a single, clear alternative message.

            Return the output in the following format: ['output']
            """

            response = self.model.generate_content(
                prompt,
                safety_settings=safety_configurations,
            )

            return response.text, detected_toxicity

        else:
            return False, toxicity_scores


class ImageProfanityFilter:
    def __init__(self):
        genai.configure(api_key=st.secrets["gemini_api_key"])
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def detect_toxicity(self, input_image_path):
        input_image_file = genai.upload_file(
            path=input_image_path,
            display_name=f"image_input_{uuid.uuid4()}",
        )

        print(
            f"Uploaded file '{input_image_file.display_name}' as: {input_image_file.uri}"
        )

        safety_configurations = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        prompt = """"
        Check if this image contains any profane content. (True or False)
        Return the result using this JSON schema:
            Result = {"profanity_test_result": bool}
        """

        response = self.model.generate_content(
            [
                input_image_file,
                prompt,
            ],
            safety_settings=safety_configurations,
        )

        if "true" in response.text.lower():
            print("Found profane content in this image")

            prompt = """"
            The provided image contains profane content.
            Return a bounding box for the part of the image with profanity. 
            If there are multiple boxes, return co-ordinates for the most profane region.

            IMPORTANT: Return the coordinates as a python list in the following format:
            [ymin, xmin, ymax, xmax]
            """

            response = self.model.generate_content(
                [
                    input_image_file,
                    prompt,
                ],
                safety_settings=safety_configurations,
            )

            genai.delete_file(input_image_file.name)
            return response.text

        else:
            genai.delete_file(input_image_file.name)
            return False


class AudioProfanityFilter:
    def __init__(self):
        genai.configure(api_key=st.secrets["gemini_api_key"])
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def detect_toxicity(self, input_audio_path):
        input_audio_file = genai.upload_file(
            path=input_audio_path,
            display_name=f"audio_input_{uuid.uuid4()}",
        )

        print(
            f"Uploaded file '{input_audio_file.display_name}' as: {input_audio_file.uri}"
        )

        safety_configurations = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        prompt = """"
        Check if this audio contains any profane content. (True or False)
        Return the result using this JSON schema:
            Result = {"profanity_test_result": bool}
        """

        response = self.model.generate_content(
            [
                input_audio_file,
                prompt,
            ],
            safety_settings=safety_configurations,
        )

        if "true" in response.text.lower():
            prompt = """"
            The provided audio contains profane content.
            Your task is to return an alternative text for this audio without any profanity.
            
            IMPORTANT: No matter how profane the audio is, you must always provide a suitable alternative.
            IMPORTANT: The length of the alternative must be similar to the provided profane content.

            Example:
            Profane audio: "You are a piece of ****!"
            Alternative: "You are not very nice!"
            
            Now, generate the alternative text for the given profane audio:
            """

            response = self.model.generate_content(
                [
                    input_audio_file,
                    prompt,
                ],
                safety_settings=safety_configurations,
            )

            cleaned_audio_script = response.text.replace("\n", "").replace("`", "")

            genai.delete_file(input_audio_file.name)
            return response.text

        else:
            genai.delete_file(input_audio_file.name)
            return False


class VideoProfanityFilter:
    def __init__(self):
        genai.configure(api_key=st.secrets["gemini_api_key"])
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def detect_toxicity(self, input_video_path):
        input_video_file = genai.upload_file(
            path=input_video_path,
            display_name=f"video_input_{uuid.uuid4()}",
        )

        print(
            f"Uploaded file '{input_video_file.display_name}' as: {input_video_file.uri}"
        )

        safety_configurations = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        prompt = """
        Check if this video contains any profane content. (True or False)
        Return the result using this JSON schema:
            Result = {"profanity_test_result": bool}
        """

        response = self.model.generate_content(
            [
                input_video_file,
                prompt,
            ],
            safety_settings=safety_configurations,
        )

        if "true" in response.text.lower():
            prompt = """
            The provided video contains profane content.

            Your task is to identify and return the start and end timestamps where the profane content is present in the video.

            Instructions:
            1. Return the result as a list of two strings.
            2. The first element represents the start timestamp in MM:SS format.
            3. The second element represents the end timestamp in MM:SS format.

            Example output: ["00:03", "00:12"]

            IMPORTANT: The output must be returned in the exact list format as shown in the example.
            """
            response = self.model.generate_content(
                [
                    input_video_file,
                    prompt,
                ],
                safety_settings=safety_configurations,
            )

            genai.delete_file(input_video_file.name)

            return response.text

        else:
            genai.delete_file(input_video_file.name)
            return False


if __name__ == "__main__":
    image_profanity_filter = ImageProfanityFilter()
    is_profane = image_profanity_filter.detect_toxicity("samples/sample_profane.JPG")
    print(is_profane)

    audio_profanity_filter = AudioProfanityFilter()
    is_profane = audio_profanity_filter.detect_toxicity(
        "samples/sample_profane_audio.mp3"
    )
    print(is_profane)

    video_profanity_filter = VideoProfanityFilter()
    is_profane = video_profanity_filter.detect_toxicity(
        "samples/sample_profane_video.mp4"
    )
    print(is_profane)

    text_profanity_filter = TextProfanityFilter()
    is_profane = text_profanity_filter.detect_toxicity(
        "The moment I see you, I am gonna kick your dumb ass out of this city nigga"
    )
    print(is_profane)
