import os
from google.adk.agents import Agent
import requests
import json
from dotenv import load_dotenv
import os
import logging
from typing import Optional, Dict, Any
from transformers import pipeline
import scipy.io.wavfile
import ffmpeg
import numpy as np

# import replicate
# import tempfile
from typing import Optional, Dict, Any, Union, BinaryIO
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Building a tool which fetches background music from user's query agent through Freesound API
def get_background_music(query: str) -> Optional[Dict[str, Any]]:
    """
    This function fetches background music from the Freesound API based on the user's query.
    
    Args:
        query: The search query for finding background music the user is interested in the query should be 1-2 words in short.
        
    Returns:
        A dictionary containing audio information or None if no results found
    """
    # Get API key from environment variable
    api_key = os.environ.get("FREESOUND_API_KEY")
    
    if not api_key:
        print("Error: FREESOUND_API_KEY environment variable not set")
        return None
    
    # Freesound API endpoint for searching sounds
    search_url = "https://freesound.org/apiv2/search/text/"
    
    # Parameters for the API request
    params = {
        "query": query,
        "filter": "duration:[1 TO 120]", # Limit to 1-120 seconds
        "fields": "id,name,previews,duration,license,username,url",
        "page_size": 5,
        "sort": "rating_desc", # Sort by highest rated
        "token": api_key
    }
    
    try:
        # Make the API request
        response = requests.get(search_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        results = response.json()
        
        # Check if we have any results
        if results["count"] == 0:
            print(f"No results found for query: {query}")
            return None
        
        # Get the first (highest rated) result
        sound = results["results"][0]
        
        # Return relevant information about the sound
        return {
            "id": sound["id"],
            "name": sound["name"],
            "preview_url": sound["previews"]["preview-hq-mp3"],
            "duration": sound["duration"],
            "license": sound["license"],
            "username": sound["username"],
            "url": sound["url"]
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching background music: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error processing API response: {e}")
        return None

# Generate music based on user query from Meta's MusicGen through Replicate API
def generate_music_musicgen(
    music_prompt: str,
    output_path: str = os.path.join(os.getcwd(), "background_music.wav"),
    use_blank: bool = False
) -> Optional[str]:
    """
    Generate background music using MusicGen or produce silence if use_blank=True.

    Args:
        music_prompt: Text prompt describing the desired music.
        output_path: Path to save the generated audio file.
        use_blank: If True, generates a silent audio file.

    Returns:
        Path to the generated audio file, or None if generation fails.
    """
    if use_blank:
        try:
            print("🎵 Generating silent placeholder music...")
            duration_seconds = 10  # Default silent duration
            silent_audio_path = output_path

            # Create 60 seconds of silence at 16kHz
            ffmpeg.input(
                "anullsrc=r=16000:cl=mono", f="lavfi", t=duration_seconds
            ).output(silent_audio_path, acodec="pcm_s16le").run(overwrite_output=True)

            print(f"✅ Silent music saved to {silent_audio_path}")
            return silent_audio_path
        except Exception as e:
            print(f"❌ Failed to generate silent background music: {e}")
            return None

    try:
        print("🎵 Generating background music...")
        synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")
        music = synthesiser(
            music_prompt,
            forward_params={"do_sample": True}
        )

        # Ensure the audio data is in the correct format
        audio_data = music["audio"]
        sampling_rate = music["sampling_rate"]

        # Convert float32 audio data to int16
        if audio_data.dtype != np.int16:
            max_int16 = np.iinfo(np.int16).max
            audio_data = (audio_data * max_int16).astype(np.int16)

        scipy.io.wavfile.write(
            output_path,
            rate=sampling_rate,
            data=audio_data
        )
        print(f"✅ Background music saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Failed to generate background music: {e}")
        return None


# def generate_music_musicgen(query: str) -> Optional[Dict[str, Any]]:
#     """
#     This function generates music using Meta's MusicGen via the Replicate API based on the user's query.
    
#     Args:
#         query: detailed long text prompt for generating music. The genre, mood, instruments, and any specific elements you want to include.
        
#     Returns:
#         A dictionary containing the name and URL of the generated music, or None if generation fails.
#     """
#     # Get Replicate API token from environment variable
#     replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
#     if not replicate_api_token:
#         logger.error("Error: REPLICATE_API_TOKEN environment variable not set")
#         return None
    
#     try:
#         # Define input parameters
#         input_params = {
#             "prompt": query,
#             "model_version": "stereo-large",
#             "output_format": "mp3",
#             "normalization_strategy": "peak"
#         }
        
#         # Call the Replicate API with the correct model identifier
#         output = replicate.run(
#             "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
#             input=input_params
#         )
#         logger.info(f"Replicate API response: {output}")
        
#         # Check if output is a dictionary and contains the generated audio URL
#         if isinstance(output, dict) and "output" in output:
#             generated_url = output["output"]
#             return {
#                 "name": f"Generated music for '{query}'",
#                 "url": generated_url
#             }
#         else:
#             logger.warning("No output received from MusicGen")
#             return None
        
#     except Exception as e:
#         logger.error(f"Error generating music: {e}")
#         return None


#Agent
root_agent = Agent(
    name="music_agent",
    model="gemini-2.0-flash",
    description=("A multi-tool music agent that can fetch background music based on user query."),
    instruction=(
    """
You are MusicAgent—a background music selector.
1. Always use the get_background_music tool when the user asks for music at first.
2. Construct the query in 1–2 words (e.g., 'soft piano', 'upbeat jazz').
3. Return response to user with name of the audio and url which user ca click to access.
4. Do not include any extra text.
5. If no music is found, or if user is not happy with the music, generate the music using the generate_music_musicgen tool.
6. The prompt for generate_music_musicgen should be a long, detailed and creative prompt with For example, tell me the genre, mood, instruments, and any specific elements you want to include.
6. Even then the user is not happpy, ask for a better prompt and use the generate_music_musicgen tool again.
"""),
    tools=[get_background_music, generate_music_musicgen],
)

# # Test the function if this file is run directly
# if __name__ == "__main__":
#     test_query = "Generate a high-energy rock track with a victorious mood, featuring electric guitar riffs, powerful drums, and a chant-like crowd effect, perfect for a football match montage."
#     result = generate_music_musicgen(test_query)
#     if result:
#         print(result)
#     else:
#         print("No sound found")


  