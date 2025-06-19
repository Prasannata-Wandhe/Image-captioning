import os
import base64
import json
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import logging
import torch
import torchvision.transforms as transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class FreeImageCaptioner:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.current_model = None
        
    def load_blip_model(self):
        """Load BLIP model from Hugging Face (Free)"""
        try:
            logger.info("Loading BLIP model...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model.to(self.device)
            self.current_model = "BLIP"
            self.model_loaded = True
            logger.info(f"BLIP model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error loading BLIP model: {str(e)}")
            return False
    
    def load_vit_gpt2_model(self):
        """Load ViT-GPT2 model from Hugging Face (Free alternative)"""
        try:
            logger.info("Loading ViT-GPT2 model...")
            self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.model.to(self.device)
            self.current_model = "ViT-GPT2"
            self.model_loaded = True
            logger.info(f"ViT-GPT2 model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error loading ViT-GPT2 model: {str(e)}")
            return False
    
    def generate_caption_blip(self, image, max_length=50):
        """Generate caption using BLIP model"""
        try:
            # Process image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Generate multiple captions with different parameters
            captions = [caption]
            
            # Try with different beam sizes for variety
            try:
                with torch.no_grad():
                    out2 = self.model.generate(**inputs, max_length=max_length, num_beams=3, temperature=0.8, do_sample=True)
                    caption2 = self.processor.decode(out2[0], skip_special_tokens=True)
                    if caption2 != caption:
                        captions.append(caption2)
            except:
                pass
            
            return {
                'success': True,
                'captions': captions,
                'primary_caption': captions[0],
                'model_used': 'BLIP'
            }
            
        except Exception as e:
            logger.error(f"Error generating BLIP caption: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'captions': [],
                'primary_caption': "Error generating caption"
            }
    
    def generate_caption_vit_gpt2(self, image, max_length=16):
        """Generate caption using ViT-GPT2 model"""
        try:
            # Process image
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(pixel_values, max_length=max_length, num_beams=4, early_stopping=True)
            
            # Decode caption
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Generate alternative captions
            captions = [caption]
            
            try:
                with torch.no_grad():
                    output_ids2 = self.model.generate(pixel_values, max_length=max_length, num_beams=3, temperature=0.7, do_sample=True)
                    caption2 = self.tokenizer.decode(output_ids2[0], skip_special_tokens=True)
                    if caption2 != caption:
                        captions.append(caption2)
            except:
                pass
            
            return {
                'success': True,
                'captions': captions,
                'primary_caption': captions[0],
                'model_used': 'ViT-GPT2'
            }
            
        except Exception as e:
            logger.error(f"Error generating ViT-GPT2 caption: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'captions': [],
                'primary_caption': "Error generating caption"
            }
    
    def generate_caption_replicate_api(self, image):
        """Use Replicate API (Free tier available)"""
        try:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Replicate API call (requires API key - but has free tier)
            # You need to sign up at replicate.com and get API key
            api_key = os.getenv('REPLICATE_API_TOKEN')
            if not api_key:
                return {
                    'success': False,
                    'error': 'Replicate API key not configured',
                    'captions': [],
                    'primary_caption': "API key required"
                }
            
            headers = {
                'Authorization': f'Token {api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "version": "2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
                "input": {
                    "image": f"data:image/jpeg;base64,{img_str}",
                    "task": "image_captioning"
                }
            }
            
            response = requests.post('https://api.replicate.com/v1/predictions', 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 201:
                prediction = response.json()
                # Note: This is simplified - you'd need to poll for completion
                return {
                    'success': True,
                    'captions': ["Caption generation in progress"],
                    'primary_caption': "Please wait for processing...",
                    'model_used': 'Replicate'
                }
            else:
                return {
                    'success': False,
                    'error': f'API error: {response.status_code}',
                    'captions': [],
                    'primary_caption': "API error"
                }
                
        except Exception as e:
            logger.error(f"Error with Replicate API: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'captions': [],
                'primary_caption': "API error"
            }
    
    def generate_caption_huggingface_api(self, image):
        """Use Hugging Face Inference API (Free tier available)"""
        try:
            # Convert PIL image to bytes
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_bytes = buffered.getvalue()
            
            # Hugging Face API
            api_key = os.getenv('HUGGINGFACE_API_TOKEN')
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            # Try multiple models
            models = [
                "Salesforce/blip-image-captioning-base",
                "nlpconnect/vit-gpt2-image-captioning"
            ]
            
            for model_name in models:
                try:
                    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
                    response = requests.post(api_url, headers=headers, data=image_bytes, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            caption = result[0].get('generated_text', 'No caption generated')
                            return {
                                'success': True,
                                'captions': [caption],
                                'primary_caption': caption,
                                'model_used': f'HuggingFace-{model_name.split("/")[-1]}'
                            }
                except Exception as e:
                    logger.warning(f"Failed to use model {model_name}: {str(e)}")
                    continue
            
            return {
                'success': False,
                'error': 'All HuggingFace models failed',
                'captions': [],
                'primary_caption': "Service unavailable"
            }
            
        except Exception as e:
            logger.error(f"Error with HuggingFace API: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'captions': [],
                'primary_caption': "API error"
            }
    
    def process_image(self, image_data):
        """Process image data and convert to PIL Image"""
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    # Remove data URL prefix
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None
    
    def generate_caption(self, image_data):
        """Main method to generate captions using available models"""
        try:
            # Process image
            image = self.process_image(image_data)
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to process image',
                    'captions': [],
                    'primary_caption': "Image processing error"
                }
            
            # Try local models first
            if self.model_loaded:
                if self.current_model == "BLIP":
                    return self.generate_caption_blip(image)
                elif self.current_model == "ViT-GPT2":
                    return self.generate_caption_vit_gpt2(image)
            
            # Fallback to API methods
            logger.info("Trying HuggingFace API...")
            result = self.generate_caption_huggingface_api(image)
            if result['success']:
                return result
            
            # If all else fails
            return {
                'success': False,
                'error': 'No captioning service available',
                'captions': [],
                'primary_caption': "Service unavailable"
            }
            
        except Exception as e:
            logger.error(f"Error in generate_caption: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'captions': [],
                'primary_caption': "Error generating caption"
            }

# Initialize the captioner
captioner = FreeImageCaptioner()

# Try to load a local model
logger.info("Attempting to load local models...")
if not captioner.load_blip_model():
    logger.info("BLIP model failed, trying ViT-GPT2...")
    if not captioner.load_vit_gpt2_model():
        logger.warning("No local models loaded, will use API fallbacks")
    else:
        logger.info("ViT-GPT2 model loaded successfully")
else:
    logger.info("BLIP model loaded successfully")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': captioner.model_loaded,
        'current_model': captioner.current_model,
        'device': captioner.device
    })

@app.route('/caption', methods=['POST'])
def generate_caption():
    """Generate caption for uploaded image"""
    try:
        # Check if image is in request
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            # Read image data
            image_data = file.read()
            
        # Handle base64 image data
        elif 'image_data' in request.json:
            image_data = request.json['image_data']
        
        # Generate caption
        result = captioner.generate_caption(image_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in caption generation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/caption-batch', methods=['POST'])
def generate_captions_batch():
    """Generate captions for multiple images"""
    try:
        images = request.files.getlist('images')
        if not images:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        results = []
        for i, image_file in enumerate(images):
            try:
                image_data = image_file.read()
                result = captioner.generate_caption(image_data)
                results.append({
                    'image_index': i,
                    'filename': image_file.filename,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'image_index': i,
                    'filename': image_file.filename,
                    'result': {
                        'success': False,
                        'error': str(e)
                    }
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in batch caption generation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/switch-model/<model_name>', methods=['POST'])
def switch_model(model_name):
    """Switch between different models"""
    try:
        if model_name.lower() == 'blip':
            success = captioner.load_blip_model()
        elif model_name.lower() == 'vit-gpt2':
            success = captioner.load_vit_gpt2_model()
        else:
            return jsonify({
                'success': False,
                'error': 'Unknown model name'
            }), 400
        
        return jsonify({
            'success': success,
            'current_model': captioner.current_model if success else None,
            'message': f'Switched to {model_name}' if success else f'Failed to load {model_name}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
