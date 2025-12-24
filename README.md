# Facial Recognition System

An enhanced people detection system with facial recognition capabilities to identify specific individuals. Built on the foundation of the webcamera main branch with advanced identification features.

⚠️ **This branch is currently in development**

## Features

### Implemented
- Live webcam feed processing
- Video file upload and processing
- Real-time people counting (displayed at top of screen)
- Bounding box detection around detected individuals
- Person identification with labels
- Custom training dataset support
- Reverse image search integration via SerpAPI

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- Ultralytics YOLO
- PIL (Pillow)
- NumPy
- face_recognition
- requests
- google-search-results (SerpAPI)
- beautifulsoup4
- lxml

## Installation
```bash
pip install opencv-python ultralytics pillow numpy face_recognition requests google-search-results beautifulsoup4 lxml
```

## SerpAPI Setup
This system uses SerpAPI for reverse image search capabilities:

1. Sign up for a free SerpAPI account at https://serpapi.com
2. Get your API key (free tier includes 100 searches/month)
3. Set your API key as an environment variable:
```bash
   export SERPAPI_KEY="your_api_key_here"
```
   Or add it directly in the script configuration

## Training Data
This system uses a custom dataset of 10-25 images per person from approximately 30 consenting participants. The model is trained to recognize specific individuals who have provided explicit permission.

### Adding New People
To add someone to the recognition database:
1. Obtain explicit consent from the individual
2. Collect 10-25 photos with variety (lighting, angles, expressions, with/without glasses)
3. Organize images in the designated training folder
4. Run the training process to update the model

## Usage

**Keyboard Controls:**
- Press `i` to identify a detected face
- Press `q` to quit the application

## Current Development Focus
- Improving recognition accuracy across different conditions
- Optimizing model performance
- Enhancing detection reliability
- Testing with real-world vlog footage

## Privacy & Ethics - READ CAREFULLY
**CRITICAL LIMITATIONS AND LEGAL REQUIREMENTS:**

### What This System Should Be Used For:
- Identifying consenting participants from your training dataset
- Processing video footage where all individuals have given explicit permission
- Personal projects with friends/family who have agreed to participate

### What This System Should NOT Be Used For:
- **Do NOT** use on strangers or people who haven't consented
- **Do NOT** scrape social media profiles (Instagram, Facebook, LinkedIn) without explicit permission, even if profiles are public - this violates Terms of Service and privacy laws
- **Do NOT** use for surveillance, tracking, or monitoring without consent
- **Do NOT** use reverse image search to identify people who haven't agreed to be identified

### Legal Compliance:
- Only use images from consenting participants
- Respect the platform Terms of Service - scraping is prohibited
- Comply with GDPR (Europe), BIPA (Illinois), CCPA (California), and other biometric privacy laws
- Participants can withdraw consent and have their data removed at any time
- Keep all facial data secure and encrypted

### Ethical Usage:
Even with the technical capability to search for faces online, doing so without consent is an invasion of privacy. This tool is designed for **consensual identification only** within your approved training dataset.

**If you plan to use reverse image search features, ensure you have explicit written permission from individuals to search for their images online.**

## Technical Details
**Author:** C. Linjewile

The system combines:
- YOLO for person detection
- Face recognition algorithms for identification
- Optional reverse image search via SerpAPI (use only with explicit consent)

## Project Status
This branch is under active development. Features and accuracy are being improved regularly.
