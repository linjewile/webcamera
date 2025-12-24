"""
Webcamera with Face Recognition using Reverse Image Search + Social Media Scraping
Author: C. Linjewile
Description: Detects people using YOLO, detects faces, and identifies them
             using Google Lens reverse image search via SerpAPI + social media scraping.

Instructions:
1. Install required packages:
   pip install opencv-python ultralytics face_recognition pillow numpy requests google-search-results beautifulsoup4 lxml

2. Get a SerpAPI key from https://serpapi.com (free tier: 100 searches/month)

3. Set your API key below or as environment variable SERPAPI_KEY

4. Run the script and press 'i' to identify a face, 'q' to quit
"""

import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import tempfile
import requests
import base64
import threading
import time
import re
import json
from queue import Queue
from urllib.parse import quote_plus, urljoin

# Try to import BeautifulSoup for web scraping
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed. Social media scraping disabled.")
    print("Install with: pip install beautifulsoup4 lxml")

# ============================================
# CONFIGURATION - SET YOUR API KEY HERE
# ============================================
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "13a2cf69c1f9b4861256f29d4cb48c7572bd990a02da3c9fae62ac0444c89c0c")

# Try to import face_recognition (optional - falls back to YOLO face detection)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not installed. Using alternative face detection.")
    print("Install with: pip install face_recognition")

# Try to import MediaPipe for face detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe available for face detection")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Info: MediaPipe not installed. Install with: pip install mediapipe")

# Try to import MTCNN for face detection
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
    print("✓ MTCNN available for face detection")
except ImportError:
    MTCNN_AVAILABLE = False
    print("Info: MTCNN not installed. Install with: pip install mtcnn tensorflow")

# Try to import RetinaFace for face detection
try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
    print("✓ RetinaFace available for face detection")
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("Info: RetinaFace not installed. Install with: pip install retina-face")

# Try to import DeepFace for AI-powered emotion detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✓ DeepFace available for AI emotion detection")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Info: DeepFace not installed. Install with: pip install deepface")

# Try to import SerpAPI
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    print("Warning: google-search-results not installed.")
    print("Install with: pip install google-search-results")


class DeepFaceRecognizer:
    """AI-powered face recognition using DeepFace with local database."""
    
    def __init__(self, db_path="face_database"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        print(f"Face database initialized at: {db_path}")
    
    def add_face(self, face_image, person_name):
        """Add a face to the recognition database."""
        if not DEEPFACE_AVAILABLE:
            print("DeepFace not available. Cannot add face to database.")
            return False
        
        try:
            # Create person directory
            person_dir = os.path.join(self.db_path, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save face image
            timestamp = int(time.time())
            face_path = os.path.join(person_dir, f"{person_name}_{timestamp}.jpg")
            cv2.imwrite(face_path, face_image)
            
            print(f"✓ Added {person_name} to face database: {face_path}")
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
    
    def recognize_face(self, face_image):
        """Recognize a face using DeepFace AI model."""
        if not DEEPFACE_AVAILABLE:
            return {"status": "error", "message": "DeepFace not available"}
        
        # Check if database has any faces
        if not os.listdir(self.db_path):
            return {"status": "empty_db", "message": "No faces in database. Press 'a' to add faces."}
        
        try:
            # Save temp image for DeepFace
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, face_image)
                temp_path = tmp.name
            
            # Use DeepFace to find matches
            print("\n🔍 Searching face database with AI...")
            results = DeepFace.find(
                img_path=temp_path,
                db_path=self.db_path,
                model_name="VGG-Face",  # Options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib
                distance_metric="cosine",  # Options: cosine, euclidean, euclidean_l2
                enforce_detection=False,
                silent=True
            )
            
            os.unlink(temp_path)
            
            # Parse results
            if isinstance(results, list) and len(results) > 0 and not results[0].empty:
                df = results[0]
                best_match = df.iloc[0]
                
                # Extract person name from path
                identity_path = best_match['identity']
                person_name = os.path.basename(os.path.dirname(identity_path))
                distance = best_match['distance'] if 'distance' in best_match else best_match.get('VGG-Face_cosine', 1.0)
                
                # Calculate confidence (lower distance = higher confidence)
                confidence = max(0, min(100, (1 - distance) * 100))
                
                print(f"✓ Match found: {person_name} (confidence: {confidence:.1f}%)")
                
                return {
                    "status": "match",
                    "name": person_name,
                    "confidence": confidence,
                    "distance": distance
                }
            else:
                print("✗ No match found in database")
                return {
                    "status": "no_match",
                    "message": "Unknown person"
                }
                
        except Exception as e:
            print(f"Recognition error: {e}")
            return {"status": "error", "message": str(e)}
    
    def list_known_faces(self):
        """List all people in the database."""
        people = []
        try:
            for person_dir in os.listdir(self.db_path):
                person_path = os.path.join(self.db_path, person_dir)
                if os.path.isdir(person_path):
                    face_count = len([f for f in os.listdir(person_path) if f.endswith('.jpg')])
                    people.append((person_dir, face_count))
        except:
            pass
        return people


class SocialMediaScraper:
    """Scrapes social media platforms for profile information."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def search_google_for_person(self, image_url):
        """Use Google Image Search to find social media profiles."""
        profiles = []
        
        if not SERPAPI_AVAILABLE:
            return profiles
        
        try:
            # Search Google for the image with social media site filters
            for site in ["linkedin.com", "facebook.com", "twitter.com", "instagram.com"]:
                params = {
                    "engine": "google_reverse_image",
                    "image_url": image_url,
                    "api_key": SERPAPI_KEY,
                }
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                # Look for inline images and image results
                if "inline_images" in results:
                    for img in results["inline_images"][:5]:
                        link = img.get("link", "")
                        if site in link.lower():
                            profiles.append({
                                "platform": site.split(".")[0].title(),
                                "url": link,
                                "title": img.get("title", ""),
                            })
                
                if "image_results" in results:
                    for img in results["image_results"][:5]:
                        link = img.get("link", "")
                        if site in link.lower():
                            profiles.append({
                                "platform": site.split(".")[0].title(),
                                "url": link,
                                "title": img.get("title", ""),
                            })
        except Exception as e:
            print(f"Google search error: {e}")
        
        return profiles
    
    def search_social_by_name(self, name):
        """Search for social media profiles by name using SerpAPI."""
        profiles = []
        
        if not SERPAPI_AVAILABLE or not name or name in ["Unknown", "API Error", "Search Error"]:
            return profiles
        
        try:
            # Search Google for social profiles - LinkedIn, Facebook, Instagram only
            search_queries = [
                f'"{name}" site:linkedin.com/in/',
                f'"{name}" site:facebook.com',
                f'"{name}" site:instagram.com',
            ]
            
            print(f"\n  Searching LinkedIn, Facebook, Instagram for: {name}")
            
            for query in search_queries:
                params = {
                    "engine": "google",
                    "q": query,
                    "api_key": SERPAPI_KEY,
                    "num": 5,
                }
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                if "organic_results" in results:
                    for result in results["organic_results"][:3]:
                        link = result.get("link", "")
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                        
                        # Determine platform
                        platform = "Unknown"
                        if "linkedin.com" in link:
                            platform = "LinkedIn"
                        elif "facebook.com" in link:
                            platform = "Facebook"
                        elif "instagram.com" in link:
                            platform = "Instagram"
                        
                        if platform != "Unknown":
                            profile_data = {
                                "platform": platform,
                                "url": link,
                                "title": title,
                                "snippet": snippet[:100] if snippet else "",
                            }
                            
                            # Auto-scrape Instagram profile pictures
                            if platform == "Instagram":
                                print(f"  📷 Scraping Instagram profile picture...")
                                pic_data = self.scrape_instagram_profile_pic(link)
                                if pic_data:
                                    profile_data["profile_pic_url"] = pic_data["profile_pic_url"]
                                    profile_data["username"] = pic_data["username"]
                            
                            # Auto-scrape Facebook profile pictures
                            elif platform == "Facebook":
                                print(f"  📷 Scraping Facebook profile picture...")
                                pic_data = self.scrape_facebook_profile_pic(link)
                                if pic_data:
                                    profile_data["profile_pic_url"] = pic_data["profile_pic_url"]
                            
                            profiles.append(profile_data)
                
                # Rate limiting - be respectful
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Social search error: {e}")
        
        return profiles
    
    def scrape_linkedin_public(self, url):
        """Scrape public LinkedIn profile information."""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200 and BS4_AVAILABLE:
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Try to extract name from meta tags or title
                name = None
                
                # Check og:title
                og_title = soup.find('meta', property='og:title')
                if og_title:
                    name = og_title.get('content', '').split('|')[0].split('-')[0].strip()
                
                # Check title tag
                if not name:
                    title = soup.find('title')
                    if title:
                        name = title.text.split('|')[0].split('-')[0].strip()
                
                return {
                    "name": name,
                    "platform": "LinkedIn",
                    "url": url,
                }
        except Exception as e:
            print(f"LinkedIn scrape error: {e}")
        
        return None
    
    def scrape_twitter_public(self, url):
        """Scrape public Twitter/X profile information."""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200 and BS4_AVAILABLE:
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Try to extract name from meta tags
                name = None
                
                og_title = soup.find('meta', property='og:title')
                if og_title:
                    content = og_title.get('content', '')
                    # Twitter format: "Name (@username)"
                    match = re.match(r'^(.+?)\s*\(@', content)
                    if match:
                        name = match.group(1).strip()
                
                return {
                    "name": name,
                    "platform": "Twitter/X",
                    "url": url,
                }
        except Exception as e:
            print(f"Twitter scrape error: {e}")
        
        return None
    
    def scrape_instagram_profile_pic(self, username_or_url):
        """Scrape Instagram profile picture (always public, even for private accounts)."""
        try:
            # Extract username from URL if needed
            username = username_or_url
            if "instagram.com" in username_or_url:
                import re
                match = re.search(r'instagram\.com/([^/\?]+)', username_or_url)
                if match:
                    username = match.group(1)
            
            # Instagram profile pictures are always public via their API-like endpoint
            # Method 1: Try the public profile page
            profile_url = f"https://www.instagram.com/{username}/"
            response = self.session.get(profile_url, timeout=10)
            
            if response.status_code == 200:
                # Instagram embeds profile data in the page
                import re
                import json
                
                # Look for the profile picture in the shared data
                match = re.search(r'"profile_pic_url":"([^"]+)"', response.text)
                if match:
                    pic_url = match.group(1).replace(r'\u0026', '&')
                    print(f"  ✓ Found Instagram profile picture for @{username}")
                    return {
                        "username": username,
                        "profile_pic_url": pic_url,
                        "profile_url": profile_url,
                        "platform": "Instagram"
                    }
                
                # Alternative method: Look in meta tags
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(response.text, 'lxml')
                    og_image = soup.find('meta', property='og:image')
                    if og_image:
                        pic_url = og_image.get('content', '')
                        if pic_url:
                            print(f"  ✓ Found Instagram profile picture (from meta) for @{username}")
                            return {
                                "username": username,
                                "profile_pic_url": pic_url,
                                "profile_url": profile_url,
                                "platform": "Instagram"
                            }
                
        except Exception as e:
            print(f"Instagram scrape error for {username_or_url}: {e}")
        
        return None
    
    def scrape_facebook_profile_pic(self, profile_url):
        """Scrape Facebook profile picture (public profiles only)."""
        try:
            response = self.session.get(profile_url, timeout=10)
            if response.status_code == 200 and BS4_AVAILABLE:
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Look for profile picture in meta tags
                og_image = soup.find('meta', property='og:image')
                if og_image:
                    pic_url = og_image.get('content', '')
                    if pic_url:
                        print(f"  ✓ Found Facebook profile picture")
                        return {
                            "profile_pic_url": pic_url,
                            "profile_url": profile_url,
                            "platform": "Facebook"
                        }
        except Exception as e:
            print(f"Facebook scrape error: {e}")
        
        return None
    
    def download_profile_picture(self, pic_url, save_path):
        """Download a profile picture and save it locally."""
        try:
            response = self.session.get(pic_url, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Saved profile picture to: {save_path}")
                return True
        except Exception as e:
            print(f"Download error: {e}")
        return False


class FaceIdentifier:
    """Handles face identification using reverse image search."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.identified_faces = {}  # Cache: face_encoding -> name
        self.search_queue = Queue()
        self.results_queue = Queue()
        self.is_searching = False
        self.social_scraper = SocialMediaScraper()
        self.found_profiles = {}  # Store found social profiles
        
        # Start background search thread
        self.search_thread = threading.Thread(target=self._search_worker, daemon=True)
        self.search_thread.start()
    
    def _search_worker(self):
        """Background thread for processing reverse image searches."""
        while True:
            try:
                face_image, face_id = self.search_queue.get(timeout=1)
                self.is_searching = True
                
                # Enhance the face image before search
                print(f"\n{'='*50}")
                print(f"Analyzing and enhancing face {face_id}...")
                enhanced_face = self._enhance_face_image(face_image)
                
                if enhanced_face is None:
                    print("Face quality too low for reliable recognition.")
                    print(f"{'='*50}")
                    self.results_queue.put((face_id, "Poor Quality - Try Again"))
                    self.is_searching = False
                    continue
                
                print(f"{'='*50}")
                
                # Step 1: Reverse image search
                result, image_url = self._reverse_image_search(enhanced_face)
                
                # Step 2: Search social media if we found a name
                profiles = []
                if result and result not in ["Unknown", "API Error", "Search Error", "Upload Failed"]:
                    print(f"\nSearching social media for: {result}")
                    profiles = self.social_scraper.search_social_by_name(result)
                    
                    if profiles:
                        print(f"\n{'='*50}")
                        print("SOCIAL MEDIA PROFILES FOUND:")
                        print("="*50)
                        for p in profiles:
                            print(f"  [{p['platform']}] {p['title']}")
                            print(f"    URL: {p['url']}")
                            if p.get('profile_pic_url'):
                                print(f"    📷 Profile Picture: {p['profile_pic_url'][:100]}...")
                            if p.get('snippet'):
                                print(f"    Info: {p['snippet']}")
                        print("="*50 + "\n")
                        
                        self.found_profiles[face_id] = profiles
                
                self.results_queue.put((face_id, result))
                
                self.is_searching = False
            except:
                continue
    
    def _reverse_image_search(self, face_image):
        """
        Perform reverse image search using SerpAPI Google Lens.
        Returns tuple of (identified name, image_url) or ('Unknown', None).
        """
        if not SERPAPI_AVAILABLE:
            return ("Install: pip install google-search-results", None)
        
        if not self.api_key or len(self.api_key) < 20:
            return ("API Key Required", None)
        
        try:
            # Save face image to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, face_image)
                temp_path = tmp.name
            
            # Upload image to a temporary image hosting service
            image_url = self._upload_image(temp_path)
            
            if not image_url:
                os.unlink(temp_path)
                return ("Upload Failed", None)
            
            print(f"Image uploaded: {image_url[:50]}...")
            
            # Try multiple search approaches for better face matching
            name = "Unknown"
            
            # Approach 1: Google Reverse Image Search (better for faces)
            print("  Trying Google Reverse Image Search...")
            try:
                params = {
                    "engine": "google_reverse_image",
                    "image_url": image_url,
                    "api_key": self.api_key
                }
                search = GoogleSearch(params)
                results = search.get_dict()
                name = self._parse_reverse_image_results(results)
            except Exception as e:
                print(f"  Reverse image search error: {e}")
            
            # Approach 2: If still unknown, try Google Lens
            if name == "Unknown" or "Unknown" in name:
                print("  Trying Google Lens...")
                try:
                    params = {
                        "engine": "google_lens",
                        "url": image_url,
                        "api_key": self.api_key
                    }
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    name = self._parse_search_results(results)
                except Exception as e:
                    print(f"  Google Lens error: {e}")
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return (name, image_url)
            
        except Exception as e:
            print(f"Search error: {e}")
            return ("Search Error", None)
    
    def _parse_reverse_image_results(self, results):
        """Parse Google Reverse Image Search results - better for faces."""
        try:
            print("\n" + "="*50)
            print("GOOGLE REVERSE IMAGE SEARCH RESULTS:")
            print("="*50)
            
            # Check for errors
            if "error" in results:
                print(f"API Error: {results['error']}")
                return "API Error"
            
            # Check inline images (often contains face matches)
            if "inline_images" in results:
                print(f"\nInline Images ({len(results['inline_images'])} found):")
                for i, img in enumerate(results["inline_images"][:20]):  # Check more results
                    source = img.get("source", "")
                    title = img.get("title", "No title")
                    link = img.get("link", "")
                    print(f"  {i+1}. {title[:80]}")
                    print(f"      Source: {source}")
                    print(f"      Link: {link[:100]}")
                    
                    # Check for social media profiles - BE LENIENT
                    link_lower = link.lower()
                    if "linkedin.com" in link_lower or "facebook.com" in link_lower or "instagram.com" in link_lower:
                        # Try multiple parsing strategies
                        name_candidates = []
                        
                        # Strategy 1: Split by common separators
                        for sep in ["-", "|", "@", "·", "•"]:
                            if sep in title:
                                candidate = title.split(sep)[0].strip()
                                if candidate:
                                    name_candidates.append(candidate)
                        
                        # Strategy 2: Take everything before parentheses or brackets
                        import re
                        match = re.match(r'^([^\(\[\{]+)', title)
                        if match:
                            name_candidates.append(match.group(1).strip())
                        
                        # Check each candidate
                        for name in name_candidates:
                            # Basic filters only - don't be too strict
                            if 3 <= len(name) <= 60:
                                name_lower = name.lower()
                                # Skip obvious non-names
                                if not any(bad in name_lower for bad in ["linkedin", "facebook", "instagram", "photo", "image", "stock", "getty", "sign in", "log in", "home page"]):
                                    print(f"\n>>> FOUND LINKEDIN/SOCIAL: {name}")
                                    print(f"    Source: {link}")
                                    return name
            
            # Check image results
            if "image_results" in results:
                print(f"\nImage Results ({len(results['image_results'])} found):")
                for i, img in enumerate(results["image_results"][:20]):  # Check more results
                    title = img.get("title", "No title")
                    source = img.get("source", "")
                    link = img.get("link", "")
                    print(f"  {i+1}. {title[:80]}")
                    print(f"      Link: {link[:100]}")
                    
                    # Check for social media profiles - BE LENIENT
                    link_lower = link.lower()
                    if "linkedin.com" in link_lower or "facebook.com" in link_lower or "instagram.com" in link_lower:
                        # Try multiple parsing strategies
                        name_candidates = []
                        
                        import re
                        # Strategy 1: Split by common separators
                        for sep in ["-", "|", "@", "·", "•"]:
                            if sep in title:
                                candidate = title.split(sep)[0].strip()
                                if candidate:
                                    name_candidates.append(candidate)
                        
                        # Strategy 2: Take everything before parentheses
                        match = re.match(r'^([^\(\[\{]+)', title)
                        if match:
                            name_candidates.append(match.group(1).strip())
                        
                        # Check each candidate
                        for name in name_candidates:
                            if 3 <= len(name) <= 60:
                                name_lower = name.lower()
                                if not any(bad in name_lower for bad in ["linkedin", "facebook", "instagram", "photo", "image", "stock", "getty", "sign in", "log in"]):
                                    print(f"\n>>> FOUND from image results: {name}")
                                    print(f"    Source: {link}")
                                    return name
            
            # Check knowledge graph
            if "knowledge_graph" in results:
                kg = results["knowledge_graph"]
                if "title" in kg:
                    print(f"\nKnowledge Graph: {kg['title']}")
                    return kg["title"]
            
            # Check search information for "best guess"
            if "search_information" in results:
                si = results["search_information"]
                if "query_displayed" in si:
                    guess = si["query_displayed"]
                    print(f"\nGoogle's best guess: {guess}")
                    # Only return if it looks like a name (not an object)
                    if " " in guess and len(guess) < 40:
                        if not any(skip in guess.lower() for skip in ["iphone", "phone", "device", "camera", "photo", "image", "man", "woman", "person", "face"]):
                            return guess
            
            print("\nNo confident face match found.")
            print("="*50 + "\n")
            return "Unknown"
            
        except Exception as e:
            print(f"Parse error: {e}")
            return "Unknown"
    
    def _upload_image(self, image_path):
        """Upload image to free image hosting service."""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Try imgbb first (if key is available)
            imgbb_key = os.environ.get("IMGBB_KEY", "")
            if imgbb_key:
                response = requests.post(
                    "https://api.imgbb.com/1/upload",
                    data={
                        "key": imgbb_key,
                        "image": image_data
                    },
                    timeout=15
                )
                if response.status_code == 200:
                    return response.json()["data"]["url"]
            
            # Try freeimage.host (no API key required)
            try:
                with open(image_path, "rb") as f:
                    files = {"source": f}
                    data = {"type": "file", "action": "upload"}
                    response = requests.post(
                        "https://freeimage.host/api/1/upload?key=6d207e02198a847aa98d0a2a901485a5",
                        files=files,
                        data=data,
                        timeout=15
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("status_code") == 200:
                            return result["image"]["url"]
            except Exception as e:
                print(f"Freeimage.host error: {e}")
            
            # Try catbox.moe (no API key required)
            try:
                with open(image_path, "rb") as f:
                    files = {"fileToUpload": f}
                    data = {"reqtype": "fileupload"}
                    response = requests.post(
                        "https://catbox.moe/user/api.php",
                        files=files,
                        data=data,
                        timeout=15
                    )
                    if response.status_code == 200 and response.text.startswith("http"):
                        return response.text.strip()
            except Exception as e:
                print(f"Catbox error: {e}")
            
            # Try 0x0.st (no API key required)
            try:
                with open(image_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(
                        "https://0x0.st",
                        files=files,
                        timeout=15
                    )
                    if response.status_code == 200:
                        return response.text.strip()
            except Exception as e:
                print(f"0x0.st error: {e}")
            
            return None
            
        except Exception as e:
            print(f"Upload error: {e}")
            return None
    
    def _check_face_quality(self, face_image):
        """Check if face image is good quality for recognition."""
        try:
            h, w = face_image.shape[:2]
            
            # Check 1: Minimum size
            if min(h, w) < 80:
                print("  ⚠ Face too small (min 80px)")
                return False, "Too small"
            
            # Check 2: Facial landmark analysis (PRIORITY)
            print("  🔍 Analyzing facial landmarks and expression...")
            landmark_data = analyze_facial_landmarks(face_image)
            
            if landmark_data:
                print(f"     Landmarks detected: {landmark_data['landmarks_detected']}")
                print(f"     Expression: {landmark_data['expression']}")
                print(f"     Face quality score: {landmark_data['quality_score']}/100")
                print(f"     Face centered: {landmark_data['face_centered']}")
                
                if not landmark_data['is_valid_face']:
                    print(f"  ⚠ Facial landmarks validation failed (score: {landmark_data['quality_score']}/100)")
                    return False, f"Poor facial structure (score: {landmark_data['quality_score']})"
                
                if landmark_data['quality_score'] < 50:
                    print(f"  ⚠ Face quality too low")
                    return False, "Low quality face"
                
                print(f"  ✓ Facial landmarks validated - this is a real human face!")
            else:
                print("  ⚠ Could not detect facial landmarks")
                # Continue with other checks if landmarks fail
            
            # Check 3: Blur detection using Laplacian variance
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:
                print(f"  ⚠ Face too blurry (score: {laplacian_var:.1f}, need >50)")
                return False, f"Too blurry ({laplacian_var:.1f})"
            
            # Check 4: Brightness check
            mean_brightness = np.mean(gray)
            if mean_brightness < 40 or mean_brightness > 220:
                print(f"  ⚠ Poor lighting (brightness: {mean_brightness:.1f})")
                return False, f"Poor lighting ({mean_brightness:.1f})"
            
            print(f"  ✓ Face quality OK (size: {w}x{h}, sharpness: {laplacian_var:.1f}, brightness: {mean_brightness:.1f})")
            return True, "Good quality"
            
        except Exception as e:
            print(f"Quality check error: {e}")
            return True, "Unknown"  # Allow if check fails
    
    def _enhance_face_image(self, face_image):
        """Enhance face image for better recognition."""
        try:
            # Check quality first
            is_good, reason = self._check_face_quality(face_image)
            if not is_good:
                return None  # Signal to skip this face
            
            # Resize to optimal size for face recognition (600px)
            h, w = face_image.shape[:2]
            if max(h, w) != 600:
                scale = 600 / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                face_image = cv2.resize(face_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge back
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Apply moderate sharpening
            kernel = np.array([[-0.5,-1,-0.5], [-1,7,-1], [-0.5,-1,-0.5]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Denoise slightly to reduce artifacts
            denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 5, 5, 7, 21)
            
            return denoised
        except Exception as e:
            print(f"Enhancement error: {e}")
            return face_image
    
    def _parse_search_results(self, results):
        """Parse Google Lens results to extract person name."""
        try:
            # Debug: Print what we got back
            print("\n" + "="*50)
            print("GOOGLE LENS SEARCH RESULTS:")
            print("="*50)
            
            # Check for errors
            if "error" in results:
                print(f"API Error: {results['error']}")
                return "API Error"
            
            # Check knowledge graph first (most reliable for people)
            if "knowledge_graph" in results:
                kg = results["knowledge_graph"]
                print(f"\nKnowledge Graph found:")
                if "title" in kg:
                    print(f"  Title: {kg['title']}")
                    return kg["title"]
                if "name" in kg:
                    print(f"  Name: {kg['name']}")
                    return kg["name"]
            
            # AGGRESSIVE filtering: Skip anything that's clearly not a person's profile
            object_keywords = [
                # Electronics
                "iphone", "phone", "samsung", "android", "device", "camera", "laptop",
                "computer", "tablet", "ipad", "watch", "airpods", "headphones", "speaker",
                "tv", "monitor", "keyboard", "mouse", "case", "charger", "cable", "apple",
                "macbook", "pc", "screen", "display", "electronics", "gadget",
                # Clothing/accessories
                "shirt", "dress", "pants", "shoes", "jacket", "hat", "glasses", "sunglasses",
                "fashion", "style", "outfit", "clothing", "wear", "apparel",
                # Objects
                "car", "vehicle", "bike", "motorcycle", "furniture", "chair", "table",
                "book", "bag", "backpack", "purse", "wallet", "mug", "bottle",
                # Stock photos
                "stock photo", "getty", "shutterstock", "alamy", "depositphotos", "istockphoto",
                "stock image", "royalty free", "creative commons", "license",
                # Generic descriptions (not names)
                "man in", "woman in", "person in", "people", "portrait", "selfie", "photo of",
                "girl in", "boy in", "guy in", "lady in", "male", "female",
                "similar", "related", "more like", "looks like", "resembles", "comparable",
                # UI/Digital
                "wallpaper", "background", "photo frame", "picture frame",
                "app", "application", "software", "website", "webpage", "social media",
                "interface", "ui", "mockup", "template", "design",
                # Commerce
                "buy", "purchase", "shop", "store", "product", "brand", "model", "price",
                "sale", "discount", "deal", "amazon", "ebay", "marketplace", "shipping",
                # Misc non-person indicators
                "how to", "tutorial", "guide", "review", "unboxing", "vs", "comparison",
                "download", "free", "best", "top", "2024", "2025", "new", "latest"
            ]
            
            # Check visual matches - look for LinkedIn, social profiles, etc.
            if "visual_matches" in results:
                print(f"\nVisual Matches ({len(results['visual_matches'])} found):")
                for i, match in enumerate(results["visual_matches"][:20]):  # Check more results
                    title = match.get("title", "No title")
                    source = match.get("source", "")
                    link = match.get("link", "")
                    print(f"  {i+1}. {title[:80]}")
                    print(f"      Source: {source[:60]}")
                    print(f"      Link: {link[:100]}")
                    
                    # Skip if it looks like an object match
                    title_lower = title.lower()
                    source_lower = source.lower()
                    combined_text = f"{title_lower} {source_lower} {link.lower()}"
                    
                    if any(obj in combined_text for obj in object_keywords):
                        print(f"      [SKIPPED - object/product detected]")
                        continue
                    
                    # Check for social media / professional profiles
                    link_lower = link.lower()
                    
                    # PRIORITY: Direct social media profile links - RELAXED validation
                    if "linkedin.com" in link_lower or "facebook.com" in link_lower or "instagram.com" in link_lower:
                        # Try multiple extraction strategies
                        import re
                        name_candidates = []
                        
                        # Strategy 1: Common separators
                        for sep in ["-", "|", "@", "·", "•", ":"]:
                            if sep in title:
                                parts = title.split(sep)
                                for part in parts[:2]:  # Check first 2 parts
                                    candidate = part.strip()
                                    if candidate:
                                        name_candidates.append(candidate)
                        
                        # Strategy 2: Before parentheses/brackets
                        match = re.match(r'^([^\(\[\{]+)', title)
                        if match:
                            name_candidates.append(match.group(1).strip())
                        
                        # Strategy 3: Full title if short enough
                        if len(title) <= 50:
                            name_candidates.append(title.strip())
                        
                        # Check each candidate - LENIENT filtering
                        for name in name_candidates:
                            if 3 <= len(name) <= 60:
                                name_lower = name.lower()
                                # Only skip obvious non-names
                                bad_terms = ["linkedin", "facebook", "instagram", "sign in", "log in", "home", 
                                           "join now", "create account", "privacy", "terms", "cookie", "help center"]
                                if not any(bad in name_lower for bad in bad_terms):
                                    # Skip if ALL CAPS (likely spam/ads)
                                    if not (name.isupper() and len(name) > 10):
                                        print(f"\n>>> FOUND SOCIAL PROFILE: {name}")
                                        print(f"    Platform: {'LinkedIn' if 'linkedin' in link_lower else 'Facebook' if 'facebook' in link_lower else 'Instagram'}")
                                        print(f"    URL: {link[:100]}")
                                        return name
                    
                    # Also check title for profile indicators
                    profile_indicators = ["profile", "linkedin", "facebook", "instagram"]
                    if any(word in title_lower for word in profile_indicators):
                        import re
                        # Extract potential names
                        for sep in ["-", "|", "@", "·"]:
                            if sep in title:
                                candidate = title.split(sep)[0].strip()
                                if 3 <= len(candidate) <= 60:
                                    # Simple check - not too restrictive
                                    if not any(bad in candidate.lower() for bad in ["sign in", "log in", "join", "create"]):
                                        print(f"\n>>> FOUND PROFILE: {candidate}")
                                        print(f"    From: {link[:100]}")
                                        return candidate
            
            # Check reverse image search results
            if "reverse_image_search" in results:
                print(f"\nReverse Image Search results found")
                ris = results["reverse_image_search"]
                if "results" in ris:
                    for r in ris["results"][:5]:
                        print(f"  - {r.get('title', 'No title')}")
            
            # Check text/OCR results
            if "text_results" in results:
                print(f"\nText Results ({len(results['text_results'])} found):")
                for text in results["text_results"][:5]:
                    print(f"  - {text.get('text', 'No text')[:50]}")
            
            # Check lens results
            if "lens_results" in results:
                print(f"\nLens Results found")
            
            # Print all top-level keys for debugging
            print(f"\nAll result keys: {list(results.keys())}")
            
            # FINAL FALLBACK: If nothing found, print ALL LinkedIn/social links detected
            print("\n" + "="*50)
            print("FALLBACK: Searching for ANY LinkedIn/social profiles...")
            print("="*50)
            found_social_links = []
            
            for section in ["visual_matches", "inline_images", "image_results"]:
                if section in results:
                    for item in results[section][:30]:
                        link = item.get("link", "")
                        title = item.get("title", "")
                        if any(social in link.lower() for social in ["linkedin.com", "facebook.com", "instagram.com"]):
                            found_social_links.append({
                                "title": title,
                                "link": link,
                                "section": section
                            })
            
            if found_social_links:
                print(f"\nFound {len(found_social_links)} social media links:")
                for i, item in enumerate(found_social_links[:10], 1):
                    print(f"\n{i}. {item['title'][:100]}")
                    print(f"   Link: {item['link'][:120]}")
                    print(f"   Section: {item['section']}")
                
                # Try to extract name from first LinkedIn link
                for item in found_social_links:
                    if "linkedin.com" in item['link'].lower():
                        import re
                        # Very permissive extraction
                        title = item['title']
                        # Remove common suffixes
                        title = re.sub(r'\s*[\|\-\u2022\u00b7]\s*(LinkedIn|Profile|Facebook|Instagram).*$', '', title, flags=re.IGNORECASE)
                        name = title.strip()
                        if 2 <= len(name) <= 70:
                            print(f"\n>>> FALLBACK MATCH (LinkedIn): {name}")
                            print(f"    From: {item['link'][:100]}")
                            return name
            else:
                print("No social media links found in results")
            
            print("="*50 + "\n")
            
            print("="*50 + "\n")
            return "Unknown - Check console for details"
            
        except Exception as e:
            print(f"Parse error: {e}")
            import traceback
            traceback.print_exc()
            return "Parse Error"
    
    def identify_face(self, face_image, face_id):
        """Queue a face for identification."""
        self.search_queue.put((face_image, face_id))
    
    def get_results(self):
        """Get any completed identification results."""
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        return results
    
    def get_profiles(self, face_id):
        """Get social media profiles for a face."""
        return self.found_profiles.get(face_id, [])


def detect_faces_yolo(frame, model):
    """Detect faces using YOLO (fallback method)."""
    # YOLO doesn't detect faces by default, so we detect persons
    # and estimate face region from upper body
    results = model(frame, verbose=False)
    faces = []
    
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Estimate face as top 1/4 of person bounding box
                face_height = (y2 - y1) // 4
                face_y2 = y1 + face_height
                faces.append((x1, y1, x2, face_y2))
    
    return faces


def detect_faces_mediapipe(frame):
    """Detect faces using MediaPipe (Google's solution - very accurate)."""
    if not MEDIAPIPE_AVAILABLE:
        return []
    
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w = frame.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    faces.append((x1, y1, x2, y2))
            return faces
    except Exception as e:
        print(f"MediaPipe error: {e}")
        return []

def analyze_facial_landmarks(face_img):
    """Analyze facial landmarks to validate it's a real face and detect expression using AI."""
    # Priority 1: Try DeepFace AI model (most accurate)
    if DEEPFACE_AVAILABLE:
        try:
            # DeepFace emotion detection
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
            
            # Extract emotion data
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            confidence = emotions.get(dominant_emotion, 0)
            
            # Map DeepFace emotions to our format
            emotion_map = {
                'angry': 'angry',
                'disgust': 'disgusted',
                'fear': 'scared',
                'happy': 'happy',
                'sad': 'sad',
                'surprise': 'surprised',
                'neutral': 'neutral'
            }
            
            expression = emotion_map.get(dominant_emotion, dominant_emotion)
            quality_score = min(int(confidence), 100)
            
            return {
                "landmarks_detected": 468,  # DeepFace uses internal landmarks
                "expression": expression,
                "quality_score": quality_score,
                "confidence": confidence,
                "all_emotions": emotions,
                "is_valid_face": confidence > 30,
                "detection_method": "DeepFace AI"
            }
        except Exception as e:
            print(f"DeepFace analysis failed: {e}, falling back to MediaPipe...")
    
    # Fallback: MediaPipe landmark-based detection
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)
            
            if not results.multi_face_landmarks:
                return None
            
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_list = face_landmarks.landmark
            
            # Key facial points for validation
            h, w = face_img.shape[:2]
            
            # Eyes (landmarks 33, 263 = left eye, 133, 362 = right eye)
            left_eye = landmarks_list[33]
            right_eye = landmarks_list[263]
            
            # Nose tip (landmark 1)
            nose_tip = landmarks_list[1]
            
            # Mouth corners (landmarks 61, 291)
            mouth_left = landmarks_list[61]
            mouth_right = landmarks_list[291]
            
            # Mouth top and bottom for expression
            mouth_top = landmarks_list[13]
            mouth_bottom = landmarks_list[14]
            
            # Eyebrows for emotion detection
            left_eyebrow_inner = landmarks_list[70]
            left_eyebrow_outer = landmarks_list[107]
            right_eyebrow_inner = landmarks_list[300]
            right_eyebrow_outer = landmarks_list[336]
            
            # Calculate face metrics
            eye_distance = abs(right_eye.x - left_eye.x) * w
            mouth_width = abs(mouth_right.x - mouth_left.x) * w
            mouth_height = abs(mouth_bottom.y - mouth_top.y) * h
            
            # Calculate eyebrow positions (for emotion detection)
            left_eyebrow_height = (left_eye.y - left_eyebrow_inner.y) * h
            right_eyebrow_height = (right_eye.y - right_eyebrow_inner.y) * h
            avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
            
            # Calculate mouth curve (smile detection)
            mouth_center_y = ((mouth_left.y + mouth_right.y) / 2) * h
            mouth_top_y = mouth_top.y * h
            mouth_curve = mouth_center_y - mouth_top_y
            
            # Detect expression (6 emotions)
            expression = "neutral"
            
            # Happy: wide smile, mouth corners raised
            if mouth_width > eye_distance * 1.15 and mouth_curve < -2:
                expression = "happy"
            
            # Excited/Surprised: wide open mouth, raised eyebrows
            elif mouth_height > mouth_width * 0.35 and avg_eyebrow_height > 15:
                expression = "excited/surprised"
            
            # Shocked: very wide open mouth, very raised eyebrows
            elif mouth_height > mouth_width * 0.5 and avg_eyebrow_height > 20:
                expression = "shocked"
            
            # Sad: mouth corners down, normal eyebrows
            elif mouth_curve > 2 and mouth_width < eye_distance * 0.9:
                expression = "sad"
            
            # Angry: eyebrows down/furrowed, tight mouth
            elif avg_eyebrow_height < 8 and mouth_width < eye_distance * 0.95:
                expression = "angry"
            
            # Neutral: default state
            else:
                expression = "neutral"
            
            # Calculate face quality score
            quality_score = 0
            
            # Check if eyes are properly spaced
            if 30 < eye_distance < 200:
                quality_score += 25
            
            # Check if mouth is visible
            if 20 < mouth_width < 150:
                quality_score += 25
            
            # Check if nose is centered
            face_center_x = w / 2
            nose_x = nose_tip.x * w
            if abs(nose_x - face_center_x) < w * 0.2:
                quality_score += 25
            
            # Check if all key landmarks are visible
            if all([left_eye.visibility > 0.5, right_eye.visibility > 0.5, 
                   nose_tip.visibility > 0.5]):
                quality_score += 25
            
            return {
                "landmarks_detected": len(landmarks_list),
                "expression": expression,
                "quality_score": quality_score,
                "eye_distance": eye_distance,
                "mouth_width": mouth_width,
                "face_centered": abs(nose_x - face_center_x) < w * 0.2,
                "is_valid_face": quality_score >= 75,
                "detection_method": "MediaPipe Landmarks"
            }
            
    except Exception as e:
        print(f"Facial landmark error: {e}")
        return None

def detect_faces_mtcnn(frame):
    """Detect faces using MTCNN (Multi-task CNN - very accurate)."""
    if not MTCNN_AVAILABLE:
        return []
    
    try:
        detector = MTCNN()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)
        
        faces = []
        for result in results:
            if result['confidence'] > 0.9:  # High confidence only
                x, y, w, h = result['box']
                faces.append((x, y, x+w, y+h))
        return faces
    except Exception as e:
        print(f"MTCNN error: {e}")
        return []

def detect_faces_retinaface(frame):
    """Detect faces using RetinaFace (state-of-the-art accuracy)."""
    if not RETINAFACE_AVAILABLE:
        return []
    
    try:
        results = RetinaFace.detect_faces(frame)
        faces = []
        
        if isinstance(results, dict):
            for key in results.keys():
                face_data = results[key]
                if face_data['score'] > 0.9:  # High confidence only
                    facial_area = face_data['facial_area']
                    x1, y1, x2, y2 = facial_area
                    faces.append((x1, y1, x2, y2))
        return faces
    except Exception as e:
        print(f"RetinaFace error: {e}")
        return []

def detect_faces_fr(frame):
    """Detect faces using face_recognition library."""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    
    # Convert from (top, right, bottom, left) to (x1, y1, x2, y2)
    faces = []
    for (top, right, bottom, left) in face_locations:
        faces.append((left, top, right, bottom))
    
    return faces

def detect_faces_combined(frame):
    """Detect faces using multiple methods for best accuracy."""
    all_faces = []
    
    # Try methods in order of accuracy
    if RETINAFACE_AVAILABLE:
        print("  Using RetinaFace (most accurate)...")
        faces = detect_faces_retinaface(frame)
        if faces:
            return faces
    
    if MEDIAPIPE_AVAILABLE:
        print("  Using MediaPipe (Google's solution)...")
        faces = detect_faces_mediapipe(frame)
        if faces:
            return faces
    
    if MTCNN_AVAILABLE:
        print("  Using MTCNN (Multi-task CNN)...")
        faces = detect_faces_mtcnn(frame)
        if faces:
            return faces
    
    if FACE_RECOGNITION_AVAILABLE:
        print("  Using face_recognition library...")
        faces = detect_faces_fr(frame)
        if faces:
            return faces
    
    # Fallback to OpenCV Haar Cascade
    print("  Using OpenCV Haar Cascade (fallback)...")
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        return [(x, y, x+w, y+h) for (x, y, w, h) in detected]
    except:
        pass
    
    return []


def main():
    print("=" * 60)
    print("AI Face Recognition + Expression Detection")
    print("=" * 60)
    print("\nControls:")
    print("  'r' - Recognize face (AI face recognition from database)")
    print("  'a' - Add face to database (learn new person)")
    print("  'l' - List known faces in database")
    print("  's' - Save current frame")
    print("  'q' - Quit")
    print("\n")
    
    # Check API key
    if SERPAPI_KEY == "YOUR_SERPAPI_KEY_HERE":
        print("WARNING: SerpAPI key not set!")
        print("Get a free key at: https://serpapi.com")
        print("Set it in the script or as environment variable SERPAPI_KEY")
        print("\n")
    
    # Initialize face identifier
    identifier = FaceIdentifier(SERPAPI_KEY)
    
    # Initialize AI face recognizer
    face_recognizer = DeepFaceRecognizer("face_database")
    known_faces = face_recognizer.list_known_faces()
    if known_faces:
        print("Known faces in database:")
        for name, count in known_faces:
            print(f"  - {name} ({count} image(s))")
    else:
        print("No faces in database yet. Press 'a' to add faces.")
    print()
    
    # Load YOLO model for person detection
    model = YOLO("yolov8n.pt")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Store identified names and recognition results for each face
    face_names = {}  # face_index -> {name, confidence, method}
    search_status = ""
    last_faces = []
    recognition_mode = False  # Toggle between expression and recognition display
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        
        annotated_frame = frame.copy()
        
        # Detect faces using combined method (tries multiple detectors)
        faces = detect_faces_combined(frame)
        
        # If no faces found with advanced methods, try YOLO as last resort
        if not faces:
            faces = detect_faces_yolo(frame, model)
        
        last_faces = faces  # Store for identification
        
        # Real-time expression detection for all detected faces
        face_expressions = {}
        for i, (x1, y1, x2, y2) in enumerate(faces):
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                landmark_data = analyze_facial_landmarks(face_crop)
                if landmark_data:
                    face_expressions[i] = {
                        'expression': landmark_data['expression'],
                        'quality': landmark_data['quality_score'],
                        'landmarks': landmark_data['landmarks_detected'],
                        'method': landmark_data.get('detection_method', 'Unknown'),
                        'confidence': landmark_data.get('confidence', landmark_data['quality_score'])
                    }
        
        # Check for completed searches
        # for face_id, name in identifier.get_results():
        #     face_names[face_id] = name
        #     search_status = f"Identified: {name}"
        
        # Draw faces with expression labels and recognition results
        for i, (x1, y1, x2, y2) in enumerate(faces):
            # Check if face is recognized
            if i in face_names:
                recognition_info = face_names[i]
                label = f"{recognition_info['name']} ({int(recognition_info['confidence'])}%)"
                color = (255, 0, 255)  # Magenta for recognized faces
            else:
                # Get expression data if available
                expression_info = face_expressions.get(i, None)
                
                if expression_info:
                    # Show emotion, confidence, and method
                    method_short = "AI" if "DeepFace" in expression_info['method'] else "MP"
                    label = f"{expression_info['expression']} {int(expression_info['confidence'])}% [{method_short}]" 
                    # Color based on confidence: green = good, yellow = ok, red = poor
                    if expression_info['confidence'] >= 75:
                        color = (0, 255, 0)  # Green
                    elif expression_info['confidence'] >= 50:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                else:
                    label = "Detecting..."
                    color = (255, 255, 0)  # Cyan
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_bg_y1 = max(y1 - 35, 0)
            cv2.rectangle(annotated_frame, (x1, label_bg_y1), (x2, y1), color, -1)
            
            # Draw expression label
            cv2.putText(
                annotated_frame,
                label[:40],  # Truncate long labels
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        # Display status
        status_text = f"Faces: {len(faces)}"
        if face_names:
            recognized = [face_names[i]['name'] for i in face_names.keys()]
            status_text += f" | Recognized: {', '.join(recognized[:2])}"
        elif faces and face_expressions:
            # Show expression summary
            expressions = [face_expressions[i]['expression'] for i in face_expressions.keys()]
            status_text += f" | Expressions: {', '.join(expressions[:3])}"
        if search_status:
            status_text += f" | {search_status}"
        
        # Convert to PIL for nicer text
        pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 30)
            small_font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 20)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Draw status
        draw.text((10, 10), status_text, font=font, fill=(0, 0, 0))
        draw.text((10, 450), "'r'=Recognize 'a'=Add face 'l'=List 's'=Save 'q'=Quit", font=small_font, fill=(100, 100, 100))
        
        annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Show frame
        cv2.imshow("Face Recognition - C.Linjewile", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        # elif key == ord('i'):
        #     # Identify faces
        #     if faces:
        #         print(f"\nIdentifying {len(faces)} face(s)...")
        #         for i, (x1, y1, x2, y2) in enumerate(faces):
        #             height, width = frame.shape[:2]
        #             face_w = x2 - x1
        #             face_h = y2 - y1
        #             
        #             # For face recognition: TIGHT crop with minimal context
        #             # This prevents capturing phones, backgrounds, etc.
        #             # Only expand by 20% to get full head/hair but exclude objects
        #             expand_w = int(face_w * 0.2)
        #             expand_h = int(face_h * 0.3)  # Slightly more vertical for full head
        #             
        #             crop_x1 = max(0, x1 - expand_w)
        #             crop_y1 = max(0, y1 - expand_h)
        #             crop_x2 = min(width, x2 + expand_w)
        #             crop_y2 = min(height, y2 + expand_h)
        #             
        #             # Crop face tightly from frame
        #             face_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        #             
        #             # CRITICAL: Blur/obscure background to force focus on face
        #             # Create a face mask
        #             face_center_x = (crop_x2 - crop_x1) // 2
        #             face_center_y = (crop_y2 - crop_y1) // 2
        #             face_mask = np.zeros(face_img.shape[:2], dtype=np.uint8)
        #             cv2.ellipse(face_mask, 
        #                        (face_center_x, face_center_y),
        #                        (face_w//2, face_h//2),
        #                        0, 0, 360, 255, -1)
        #             
        #             # Blur the background heavily
        #             blurred_bg = cv2.GaussianBlur(face_img, (51, 51), 0)
        #             
        #             # Combine: sharp face + blurred background
        #             face_mask_3ch = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR) / 255.0
        #             face_img = (face_img * face_mask_3ch + blurred_bg * (1 - face_mask_3ch)).astype(np.uint8)
        #             
        #             # CRITICAL VALIDATION: Verify this actually contains a face using multiple methods
        #             face_validated = False
        #             validation_method = "None"
        #             
        #             # PRIORITY 1: Facial Landmark Analysis (most reliable)
        #             print("\n  🔍 Step 1: Analyzing facial landmarks...")
        #             landmark_data = analyze_facial_landmarks(face_img)
        #             if landmark_data and landmark_data['is_valid_face']:
        #                 face_validated = True
        #                 validation_method = f"Facial Landmarks (score: {landmark_data['quality_score']}/100, expression: {landmark_data['expression']})"
        #                 print(f"  ✓✓✓ VALIDATED: Real human face detected!")
        #                 print(f"       Expression: {landmark_data['expression']}")
        #                 print(f"       Quality Score: {landmark_data['quality_score']}/100")
        #             else:
        #                 print("  ⚠ Facial landmarks not detected or quality too low")
        #             
        #             # Try RetinaFace (most accurate object detection)
        #             if RETINAFACE_AVAILABLE and not face_validated:
        #                 print("\n  🔍 Step 2: Trying RetinaFace...")
        #                 try:
        #                     results = RetinaFace.detect_faces(face_img)
        #                     if isinstance(results, dict) and len(results) > 0:
        #                         face_validated = True
        #                         validation_method = "RetinaFace"
        #                         print("  ✓ Validated with RetinaFace")
        #                 except:
        #                     pass
        #             
        #             # Try MediaPipe
        #             if MEDIAPIPE_AVAILABLE and not face_validated:
        #                 print("\n  🔍 Step 3: Trying MediaPipe...")
        #                 try:
        #                     import mediapipe as mp
        #                     mp_face_detection = mp.solutions.face_detection
        #                     
        #                     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        #                         rgb_check = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        #                         results = face_detection.process(rgb_check)
        #                         if results.detections and len(results.detections) > 0:
        #                             face_validated = True
        #                             validation_method = "MediaPipe"
        #                             print("  ✓ Validated with MediaPipe")
        #                 except:
        #                     pass
        #             
        #             # Try MTCNN
        #             if MTCNN_AVAILABLE and not face_validated:
        #                 print("\n  🔍 Step 4: Trying MTCNN...")
        #                 try:
        #                     detector = MTCNN()
        #                     rgb_check = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        #                     results = detector.detect_faces(rgb_check)
        #                     if results and len(results) > 0 and results[0]['confidence'] > 0.8:
        #                         face_validated = True
        #                         validation_method = "MTCNN"
        #                         print("  ✓ Validated with MTCNN")
        #                 except:
        #                     pass
        #             
        #             # Try face_recognition
        #             if FACE_RECOGNITION_AVAILABLE and not face_validated:
        #                 print("\n  🔍 Step 5: Trying face_recognition...")
        #                 try:
        #                     rgb_check = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        #                     face_check = face_recognition.face_locations(rgb_check, model="hog")
        #                     if len(face_check) > 0:
        #                         face_validated = True
        #                         validation_method = "face_recognition"
        #                         print("  ✓ Validated with face_recognition")
        #                 except:
        #                     pass
        #             
        #             # Fallback: OpenCV Haar Cascade
        #             if not face_validated:
        #                 print("\n  🔍 Step 6: Trying OpenCV Haar Cascade (fallback)...")
        #                 try:
        #                     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #                     gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        #                     detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        #                     
        #                     if len(detected_faces) > 0:
        #                         face_validated = True
        #                         validation_method = "OpenCV Haar"
        #                         print("  ✓ Validated with OpenCV Haar")
        #                 except Exception as e:
        #                     print(f"  ⚠ Face validation error: {e}")
        #             
        #             if face_validated:
        #                 print(f"\n  ✅ Face {i}: VALIDATED with {validation_method}")
        #             else:
        #                 print(f"\n  ❌ Face {i}: FAILED ALL VALIDATIONS")
        #                 print(f"     This is likely an OBJECT, not a face (glasses, phone, etc.)")
        #                 print(f"     Skipping to prevent false identification...")
        #                 continue
        #             
        #             if face_img.size > 0:
        #                 # Save debug images for analysis
        #                 debug_path = f"face_capture_{i}.jpg"
        #                 debug_enhanced_path = f"face_capture_{i}_enhanced.jpg"
        #                 cv2.imwrite(debug_path, face_img)
        #                 print(f"  Saved debug image: {debug_path}")
        #                 
        #                 identifier.identify_face(face_img, i)
        #                 face_names[i] = "Searching..."
        #         search_status = "Search started..."
        #     else:
        #         print("No faces detected!")
        #         search_status = "No faces detected"
        
        elif key == ord('r'):
            # Recognize faces using AI
            if not last_faces:
                print("No faces detected!")
                search_status = "No faces detected"
            else:
                print(f"\n{'='*50}")
                print(f"Recognizing {len(last_faces)} face(s) using AI...")
                print(f"{'='*50}")
                
                face_names.clear()  # Clear previous recognitions
                
                for i, (x1, y1, x2, y2) in enumerate(last_faces):
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        print(f"\nFace {i+1}:")
                        result = face_recognizer.recognize_face(face_crop)
                        
                        if result['status'] == 'match':
                            face_names[i] = {
                                'name': result['name'],
                                'confidence': result['confidence'],
                                'method': 'DeepFace AI'
                            }
                            search_status = f"Recognized: {result['name']}"
                        elif result['status'] == 'empty_db':
                            search_status = "No faces in database"
                            print(result['message'])
                        else:
                            search_status = "Unknown person"
                            print(result.get('message', 'No match found'))
                
                if face_names:
                    print(f"\n{'='*50}")
                    print("Recognition complete!")
                    print(f"{'='*50}")
        
        elif key == ord('a'):
            # Add face to database
            if not last_faces:
                print("No faces detected!")
                search_status = "No faces detected"
            else:
                print("\n" + "="*50)
                print(f"Detected {len(last_faces)} face(s)")
                print("="*50)
                
                # Ask for name
                print("\nEnter person's name (or press Enter to cancel): ", end='', flush=True)
                person_name = input().strip()
                
                if person_name:
                    # Add all detected faces
                    added = 0
                    for i, (x1, y1, x2, y2) in enumerate(last_faces):
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            if face_recognizer.add_face(face_crop, person_name):
                                added += 1
                    
                    search_status = f"Added {added} face(s) for {person_name}"
                    print(f"\n✓ Added {added} face(s) to database as '{person_name}'")
                else:
                    search_status = "Cancelled"
                    print("Cancelled")
        
        elif key == ord('l'):
            # List known faces
            known_faces = face_recognizer.list_known_faces()
            if known_faces:
                print("\n" + "="*50)
                print("Known Faces in Database:")
                print("="*50)
                for name, count in known_faces:
                    print(f"  {name} ({count} image(s))")
                print("="*50)
                search_status = f"{len(known_faces)} people in database"
            else:
                print("\nNo faces in database yet.")
                search_status = "Database empty"
        
        elif key == ord('s'):
            # Save frame
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            search_status = f"Saved: {filename}"
        
        # elif key == ord('d'):
        #     # Download profile pictures
        #     downloaded = 0
        #     print("\nDownloading profile pictures...")
        #     for face_id in face_names.keys():
        #         profiles = identifier.get_profiles(face_id)
        #         if profiles:
        #             for p in profiles:
        #                 if p.get('profile_pic_url'):
        #                     platform = p['platform'].lower()
        #                     username = p.get('username', f"face_{face_id}")
        #                     filename = f"profile_pic_{platform}_{username}_{int(time.time())}.jpg"
        #                     
        #                     if identifier.social_scraper.download_profile_picture(p['profile_pic_url'], filename):
        #                         downloaded += 1
        #     
        #     if downloaded > 0:
        #         print(f"\n✓ Downloaded {downloaded} profile picture(s)")
        #         search_status = f"Downloaded {downloaded} pics"
        #     else:
        #         print("No profile pictures to download. Press 'i' first to identify.")
        #         search_status = "No pics found"
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
