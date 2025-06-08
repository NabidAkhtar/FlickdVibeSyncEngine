import logging
import json
import re
from typing import List, Dict, Optional
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Load vibe taxonomy
with open('data/vibes_list.json', 'r') as f:
    VIBES = json.load(f)

# Expanded keywords for rule-based classification
vibe_keywords = {
    'Coquette': ['pink', 'bows', 'lace', 'feminine', 'frilly', 'cute', 'delicate', 'heart', 'pastel', 'flirtatious', 'girly'],
    'Clean Girl': ['minimal', 'neutral', 'simple', 'elegant', 'sleek', 'white', 'beige', 'natural', 'fresh', 'classic', 'tidy'],
    'Cottagecore': ['floral', 'rustic', 'vintage', 'earthy', 'plaid', 'linen', 'countryside', 'garden', 'cozy', 'whimsical', 'homemade'],
    'Streetcore': ['urban', 'edgy', 'bold', 'streetwear', 'graffiti', 'denim', 'leather', 'sneakers', 'grunge', 'skater', 'hiphop'],
    'Y2K': ['shiny', 'metallic', 'retro', 'futuristic', 'glitter', 'butterfly', 'neon', '90s', 'cyber', 'pop', 'nostalgic'],
    'Boho': ['bohemian', 'flowy', 'eclectic', 'fringe', 'tassel', 'ethnic', 'gypsy', 'maxi', 'hippie', 'layered', 'earthy'],
    'Party Glam': ['sparkly', 'glamorous', 'sequins', 'bold', 'glitter', 'shimmer', 'rhinestone', 'dazzling', 'fancy', 'nightlife', 'luxurious']
}

# Vibe descriptions for transformer-based classification
vibe_descriptions = {
    'Coquette': 'A flirty, feminine aesthetic with pink tones, lace, bows, and a delicate, girly vibe.',
    'Clean Girl': 'A minimalist, elegant look with neutral colors like white and beige, focusing on simplicity and natural beauty.',
    'Cottagecore': 'A cozy, rustic aesthetic inspired by countryside living, with floral patterns, vintage styles, and earthy tones.',
    'Streetcore': 'An edgy, urban style with bold streetwear, denim, leather, and a grunge or skater influence.',
    'Y2K': 'A nostalgic, futuristic style from the late 90s and early 2000s, featuring shiny metallics, glitter, neon colors, and retro vibes.',
    'Boho': 'A bohemian, free-spirited look with flowy fabrics, eclectic patterns, fringe, and earthy tones.',
    'Party Glam': 'A glamorous, dazzling style for nightlife, featuring sparkly sequins, glitter, and bold, luxurious looks.'
}

# Fallback vibe inference based on detected products
vibe_fallback = {
    ('dress', 'pink'): 'Coquette',
    ('dress', 'purple'): 'Coquette',
    ('top', 'white'): 'Clean Girl',
    ('bottom', 'beige'): 'Clean Girl',
    ('dress', 'floral'): 'Cottagecore',
    ('top', 'plaid'): 'Cottagecore',
    ('jacket', 'leather'): 'Streetcore',
    ('bottom', 'denim'): 'Streetcore',
    ('top', 'metallic'): 'Y2K',
    ('bag', 'glitter'): 'Y2K',
    ('dress', 'flowy'): 'Boho',
    ('top', 'fringe'): 'Boho',
    ('dress', 'sparkly'): 'Party Glam',
    ('top', 'sequins'): 'Party Glam'
}

class VibeClassifier:
    def __init__(self):
        """
        Initialize the vibe classifier with DistilBERT model for embeddings.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.vibe_names = list(vibe_descriptions.keys())
        self.vibe_desc_embeddings = None

    def _load_model(self):
        """
        Lazy-load the DistilBERT model and precompute vibe description embeddings.
        """
        if self.model is None:
            logger.info("Loading DistilBERT model...")
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
            self.model.eval()
            # Precompute embeddings for vibe descriptions
            self.vibe_desc_embeddings = self._get_embeddings(list(vibe_descriptions.values()))
            logger.info("DistilBERT model loaded and embeddings precomputed")

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute DistilBERT embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed.
        
        Returns:
            np.ndarray: Array of embeddings.
        """
        try:
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token embedding
            # Clear memory
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return embeddings
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            return np.zeros((len(texts), 768))  # DistilBERT embedding size

    def _extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text.
        
        Args:
            text (str): Input text.
        
        Returns:
            List[str]: List of hashtags (without #).
        """
        return [tag.strip("#") for tag in re.findall(r'#\w+', text)]

    def _rule_based_classification(self, text: str) -> Dict[str, float]:
        """
        Rule-based vibe classification using keyword matching.
        
        Args:
            text (str): Combined text (caption + hashtags + transcript).
        
        Returns:
            Dict[str, float]: Vibe scores based on keyword matches.
        """
        text_lower = text.lower()
        vibe_scores = {vibe: 0.0 for vibe in VIBES}
        
        for vibe in VIBES:
            if vibe in vibe_keywords:
                keywords = vibe_keywords[vibe]
                matches = sum(1 for word in keywords if word in text_lower)
                vibe_scores[vibe] = matches / len(keywords) if keywords else 0.0
        
        return vibe_scores

    def _transformer_based_classification(self, text: str) -> Dict[str, float]:
        """
        Transformer-based vibe classification using DistilBERT embeddings.
        
        Args:
            text (str): Combined text (caption + hashtags + transcript).
        
        Returns:
            Dict[str, float]: Vibe scores based on embedding similarity.
        """
        self._load_model()  # Lazy-load the model
        text_embedding = self._get_embeddings([text])[0]
        similarities = cosine_similarity([text_embedding], self.vibe_desc_embeddings)[0]
        vibe_scores = {vibe: float(score) for vibe, score in zip(self.vibe_names, similarities)}
        return vibe_scores

    def classify_vibe(self, caption: str, hashtags: Optional[List[str]] = None, transcript: Optional[str] = None, products: Optional[List[Dict]] = None) -> List[str]:
        """
        Classify vibes using a hybrid NLP approach (rule-based + transformer-based).
        
        Args:
            caption (str): Video caption.
            hashtags (Optional[List[str]]): List of hashtags (without #).
            transcript (Optional[str]): Audio transcript.
            products (Optional[List[Dict]]): List of detected products with type and color.
        
        Returns:
            List[str]: List of up to 3 vibes or ['unknown'] if none match.
        """
        try:
            # Combine all text inputs
            combined_text = caption.strip()
            
            # Add hashtags
            if hashtags:
                combined_text += ' ' + ' '.join(hashtags)
            else:
                # Extract hashtags from caption if not provided separately
                extracted_hashtags = self._extract_hashtags(caption)
                if extracted_hashtags:
                    combined_text += ' ' + ' '.join(extracted_hashtags)
            
            # Add transcript
            if transcript:
                combined_text += ' ' + transcript.strip()
            
            combined_text = combined_text.lower()
            logger.info(f"Combined text for vibe classification: {combined_text}")
            
            # Rule-based classification
            rule_scores = self._rule_based_classification(combined_text)
            logger.info(f"Rule-based scores: {rule_scores}")
            
            # Transformer-based classification
            transformer_scores = self._transformer_based_classification(combined_text)
            logger.info(f"Transformer-based scores: {transformer_scores}")
            
            # Combine scores (weighted average: 0.4 rule-based, 0.6 transformer-based)
            final_scores = {vibe: 0.4 * rule_scores[vibe] + 0.6 * transformer_scores[vibe] for vibe in VIBES}
            logger.info(f"Final scores: {final_scores}")
            
            # Select top 3 vibes with scores above threshold
            sorted_vibes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            top_vibes = [vibe for vibe, score in sorted_vibes if score > 0.3][:3]  # Threshold 0.3
            
            # Fallback using products if no vibes detected
            if not top_vibes and products:
                for product in products:
                    product_type = product.get('type', '').lower()
                    color = product.get('color', '').lower()
                    key = (product_type, color)
                    if key in vibe_fallback:
                        top_vibes.append(vibe_fallback[key])
                top_vibes = list(dict.fromkeys(top_vibes))[:3]  # Remove duplicates, limit to 3
            
            return top_vibes if top_vibes else ['unknown']
        except Exception as e:
            logger.error(f"Error classifying vibe: {str(e)}")
            return ['unknown']

# Singleton instance for efficiency
vibe_classifier = VibeClassifier()

def classify_vibe(caption: str, hashtags: Optional[List[str]] = None, transcript: Optional[str] = None, products: Optional[List[Dict]] = None) -> List[str]:
    """
    Wrapper function to classify vibes using the VibeClassifier instance.
    
    Args:
        caption (str): Video caption.
        hashtags (Optional[List[str]]): List of hashtags (without #).
        transcript (Optional[str]): Audio transcript.
        products (Optional[List[Dict]]): List of detected products with type and color.
    
    Returns:
        List[str]: List of up to 3 vibes or ['unknown'] if none match.
    """
    return vibe_classifier.classify_vibe(caption, hashtags, transcript, products)