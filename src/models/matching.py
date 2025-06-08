import logging
import pandas as pd
import clip
import torch
from PIL import Image
import faiss
import numpy as np
import aiohttp
import asyncio
import io
import os
import hashlib
import pickle
from typing import List, Tuple
import psutil
import gc

logger = logging.getLogger(__name__)

class ProductMatcher:
    def __init__(self, images_path: str, product_data_path: str, cache_dir: str = "cache/images"):
        """
        Initialize the ProductMatcher with catalog data and set up the CLIP model and FAISS index.

        Args:
            images_path (str): Path to the CSV file containing image URLs and product IDs
            product_data_path (str): Path to the Excel file containing product information
            cache_dir (str, optional): Directory to store cached images and embeddings. Defaults to "cache/images".
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load and merge catalog data
        self.images_df = pd.read_csv(images_path)
        self.product_df = pd.read_excel(product_data_path)
        self.catalog = pd.merge(self.images_df, self.product_df, on='id', how='inner')
        self.product_ids = self.catalog['id'].astype(str).tolist()
        self.image_urls = self.catalog['image_url'].tolist()
        self.product_info = self.catalog.to_dict('records')
        
        # Setup cache
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_status_file = os.path.join(self.cache_dir, "cache_status.pkl")
        self.embeddings_cache_file = os.path.join(self.cache_dir, "embeddings.npy")
        self.indices_cache_file = os.path.join(self.cache_dir, "valid_indices.pkl")
        self.valid_embeddings_file = os.path.join(self.cache_dir, "valid_embeddings.pkl")
        self.catalog_hash_file = os.path.join(self.cache_dir, "catalog_hash.txt")
        
        # Compute catalog hash to detect changes
        with open(images_path, 'rb') as f:
            self.current_catalog_hash = hashlib.md5(f.read()).hexdigest()
        
        # Check if catalog has changed
        if os.path.exists(self.catalog_hash_file):
            with open(self.catalog_hash_file, 'r') as f:
                stored_hash = f.read().strip()
            if stored_hash != self.current_catalog_hash:
                logger.info("Catalog file has changed. Invalidating cache...")
                self.cache_status = {}
                for f in [self.embeddings_cache_file, self.indices_cache_file, self.valid_embeddings_file]:
                    if os.path.exists(f):
                        os.remove(f)
            else:
                self.cache_status = self._load_cache_status()
        else:
            self.cache_status = {}
        
        with open(self.catalog_hash_file, 'w') as f:
            f.write(self.current_catalog_hash)
        
        # Build FAISS index
        self.index, self.valid_indices, self.valid_embeddings_mask = self._build_faiss_index()
        if self.index is None:
            logger.warning("No valid images loaded. Matching will return 'no_match'.")
        else:
            logger.info(f"FAISS index built with {self.index.ntotal} embeddings, {sum(self.valid_embeddings_mask)} valid")

    def _log_memory_usage(self, stage: str):
        """
        Log the current memory usage at a specific stage of processing.

        Args:
            stage (str): Description of the current processing stage for logging context
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"Memory usage at {stage}: {mem_info.rss / 1024 / 1024:.2f} MB")

    def _load_cache_status(self) -> dict:
        """
        Load the cache status from disk.

        Returns:
            dict: Dictionary containing cache status for each URL
        """
        if os.path.exists(self.cache_status_file):
            try:
                with open(self.cache_status_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache status: {str(e)}. Resetting cache.")
                return {}
        return {}

    def _save_cache_status(self):
        """
        Save the current cache status to disk.
        """
        try:
            with open(self.cache_status_file, 'wb') as f:
                pickle.dump(self.cache_status, f)
        except Exception as e:
            logger.error(f"Error saving cache status: {str(e)}")

    def _get_cache_path(self, url: str) -> str:
        """
        Generate a cache file path for a given URL.

        Args:
            url (str): Image URL to generate cache path for

        Returns:
            str: Absolute path to the cached file
        """
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.jpg")

    async def _load_image(self, url: str, session: aiohttp.ClientSession = None, retries: int = 3) -> np.ndarray:
        """
        Load an image from URL or local file system with caching.

        Args:
            url (str): URL or file path of the image to load
            session (aiohttp.ClientSession, optional): HTTP session for making requests. Defaults to None.
            retries (int, optional): Number of retry attempts for failed downloads. Defaults to 3.


        Returns:
            np.ndarray: Loaded image as a numpy array, or None if loading failed
        """
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        cache_path = self._get_cache_path(url)
        
        # Check cache first
        if url_hash in self.cache_status:
            if self.cache_status[url_hash] == 'failed':
                return None
            if os.path.exists(cache_path):
                try:
                    image = Image.open(cache_path).convert('RGB')
                    image = image.resize((512, 512), Image.Resampling.LANCZOS)
                    return np.array(image)
                except Exception as e:
                    logger.error(f"Failed to load cached image for {url}: {str(e)}")
                    self.cache_status[url_hash] = 'failed'
                    self._save_cache_status()
                    return None
        
        # If not cached, download or load local file
        try:
            if url.startswith('file://'):
                file_path = url.replace('file://', '')
                if os.path.exists(file_path):
                    image = Image.open(file_path).convert('RGB')
                else:
                    raise FileNotFoundError(f"Local file not found: {file_path}")
            else:
                if session is None:
                    raise ValueError("HTTP session required for URL download")
                for attempt in range(retries):
                    try:
                        async with session.get(url, timeout=30) as response:
                            response.raise_for_status()
                            content = await response.read()
                            image = Image.open(io.BytesIO(content)).convert('RGB')
                            break
                    except Exception as e:
                        wait_time = (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {str(e)}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        if attempt == retries - 1:
                            raise e
            
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            image.save(cache_path, 'JPEG')
            self.cache_status[url_hash] = 'success'
            self._save_cache_status()
            img_array = np.array(image)
            del image
            gc.collect()
            return img_array
        except Exception as e:
            logger.error(f"Failed to load image {url}: {str(e)}")
            self.cache_status[url_hash] = 'failed'
            self._save_cache_status()
            return None

    async def _download_images(self) -> List[Tuple[int, np.ndarray]]:
        """
        Download all catalog images asynchronously.

        Returns:
            List[Tuple[int, np.ndarray]]: List of tuples containing (index, image_array) for each successfully loaded image
        """
        results = []
        async with aiohttp.ClientSession() as session:
            for idx, url in enumerate(self.image_urls):
                try:
                    img = await self._load_image(url, session)
                    results.append((idx, img))
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processed {idx + 1}/{len(self.image_urls)} images")
                        self._log_memory_usage(f"after processing {idx + 1} images")
                except Exception as e:
                    logger.error(f"Error processing image {idx}: {str(e)}")
                    results.append((idx, None))
        return results

    def _get_embeddings(self, images: List[np.ndarray], batch_size: int = 4) -> np.ndarray:
        """
        Compute CLIP embeddings for a list of images in batches.

        Args:
            images (List[np.ndarray]): List of images to compute embeddings for
            batch_size (int, optional): Number of images to process in each batch. Defaults to 4.


        Returns:
            np.ndarray: 2D array of embeddings with shape (n_images, embedding_dim),
                     or empty array if computation fails
        """
        """
        Compute CLIP embeddings for a list of images.
        
        Args:
            images (List[np.ndarray]): List of images to compute embeddings for.
            batch_size (int): Batch size for embedding computation.
        
        Returns:
            np.ndarray: Array of embeddings, or empty array if computation fails.
        """
        try:
            if not images:
                logger.warning("No images provided for embedding computation")
                return np.array([])

            embeddings = []
            total_images = len(images)
            for i in range(0, total_images, batch_size):
                batch = [self.preprocess(Image.fromarray(img)).to(self.device) for img in images[i:i + batch_size] if img is not None]
                if not batch:
                    continue
                batch_images = torch.stack(batch)
                with torch.no_grad():
                    batch_embeddings = self.model.encode_image(batch_images)
                # Convert to numpy and check for NaN/infinite values
                batch_emb = batch_embeddings.cpu().numpy()
                if np.any(np.isnan(batch_emb)) or np.any(np.isinf(batch_emb)):
                    logger.warning("NaN or infinite values found in embeddings. Skipping batch.")
                    continue
                embeddings.append(batch_emb)
                del batch_images
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                logger.info(f"Computed embeddings for batch {i // batch_size + 1}/{(total_images + batch_size - 1) // batch_size}")
                self._log_memory_usage(f"after batch {i // batch_size + 1}")
            
            if not embeddings:
                logger.warning("No valid embeddings computed")
                return np.array([])
            
            result = np.concatenate(embeddings, axis=0)
            # Ensure the result is a 2D array with shape (n, d)
            if len(result.shape) != 2:
                logger.error(f"Invalid embedding shape: {result.shape}")
                return np.array([])
            return result
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            return np.array([])

    def _build_faiss_index(self) -> tuple:
        """
        Build or load a FAISS index for efficient similarity search.

        Returns:
            tuple: (faiss.Index, List[int], List[bool]) containing:
                - FAISS index for similarity search
                - List of valid catalog indices
                - Boolean mask indicating which catalog entries have valid embeddings
        """
        try:
            if os.path.exists(self.embeddings_cache_file) and os.path.exists(self.indices_cache_file) and os.path.exists(self.valid_embeddings_file):
                embeddings = np.load(self.embeddings_cache_file)
                with open(self.indices_cache_file, 'rb') as f:
                    valid_indices = pickle.load(f)
                with open(self.valid_embeddings_file, 'rb') as f:
                    valid_embeddings_mask = pickle.load(f)
                logger.info(f"Loaded {len(embeddings)} embeddings from cache, {sum(valid_embeddings_mask)} valid")
            else:
                logger.info("Starting image download...")
                loop = asyncio.get_event_loop()
                image_pairs = loop.run_until_complete(self._download_images())
                
                if not image_pairs:
                    logger.error("No images processed. Matching disabled.")
                    return None, [], []
                
                image_pairs.sort(key=lambda x: x[0])
                images = []
                valid_indices = list(range(len(self.image_urls)))
                valid_embeddings_mask = [False] * len(self.image_urls)
                
                for idx, img in image_pairs:
                    if img is not None:
                        images.append(img)
                        valid_embeddings_mask[idx] = True
                
                if not any(valid_embeddings_mask):
                    logger.error("No valid images loaded. Matching disabled.")
                    return None, [], []
                
                logger.info(f"Computing embeddings for {len(images)} images...")
                self._log_memory_usage("before computing embeddings")
                real_embeddings = self._get_embeddings(images, batch_size=4)
                if real_embeddings.size == 0:
                    logger.error("No valid embeddings generated. Matching disabled.")
                    return None, [], []
                
                embedding_dim = real_embeddings.shape[1]
                embeddings = np.zeros((len(self.image_urls), embedding_dim), dtype=np.float32)
                
                real_idx = 0
                for idx, is_valid in enumerate(valid_embeddings_mask):
                    if is_valid:
                        embeddings[idx] = real_embeddings[real_idx]
                        real_idx += 1
                
                np.save(self.embeddings_cache_file, embeddings)
                with open(self.indices_cache_file, 'wb') as f:
                    pickle.dump(valid_indices, f)
                with open(self.valid_embeddings_file, 'wb') as f:
                    pickle.dump(valid_embeddings_mask, f)
                logger.info(f"Cached {len(embeddings)} embeddings, {sum(valid_embeddings_mask)} valid")
            
            logger.info("Building FAISS index...")
            self._log_memory_usage("before building FAISS index")
            index = faiss.IndexFlatIP(embeddings.shape[1])
            # Normalize embeddings before adding to FAISS index
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                logger.error("NaN or infinite values in catalog embeddings. Cannot build FAISS index.")
                return None, [], []
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            self._log_memory_usage("after building FAISS index")
            logger.info("FAISS index built successfully")
            return index, valid_indices, valid_embeddings_mask
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            return None, [], []

    def match_product(self, image: np.ndarray, detected_label: str = None) -> tuple:
        """
        Match a product image against the catalog.

        Args:
            image (np.ndarray): Input image to match (BGR format)
            detected_label (str, optional): Detected product category for validation. Defaults to None.


        Returns:
            tuple: (match_type, product_id, similarity_score) where:
                - match_type: 'exact', 'similar', or 'no_match'
                - product_id: Matched product ID or None if no match
                - similarity_score: Float between 0 and 1 indicating match confidence
        """
        try:
            if self.index is None:
                logger.warning("No FAISS index available. Returning 'no_match'.")
                return 'no_match', None, 0.0
        
            # Resize input image for matching
            if image.shape[0] > 512 or image.shape[1] > 512:
                image = Image.fromarray(image).resize((512, 512), Image.Resampling.LANCZOS)
                image = np.array(image)
        
            embedding = self._get_embeddings([image])
            if embedding.size == 0:
                logger.warning("Failed to compute embedding for the cropped image.")
                return 'no_match', None, 0.0
        
            if len(embedding.shape) != 2 or embedding.shape[0] != 1:
                logger.error(f"Invalid embedding shape: {embedding.shape}")
                return 'no_match', None, 0.0
        
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                logger.warning("NaN or infinite values in embedding. Cannot perform matching.")
                return 'no_match', None, 0.0
        
            embedding = embedding.astype(np.float32)
            faiss.normalize_L2(embedding)
            distances, indices = self.index.search(embedding, k=1)
            similarity = float(distances[0][0])
            logger.info(f"Similarity score: {similarity}")
        
            matched_idx = indices[0][0]
            if not self.valid_embeddings_mask[matched_idx]:
                logger.info(f"Matched index {matched_idx} has a placeholder embedding. Returning 'no_match'.")
                return 'no_match', None, similarity
        
            # Adjust similarity thresholds
            if similarity > 0.90:
                match_type = 'exact'
            elif similarity > 0.75:
                match_type = 'similar'
            else:
                match_type = 'no_match'
        
            if match_type == 'no_match':
                logger.info(f"No match found with similarity {similarity}")
                return 'no_match', None, similarity
        
            catalog_idx = self.valid_indices[matched_idx]
            product_id = str(self.product_ids[catalog_idx])
        
            # Validate label-to-product-type match
            product_info = self.get_product_info(product_id)
            product_type = product_info["product_type"].lower()
        
            if detected_label:
                # Define mapping of detected labels to acceptable product types
                label_to_product_types = {
                    "dress": ["dress", "co-ord", "playsuit"],
                    "skirt": ["skirt", "skorts"],
                    "jumpsuit": ["jumpsuit", "co-ord"],
                    "top": ["top", "shirt", "t-shirt"],
                    "jeans": ["jeans"],
                    "jacket": ["jacket"],
                    "shorts": ["shorts"],
                    "trouser": ["trouser"],
                    "neckline": ["top"],  # Treat "neckline" as a top
                }
            
                detected_label_lower = detected_label.lower()
                if detected_label_lower in label_to_product_types:
                    if product_type not in label_to_product_types[detected_label_lower]:
                        logger.info(f"Match rejected: detected label '{detected_label}' does not match product type '{product_type}'")
                        return 'no_match', None, similarity
                else:
                    logger.info(f"Invalid detected label '{detected_label}'. Returning 'no_match'.")
                    return 'no_match', None, similarity
        
            logger.info(f"Match found: type={match_type}, product_id={product_id}, similarity={similarity}")
            return match_type, product_id, similarity
        except Exception as e:
            logger.error(f"Error matching product: {str(e)}")
            return 'no_match', None, 0.0

    def get_product_info(self, product_id: str) -> dict:
        """
        Retrieve product information including type and color.

        Args:
            product_id (str): ID of the product to look up

        Returns:
            dict: Dictionary containing product information with keys:
                - product_type: Type/category of the product
                - color: Detected color of the product
        """
        product = self.catalog[self.catalog['id'] == int(product_id)].iloc[0]
        tags = product['product_tags']
        color = 'unknown'
        for tag in tags.split(','):
            if tag.strip().startswith('Colour:'):
                color = tag.split(':')[1].strip().lower()
                break
        return {'product_type': product['product_type'], 'color': color}

    