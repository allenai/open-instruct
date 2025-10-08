import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import random
from anthropic import Anthropic

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None
import os
from typing import List, Dict, Tuple, Any, Optional
import logging
import pandas as pd
from tqdm import tqdm
import math
import json
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalClustering:
    def __init__(
        self, 
        criteria: List[str], 
        n_top_clusters: int, 
        num_levels: int,
        anthropic_api_key: str = None,
        openai_api_key: str = None
    ):
        """
        Initialize the hierarchical clustering algorithm.
        
        Args:
            criteria: List of base-level criteria to cluster
            n_top_clusters: Desired number of top-level clusters
            num_levels: Number of levels in the hierarchy
            anthropic_api_key: API key for Anthropic's Claude API
            openai_api_key: API key for OpenAI's GPT API
        """
        self.criteria = criteria
        self.n_base = len(criteria)
        self.n_top = n_top_clusters
        self.L = num_levels
        
        # Calculate the ratio between successive levels
        self.ratio = (self.n_top / self.n_base) ** (1 / (self.L - 1))
        
        # Initialize the embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize LLM client if an API key is provided
        self.client: Optional[Any] = None
        self.llm_provider: Optional[str] = None
        self.llm_model: Optional[str] = None

        # Prefer Anthropic if available (explicit arg overrides env)
        anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.client = Anthropic(api_key=anthropic_key)
            self.llm_provider = "anthropic"
            self.llm_model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

        # Fall back to OpenAI if Anthropic isn't configured
        if not self.client:
            openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                if OpenAI is None:
                    logger.warning(
                        "OPENAI_API_KEY provided but the `openai` package is not installed. "
                        "Install openai>=1.0.0 to enable OpenAI support."
                    )
                else:
                    self.client = OpenAI(api_key=openai_api_key)
                    self.llm_provider = "openai"
                    self.llm_model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

        if not self.client:
            logger.warning(
                "No Anthropic or OpenAI API key provided. LLM-based refinement will be skipped."
            )
        
        # Will store the hierarchy
        self.hierarchy = {}

    def _llm_generate(self, prompt: str, max_tokens: int) -> Optional[str]:
        """
        Helper to generate LLM output using the configured client.
        """
        if not self.client or not self.llm_model or not self.llm_provider:
            return None

        try:
            if self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.llm_model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                if response and getattr(response, "content", None):
                    return response.content[0].text
            elif self.llm_provider == "openai":
                try:
                    response = self.client.responses.create(
                        model=self.llm_model,
                        input=prompt,
                        max_output_tokens=max_tokens,
                    )
                except TypeError:
                    # Handle older OpenAI SDKs that expect chat completions or ChatCompletion.create
                    chat_messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=chat_messages,
                        max_tokens=max_tokens,
                    )
                    if response and response.choices:
                        message = response.choices[0].message
                        if message and getattr(message, "content", None):
                            return message.content
                    return None
                if response is None:
                    return None

                output_text = getattr(response, "output_text", None)
                if output_text:
                    return output_text

                output = getattr(response, "output", None)
                if output:
                    collected = []
                    for item in output:
                        contents = getattr(item, "content", [])
                        for content in contents:
                            text = getattr(content, "text", None)
                            if text:
                                collected.append(text)
                    if collected:
                        return "\n".join(collected)
            else:
                logger.error(f"Unknown LLM provider '{self.llm_provider}'")
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")

        return None
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using the sentence transformer model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        return self.embedding_model.encode(texts, show_progress_bar=True)
    
    def _best_parent_via_embeddings(
        self,
        items: List[str],
        description: str,
        parent_descriptions: List[str],
    ) -> int:
        """Fallback strategy that uses embedding similarity to choose a parent cluster."""

        combined_text = description + " " + " ".join(items[:10])  # Limit tokens
        embedding = self.embed_texts([combined_text])[0]
        parent_embeddings = self.embed_texts(parent_descriptions)

        similarities = np.dot(parent_embeddings, embedding) / (
            np.linalg.norm(parent_embeddings, axis=1) * np.linalg.norm(embedding)
        )

        return int(np.argmax(similarities))

    def generate_cluster_description(self, 
                                    cluster_items: List[str], 
                                    nearby_items: List[str] = None) -> str:
        """
        Generate a description for a cluster using Claude.
        
        Args:
            cluster_items: Items in the cluster
            nearby_items: Items from nearby clusters to help with boundary cases
            
        Returns:
            Cluster description
        """
        prompt = (
            "You are analyzing a cluster of AI evaluation criteria to generate an accurate label and description.\n\n"
            "Here are the items in this cluster:\n"
            f"{', '.join(cluster_items)}\n\n"
        )

        if nearby_items:
            prompt += (
                "For context, here are some items from nearby clusters (these should NOT be included in your description,"
                " but help you understand boundaries):\n"
                f"{', '.join(nearby_items)}\n\n"
            )

        prompt += (
            "Please provide:\n"
            "1. A concise label (3-5 words) that captures the essence of this cluster\n"
            "2. A brief description (1-2 sentences) explaining what unifies these criteria\n\n"
            "Format your response as:\n"
            "Label: [your label]\n"
            "Description: [your description]\n\n"
            "Focus on what genuinely unifies these criteria rather than superficial similarities."
        )

        response_text = self._llm_generate(prompt, max_tokens=300)
        if response_text is None:
            return f"Cluster containing {len(cluster_items)} items"

        return response_text
    
    def deduplicate_and_refine_descriptions(self, descriptions: List[str]) -> List[str]:
        """
        Deduplicate and refine cluster descriptions using LLM.
        
        Args:
            descriptions: List of candidate descriptions
            
        Returns:
            Refined list of descriptions
        """
        prompt = f"""You are reviewing a set of AI evaluation criteria cluster descriptions.
Your task is to deduplicate them and ensure they are distinct while maintaining coverage.

Here are the current descriptions:
{descriptions}

Please provide a refined set of descriptions that:
1. Eliminates duplicates or highly similar concepts
2. Ensures each description is clearly distinct
3. Maintains coverage of the full conceptual space

Format your response as a numbered list:
1. [Refined description 1]
2. [Refined description 2]
etc.

Use the same number of descriptions as the original list unless there are clear duplicates.
"""

        response_text = self._llm_generate(prompt, max_tokens=1000)
        if response_text is None:
            return descriptions

        refined_descriptions = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                refined_descriptions.append(line.split('. ', 1)[1])

        return refined_descriptions or descriptions
    
    def assign_to_parent_clusters(self, 
                                 lower_clusters: Dict[int, Dict], 
                                 parent_descriptions: List[str]) -> Dict[int, List[int]]:
        """
        Assign lower-level clusters to parent clusters using randomized sampling.
        
        Args:
            lower_clusters: Dictionary of lower-level clusters
            parent_descriptions: List of parent cluster descriptions
            
        Returns:
            Dictionary mapping parent cluster IDs to lists of child cluster IDs
        """
        assignments = {i: [] for i in range(len(parent_descriptions))}
        
        # Randomize the order of lower clusters to avoid order-based bias
        lower_cluster_ids = list(lower_clusters.keys())
        random.shuffle(lower_cluster_ids)
        
        for cluster_id in lower_cluster_ids:
            cluster = lower_clusters[cluster_id]
            
            # Get the cluster's items and description
            items = cluster['items']
            description = cluster['description']
            
            # Find the most appropriate parent
            best_parent = self._find_best_parent(items, description, parent_descriptions)
            
            # Assign to parent
            assignments[best_parent].append(cluster_id)
            
        return assignments
    
    def _find_best_parent(self, 
                         items: List[str], 
                         description: str, 
                         parent_descriptions: List[str]) -> int:
        """
        Find the best parent cluster for a given lower-level cluster.
        
        Args:
            items: Items in the lower-level cluster
            description: Description of the lower-level cluster
            parent_descriptions: List of parent cluster descriptions
            
        Returns:
            Index of the best parent cluster
        """
        # If no LLM is available, fall back to embedding similarity
        if not self.client:
            return self._best_parent_via_embeddings(items, description, parent_descriptions)

        # Use the configured LLM to find the best parent
        items_sample = random.sample(items, min(5, len(items)))
        
        prompt = f"""You are assigning a lower-level cluster to the most appropriate parent cluster.

Lower-level cluster:
- Description: {description}
- Sample items: {', '.join(items_sample)}

Potential parent clusters:
"""
        
        for i, desc in enumerate(parent_descriptions):
            prompt += f"{i+1}. {desc}\n"
            
        prompt += """
Which parent cluster (by number) is the most appropriate for this lower-level cluster? 
Provide your short reasoning and then conclude with "Best parent: X" where X is the number.
"""

        response_text = self._llm_generate(prompt, max_tokens=500)
        if response_text is None:
            return self._best_parent_via_embeddings(items, description, parent_descriptions)

        if "Best parent:" in response_text:
            parent_str = response_text.split("Best parent:")[1].strip().split()[0]
            try:
                parent_idx = int(parent_str) - 1  # Convert to 0-indexed
                return parent_idx
            except ValueError:
                logger.warning(
                    "LLM response did not contain a valid parent index; falling back to embedding similarity"
                )

        logger.warning(
            "Could not extract best parent from LLM response, using embedding similarity"
        )
        return self._best_parent_via_embeddings(items, description, parent_descriptions)
    
    def regenerate_parent_descriptions(self, 
                                      parent_clusters: Dict[int, Dict], 
                                      assignments: Dict[int, List[int]], 
                                      lower_clusters: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Regenerate descriptions for parent clusters based on their assigned contents.
        
        Args:
            parent_clusters: Dictionary of parent clusters
            assignments: Dictionary mapping parent cluster IDs to lists of child cluster IDs
            lower_clusters: Dictionary of lower-level clusters
            
        Returns:
            Updated parent clusters dictionary
        """
        for parent_id, child_ids in assignments.items():
            # Collect all items from children
            all_items = []
            for child_id in child_ids:
                all_items.extend(lower_clusters[child_id]['items'])
            
            # Collect child descriptions
            child_descriptions = [lower_clusters[child_id]['description'] for child_id in child_ids]
            
            # Regenerate parent description
            description = self._generate_parent_description(all_items, child_descriptions)
            
            # Update parent cluster
            parent_clusters[parent_id]['items'] = all_items
            parent_clusters[parent_id]['description'] = description
            parent_clusters[parent_id]['children'] = child_ids
            
        return parent_clusters
    
    def _generate_parent_description(self, 
                                    items: List[str], 
                                    child_descriptions: List[str]) -> str:
        """
        Generate a description for a parent cluster based on its items and child descriptions.
        
        Args:
            items: All items in the parent cluster
            child_descriptions: Descriptions of child clusters
            
        Returns:
            Parent cluster description
        """
        # Sample items to reduce token count
        items_sample = random.sample(items, min(10, len(items)))
        
        prompt = f"""You are generating a concise name and description for a parent-level cluster based on its contents.

This parent cluster contains the following child clusters:
{child_descriptions}

Sample items from this parent cluster:
{', '.join(items_sample)}

Please provide:
1. A concise label (3-5 words) for this parent cluster
2. A brief description (1-2 sentences) explaining what unifies these items

Format your response as:
Label: [your label]
Description: [your description]

Focus on capturing the overarching theme that unites these criteria rather than specific details.
"""

        response_text = self._llm_generate(prompt, max_tokens=300)
        if response_text is None:
            return f"Parent cluster with {len(items)} items"

        return response_text
    
    def build_hierarchy(self) -> Dict:
        """
        Build the complete hierarchy using the described algorithm.
        
        Returns:
            Dictionary representing the complete hierarchy
        """
        logger.info("Starting hierarchical clustering process...")
        
        # Initialize level 0 (base) clusters
        current_level = 0
        current_clusters = {}
        
        for i, criterion in enumerate(self.criteria):
            current_clusters[i] = {
                'level': current_level,
                'items': [criterion],
                'description': criterion,
                'children': []
            }
        
        self.hierarchy[current_level] = current_clusters
        
        # Calculate the number of clusters at each level
        level_sizes = [self.n_base]
        for l in range(1, self.L):
            n_l = int(self.n_base * (self.ratio ** l))
            level_sizes.append(n_l)
        
        logger.info(f"Level sizes: {level_sizes}")
        
        # Build each level
        while current_level < self.L - 1:
            next_level = current_level + 1
            n_clusters = level_sizes[next_level]
            
            logger.info(f"Building level {next_level} with {n_clusters} clusters...")
            
            # Phase 1: Embed clusters and form neighborhoods
            cluster_texts = [cluster['description'] for cluster in current_clusters.values()]
            embeddings = self.embed_texts(cluster_texts)
            
            # Apply k-means clustering to form neighborhoods
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Group clusters by neighborhood
            neighborhoods = {}
            for i, label in enumerate(cluster_labels):
                if label not in neighborhoods:
                    neighborhoods[label] = []
                neighborhoods[label].append(i)
            
            # Phase 2: Generate candidate descriptions for each neighborhood
            logger.info("Generating candidate descriptions...")
            candidate_descriptions = []
            
            for neighborhood_id, cluster_indices in tqdm(neighborhoods.items()):
                # Get in-group clusters
                in_group_clusters = [current_clusters[idx] for idx in cluster_indices]
                in_group_items = []
                for cluster in in_group_clusters:
                    in_group_items.extend(cluster['items'])
                
                # Get nearby out-group clusters for boundary cases
                # This is a simplified approach - we're taking clusters from other neighborhoods
                out_group_indices = []
                for other_id, other_indices in neighborhoods.items():
                    if other_id != neighborhood_id:
                        out_group_indices.extend(other_indices[:2])  # Take 2 clusters from each other neighborhood
                
                out_group_items = []
                for idx in out_group_indices[:10]:  # Limit to 10 out-group clusters
                    if idx < len(current_clusters):
                        out_group_items.extend(current_clusters[idx]['items'][:5])  # Take 5 items from each
                
                # Generate description
                description = self.generate_cluster_description(in_group_items, out_group_items)
                candidate_descriptions.append(description)
            
            # Deduplicate and refine descriptions
            refined_descriptions = self.deduplicate_and_refine_descriptions(candidate_descriptions)
            
            # Initialize next level clusters
            next_clusters = {}
            for i, desc in enumerate(refined_descriptions):
                next_clusters[i] = {
                    'level': next_level,
                    'items': [],
                    'description': desc,
                    'children': []
                }
            
            # Phase 3: Assign lower clusters to parent clusters
            logger.info("Assigning clusters to parents...")
            assignments = self.assign_to_parent_clusters(current_clusters, refined_descriptions)
            
            # Phase 4: Regenerate parent cluster descriptions
            logger.info("Regenerating parent descriptions...")
            next_clusters = self.regenerate_parent_descriptions(next_clusters, assignments, current_clusters)
            
            # Store the level in hierarchy
            self.hierarchy[next_level] = next_clusters
            
            # Move to next level
            current_level = next_level
            current_clusters = next_clusters
        
        logger.info("Hierarchical clustering complete!")
        return self.hierarchy
    
    def visualize_hierarchy(self, output_file: str = "hierarchy.html") -> str:
        """
        Generate a visualization of the hierarchy.
        
        Args:
            output_file: File to save the visualization to
            
        Returns:
            Path to the visualization file
        """
        # This is a placeholder - in a real implementation, you'd use a visualization library
        # like D3.js, NetworkX, or Plotly to create an interactive visualization
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hierarchical Clustering Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .level { margin-bottom: 30px; }
                .cluster { margin-bottom: 15px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
                .children { margin-left: 20px; }
            </style>
        </head>
        <body>
            <h1>Hierarchical Clustering Visualization</h1>
        """
        
        # Add each level
        for level in range(self.L):
            clusters = self.hierarchy.get(level, {})
            html += f"<div class='level'><h2>Level {level} (Clusters: {len(clusters)})</h2>"
            
            for cluster_id, cluster in clusters.items():
                html += f"<div class='cluster'><h3>Cluster {cluster_id}</h3>"
                html += f"<p><strong>Description:</strong> {cluster['description']}</p>"
                
                # Add items if this is a base-level cluster
                if level == 0:
                    html += f"<p><strong>Items:</strong> {', '.join(cluster['items'])}</p>"
                else:
                    # Add children
                    html += "<div class='children'><strong>Children:</strong><ul>"
                    for child_id in cluster['children']:
                        # Get the child from the previous level
                        child = self.hierarchy[level-1].get(child_id, {})
                        child_desc = child.get('description', f"Child {child_id}")
                        html += f"<li>{child_desc}</li>"
                    html += "</ul></div>"
                
                html += "</div>"
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        with open(output_file, "w") as f:
            f.write(html)
        
        return output_file
    
    def export_to_csv(self, output_file: str = "hierarchy.csv") -> str:
        """
        Export the hierarchy to a CSV file.
        
        Args:
            output_file: File to save the CSV to
            
        Returns:
            Path to the CSV file
        """
        rows = []
        
        # Generate rows for each level and cluster
        for level in range(self.L):
            clusters = self.hierarchy.get(level, {})
            
            for cluster_id, cluster in clusters.items():
                description = cluster['description']
                
                # For base level, add each item
                if level == 0:
                    for item in cluster['items']:
                        row = {"Level": level, "Cluster_ID": cluster_id, "Description": description, "Item": item}
                        rows.append(row)
                else:
                    # For higher levels, add one row per cluster with comma-separated children
                    children_ids = cluster['children']
                    children_descs = []
                    
                    for child_id in children_ids:
                        child = self.hierarchy[level-1].get(child_id, {})
                        child_desc = child.get('description', f"Child {child_id}")
                        children_descs.append(child_desc)
                    
                    row = {
                        "Level": level, 
                        "Cluster_ID": cluster_id, 
                        "Description": description, 
                        "Children": ", ".join(map(str, children_ids)),
                        "Children_Descriptions": "; ".join(children_descs)
                    }
                    rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        return output_file

    def compute_diversity(
        self,
        level: Optional[int] = None,
        sample_size: Optional[int] = 5000,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """Compute a diversity metric for the clustered criteria.

        The metric combines two components:
            1. Semantic spread across individual criteria, measured via cosine similarity
               of the criterion embeddings.
            2. Cluster evenness, measured by the normalized entropy of item counts across
               clusters at a given hierarchy level.

        Args:
            level: Hierarchy level to evaluate. Defaults to the highest level built.
            sample_size: Number of criterion pairs to sample when estimating mean cosine
                similarity. If ``None`` or larger than the total number of unique pairs,
                all unique pairs are used.
            random_state: Random seed for sampling criterion pairs.

        Returns:
            Dictionary containing ``semantic_spread``, ``cluster_evenness``, and the
            combined ``diversity_score`` (product of the two components).
        """

        if not self.criteria:
            raise ValueError("No criteria available to compute diversity.")
        if not self.hierarchy:
            raise ValueError("Hierarchy has not been built. Call build_hierarchy() first.")

        embeddings = self.embed_texts(self.criteria)
        if embeddings.ndim != 2 or embeddings.shape[0] <= 1:
            return {
                "semantic_spread": 0.0,
                "cluster_evenness": 0.0,
                "diversity_score": 0.0,
            }

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.clip(norms, 1e-12, None)
        n = normalized_embeddings.shape[0]

        total_pairs = n * (n - 1) // 2
        rng = np.random.default_rng(random_state)

        if sample_size is None or sample_size >= total_pairs:
            similarity_matrix = normalized_embeddings @ normalized_embeddings.T
            tri_upper = np.triu_indices(n, k=1)
            mean_similarity = float(similarity_matrix[tri_upper].mean()) if tri_upper[0].size else 1.0
        else:
            idx_i = rng.integers(0, n, size=sample_size)
            idx_j = rng.integers(0, n, size=sample_size)

            duplicate_mask = idx_i == idx_j
            if duplicate_mask.any():
                idx_j[duplicate_mask] = (idx_j[duplicate_mask] + 1) % n

            pair_sims = np.einsum(
                "ij,ij->i",
                normalized_embeddings[idx_i],
                normalized_embeddings[idx_j],
            )
            mean_similarity = float(pair_sims.mean()) if pair_sims.size else 1.0

        semantic_spread = max(0.0, 1.0 - mean_similarity)

        target_level = level if level is not None else max(self.hierarchy.keys())
        clusters = self.hierarchy.get(target_level)
        if clusters is None:
            raise ValueError(f"Hierarchy level {target_level} is not available.")

        cluster_counts = np.array([len(cluster.get("items", [])) for cluster in clusters.values()], dtype=float)
        total_items = cluster_counts.sum()
        if total_items == 0 or len(cluster_counts) <= 1:
            cluster_evenness = 0.0
        else:
            probs = cluster_counts / total_items
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            cluster_evenness = float(entropy / np.log(len(cluster_counts)))

        diversity_score = float(semantic_spread * cluster_evenness)

        return {
            "semantic_spread": semantic_spread,
            "cluster_evenness": cluster_evenness,
            "diversity_score": diversity_score,
        }
    
    def visualize_tree(self, output_file: str = "hierarchy_tree.html") -> str:
        """
        Generate a minimal tree-like visualization of the hierarchy.
        
        Args:
            output_file: File to save the visualization to
            
        Returns:
            Path to the visualization file
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hierarchy Tree Visualization</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .tree {
                    --spacing: 1.5rem;
                    --radius: 10px;
                }
                .tree li {
                    display: block;
                    position: relative;
                    padding-left: calc(2 * var(--spacing) - var(--radius) - 2px);
                }
                .tree ul {
                    margin-left: calc(var(--radius) - var(--spacing));
                    padding-left: 0;
                }
                .tree ul li {
                    border-left: 2px solid #ddd;
                }
                .tree ul li:last-child {
                    border-color: transparent;
                }
                .tree ul li::before {
                    content: '';
                    display: block;
                    position: absolute;
                    top: calc(var(--spacing) / -2);
                    left: -2px;
                    width: calc(var(--spacing) + 2px);
                    height: calc(var(--spacing) + 1px);
                    border: solid #ddd;
                    border-width: 0 0 2px 2px;
                }
                .tree summary {
                    display: block;
                    cursor: pointer;
                }
                .tree summary::marker,
                .tree summary::-webkit-details-marker {
                    display: none;
                }
                .tree summary:focus {
                    outline: none;
                }
                .tree summary:focus-visible {
                    outline: 1px dotted #000;
                }
                .tree li div.node {
                    position: relative;
                    display: flex;
                    align-items: center;
                    min-height: var(--spacing);
                    padding: 0.25rem 0.75rem;
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: var(--radius);
                    color: #333;
                    font-size: 0.9rem;
                }
                .level-0 > div.node { background-color: #e6f3ff; border-color: #b3d7ff; }
                .level-1 > div.node { background-color: #e6ffe6; border-color: #b3ffb3; }
                .level-2 > div.node { background-color: #fff2e6; border-color: #ffcc99; }
                .level-3 > div.node { background-color: #f9e6ff; border-color: #e6b3ff; }
                .level-4 > div.node { background-color: #ffe6e6; border-color: #ffb3b3; }
                
                details.tree-item {
                    margin-bottom: 0.125rem;
                }
                
                details.tree-item[open] > summary div.node::before {
                    content: "▼";
                    position: absolute;
                    right: 0.5rem;
                    font-size: 0.7rem;
                    color: #666;
                }
                
                details.tree-item:not([open]) > summary div.node::before {
                    content: "▶";
                    position: absolute;
                    right: 0.5rem;
                    font-size: 0.7rem;
                    color: #666;
                }
                
                .node-label {
                    font-weight: bold;
                    margin-right: 0.5rem;
                }
                
                .node-desc {
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    max-width: 500px;
                }
                
                .leaf-item {
                    font-size: 0.85rem;
                    padding: 0.1rem 0;
                }
            </style>
        </head>
        <body>
            <h1>Hierarchy Tree Visualization</h1>
            <p>Click on nodes to expand/collapse. The tree shows minimal descriptions of each cluster.</p>
            <ul class="tree">
        """
        
        # Get the top level
        top_level = max(self.hierarchy.keys())
        top_clusters = self.hierarchy[top_level]

        # Function to recursively build the tree
        def build_tree(level, cluster_id, depth=0):
            cluster = self.hierarchy[level].get(cluster_id, {})
            description = cluster.get('description', '')
            
            tree_html = f'<li class="level-{level}">'
            
            # If this is a leaf node (base level)
            if level == 0:
                # For leaf nodes, use the full original criteria string
                original_criteria = cluster.get('items', [''])[0]
                tree_html += f'<div class="node leaf-node"><span class="node-label">Base {cluster_id}:</span> <span class="node-desc">{original_criteria}</span></div>'
            else:
                # For parent nodes, extract a brief label from the description (first line or first sentence)
                if '\n' in description:
                    label = description.split('\n')[0]
                elif ':' in description:
                    label = description.split(':', 1)[1].strip() if description.startswith('Label:') else description.split(':', 1)[0].strip()
                else:
                    label = description.split('.')[0]
                
                # Limit label length for parent nodes
                label = label[:50] + ('...' if len(label) > 50 else '')
                
                # This is a parent node with children
                children = cluster.get('children', [])
                
                tree_html += f'<details class="tree-item" {"open" if depth < 2 else ""}><summary>'
                tree_html += f'<div class="node"><span class="node-label">L{level}-{cluster_id}:</span> <span class="node-desc">{label}</span></div>'
                tree_html += '</summary><ul>'
                
                # Add children
                for child_id in children:
                    tree_html += build_tree(level - 1, child_id, depth + 1)
                
                tree_html += '</ul></details>'
            
            tree_html += '</li>'
            return tree_html
        
        # # Function to recursively build the tree
        # def build_tree(level, cluster_id, depth=0):
        #     cluster = self.hierarchy[level].get(cluster_id, {})
        #     description = cluster.get('description', '')


            
        #     # Extract a brief label from the description (first line or first sentence)
        #     if '\n' in description:
        #         label = description.split('\n')[0]
        #     elif ':' in description:
        #         label = description.split(':', 1)[1].strip() if description.startswith('Label:') else description.split(':', 1)[0].strip()
        #     else:
        #         label = description.split('.')[0]
            
        #     # Limit label length
        #     label = label[:50] + ('...' if len(label) > 50 else '')
            
        #     tree_html = f'<li class="level-{level}">'
            
        #     # If this is a leaf node (base level)
        #     if level == 0:
        #         tree_html += f'<div class="node"><span class="node-label">Base {cluster_id}:</span> <span class="node-desc">{label}</span></div>'
        #     else:
        #         # This is a parent node with children
        #         children = cluster.get('children', [])
                
        #         tree_html += f'<details class="tree-item" {"open" if depth < 2 else ""}><summary>'
        #         tree_html += f'<div class="node"><span class="node-label">L{level}-{cluster_id}:</span> <span class="node-desc">{label}</span></div>'
        #         tree_html += '</summary><ul>'
                
        #         # Add children
        #         for child_id in children:
        #             tree_html += build_tree(level - 1, child_id, depth + 1)
                
        #         tree_html += '</ul></details>'
            
        #     tree_html += '</li>'
        #     return tree_html
        
        # Build tree starting from top level
        for cluster_id in top_clusters:
            html += build_tree(top_level, cluster_id)
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        with open(output_file, "w") as f:
            f.write(html)
        
        return output_file
    
# Example usage
def run_example(rubric_id: str, num_criteria: int):
    # Set your Anthropic API key here or in environment variables
    # api_key = os.environ["ANTHROPIC_API_KEY"]
    openai_api_key = os.environ["OPENAI_API_KEY"]
    
    # Example criteria for evaluating AI responses
    criteria = [
        "Factual accuracy of information",
        "Logical coherence of arguments",
        "Grammatical correctness",
        "Spelling and punctuation",
        "Relevance to the prompt",
        "Completeness of answer",
        "Clarity of explanation",
        "Conciseness of response",
        "Appropriate level of detail",
        "Absence of hallucinated information",
        "Appropriate citation of sources",
        "Ethical reasoning",
        "Cultural sensitivity",
        "Avoidance of harmful content",
        "Helpful tone",
        "Professional language",
        "Appropriate formality level",
        "Engagement with complex aspects of the query",
        "Creativity when appropriate",
        "Adaptability to user's knowledge level",
        "Consistency within response",
        "Appropriate handling of uncertainty",
        "Balance in presenting multiple viewpoints",
        "Absence of political bias",
        "Transparency about limitations",
        "Appropriateness for intended audience",
        "Conversational flow",
        "Natural language use",
        "Thoughtfulness of analysis",
        "Practicality of advice or solutions"
    ]

    # or read the criteria from a jsonl file
    # with open("outputs/rubric_analysis/wildchat_4k_positive_significant_rubric.jsonl", "r") as f:
    #     criteria = [json.loads(line)["criteria"] for line in f.readlines()]
    # rubric_id = "rl_rag_train_sqa_1k_clean_search_rubric_longform_rubrics"
    rubric_data = load_dataset("rl-rag/"+rubric_id, split="train")
    criteria = []
    for example in rubric_data:
        for key in ["Answer Critical"]: #, "Valuable", "Context"]:
            per_sample_criteria = json.loads(example["ground_truth"])[key]
            for criterion in per_sample_criteria:
                criteria.append(criterion["Ingredient"])

    print(f"Number of criteria: {len(criteria)}")
    newline = "\n"
    print(f"Few examples of criteria: {newline.join(criteria[:5])}")

    rng = random.Random(42)
    rng.shuffle(criteria)

    criteria = criteria[:num_criteria]
    
    # Initialize the clustering algorithm
    # For a real-world scenario with 3,307 values as mentioned in the paper, 
    # you would use different parameters
    hierarchical_clustering = HierarchicalClustering(
        criteria=criteria,
        n_top_clusters=4,  # Desired number of top-level clusters
        num_levels=4,      # Number of levels in the hierarchy
        anthropic_api_key=None,
        openai_api_key=openai_api_key
    )
    
    # Build the hierarchy
    hierarchy = hierarchical_clustering.build_hierarchy()
    
    diversity_metrics = hierarchical_clustering.compute_diversity()
    logger.info(
        "Hierarchy diversity metrics: "
        f"semantic_spread={diversity_metrics['semantic_spread']:.4f}, "
        f"cluster_evenness={diversity_metrics['cluster_evenness']:.4f}, "
        f"diversity_score={diversity_metrics['diversity_score']:.4f}"
    )

    # Ensure output directory exists before writing files
    output_dir = os.path.join("outputs", "rubric_analysis")
    os.makedirs(output_dir, exist_ok=True)

    diversity_path = os.path.join(
        output_dir,
        f"{rubric_id}_hierarchy_diversity_{num_criteria}.json",
    )
    with open(diversity_path, "w", encoding="utf-8") as f:
        json.dump(diversity_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Diversity metrics saved to {diversity_path}")

    # Visualize the hierarchy
    viz_path = os.path.join(output_dir, f"{rubric_id}_hierarchy_{num_criteria}.html")
    viz_file = hierarchical_clustering.visualize_hierarchy(viz_path)
    logger.info(f"Hierarchy visualization saved to {viz_file}")

    # Create the tree visualization
    tree_path = os.path.join(output_dir, f"{rubric_id}_hierarchy_tree_{num_criteria}.html")
    tree_file = hierarchical_clustering.visualize_tree(tree_path)
    logger.info(f"Tree visualization saved to {tree_file}")
    
    # Export to CSV
    csv_path = os.path.join(output_dir, f"{rubric_id}_hierarchy_{num_criteria}.csv")
    csv_file = hierarchical_clustering.export_to_csv(csv_path)
    logger.info(f"Hierarchy exported to {csv_file}")

    
    return hierarchy

if __name__ == "__main__":
    rubric_ids = [
        "rl_rag_train_sqa_1k_clean_search_rubric_longform_rubrics",
        "rl_rag_train_sqa_1k_clean_dr_rubric_longform_rubrics",
        "rl_rag_train_sqa_1k_clean_cb_rubric_longform_rubrics"
    ]
    num_criteria = 300
    for rubric_id in rubric_ids:
        hierarchy = run_example(rubric_id, num_criteria)

    # # Create the tree visualization as well
    # hierarchical_clustering = HierarchicalClustering(
    #     criteria=[],  # Empty list since we're not rebuilding the hierarchy 
    #     n_top_clusters=4,
    #     num_levels=3
    # )
    
    # # Set the hierarchy from the previous run
    # hierarchical_clustering.hierarchy = hierarchy
    # hierarchical_clustering.L = 3
    

