import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from transformers import AutoTokenizer, AutoModel

# Step 1: Load your DataFrame
df = pd.read_pickle('../final_draft/df.pkl')
df['report'] = df['PNT_NM'] + df['QUALIFIER_TXT'] + df['PNT_ATRISKNOTES_TX'] + df['PNT_ATRISKFOLWUPNTS_TX']

# Assuming df is your DataFrame
reports = df['report'].values
ovr_danger = df['ovr_danger'].values

# Step 2: Load the transformer model and tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Choose a model that suits your needs
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Step 3: Generate embeddings
def get_embeddings(reports):
    inputs = tokenizer(reports.tolist(), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # Use the [CLS] token representation
    return embeddings.numpy()

embeddings = get_embeddings(reports)

# Step 4: Prepare data for t-SNE
tsne = TSNE(n_components=2, random_state=42)
embedded_2d = tsne.fit_transform(embeddings)

# Step 5: Create a scatter plot
plt.figure(figsize=(10, 8))
#cmap = mcolors.LinearSegmentedColormap.from_list('red_green', ['green', 'yellow', 'red'])
scatter = plt.scatter(embedded_2d[:, 0], embedded_2d[:, 1], c=ovr_danger, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Danger Magnitude')
plt.title('t-SNE Visualization of Reports by Danger Magnitude')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
