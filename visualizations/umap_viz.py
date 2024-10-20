import umap
import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# Step 4: Prepare data for UMAP
umap_model = umap.UMAP(n_components=3, random_state=42)
embedded_3d = umap_model.fit_transform(embeddings)

# Step 5: Create a 3D scatter plot using UMAP results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(embedded_3d[:, 0], embedded_3d[:, 1], embedded_3d[:, 2], c=ovr_danger, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Danger Magnitude')
ax.set_title('UMAP Visualization of Reports by Danger Magnitude')
ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')
ax.set_zlabel('UMAP Component 3')
plt.show()
