import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import laspy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Dataset class for LiDAR ground classification
class LiDARGroundDataset(Dataset):
    def __init__(self, las_file, patch_radius=10, sample_size=1024, train=True):
        """
        Dataset for training ground vs. non-ground classification
        
        Args:
            las_file: Path to LAZ/LAS file
            patch_radius: Radius for local neighborhood (meters)
            sample_size: Number of points per neighborhood sample
            train: If True, return training set, else validation set
        """
        super().__init__()
        
        # Load the LiDAR file
        with laspy.open(las_file, laz_backend=laspy.LazBackend.Lazrs) as f:
            self.las = f.read()
        
        print(f"Loaded point cloud with {len(self.las.points)} points")
        
        # Extract coordinates
        self.coordinates = np.vstack((self.las.x, self.las.y, self.las.z)).transpose()
        
        # Normalize coordinates
        self.coord_min = np.min(self.coordinates, axis=0)
        self.coord_max = np.max(self.coordinates, axis=0)
        self.normalized_coords = (self.coordinates - self.coord_min) / (self.coord_max - self.coord_min + 1e-8)
        
        # Extract basic features
        self.features = np.zeros((len(self.las.points), 10))
        self.features[:, 0] = self.las.intensity / (np.max(self.las.intensity) + 1e-8)  # Normalized intensity
        self.features[:, 1] = self.las.return_number 
        self.features[:, 2] = self.las.number_of_returns
        self.features[:, 3] = self.las.scan_direction_flag
        self.features[:, 4] = self.las.edge_of_flight_line
        self.features[:, 5] = (self.las.scan_angle_rank + 90) / 180  # Normalize scan angle
        self.features[:, 6] = self.las.user_data
        
        # Add RGB features if available
        if hasattr(self.las, 'red'):
            max_val = max(np.max(self.las.red), np.max(self.las.green), np.max(self.las.blue))
            self.features[:, 7] = self.las.red / max_val
            self.features[:, 8] = self.las.green / max_val
            self.features[:, 9] = self.las.blue / max_val
        
        # Create ground truth labels (2 = ground in LAS standard)
        self.labels = np.zeros(len(self.las.points), dtype=np.int64)
        if hasattr(self.las, 'classification'):
            self.labels[self.las.classification == 2] = 1  # 1 = ground, 0 = not ground
            print(f"Found {np.sum(self.labels == 1)} ground points and {np.sum(self.labels == 0)} non-ground points")
        else:
            # If no classification exists, estimate ground based on height
            print("No classification found, generating temporary labels based on height")
            z_sorted = np.sort(self.coordinates[:, 2])
            z_threshold = z_sorted[int(0.2 * len(z_sorted))]  # Bottom 20% as ground estimate
            self.labels[self.coordinates[:, 2] <= z_threshold] = 1
        
        # Create geometric features (important for ground detection)
        self.geometric_features = self._compute_geometric_features(patch_radius)
        
        # Combine all features
        self.all_features = np.hstack([
            self.normalized_coords,  # 3 features
            self.features,           # 10 features  
            self.geometric_features  # 6 features
        ])
        
        # Create neighborhood samples
        self.neighborhoods = self._create_neighborhood_samples(sample_size)
        
        # Split into train/validation
        indices = np.arange(len(self.neighborhoods))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        self.sample_indices = train_indices if train else val_indices
        self.sample_size = sample_size
    
    def _compute_geometric_features(self, radius):
        """Compute geometric features for each point using local neighborhoods"""
        from sklearn.neighbors import KDTree
        
        # Build KD-Tree for efficient neighbor lookup
        tree = KDTree(self.coordinates)
        
        # Features to compute:
        # 1. Height difference from local minimum
        # 2. Local height variance
        # 3. Planarity feature (from eigenvalues)
        # 4. Surface variation
        # 5. Normal vector z-component
        # 6. Angle with vertical
        
        geometric_features = np.zeros((len(self.las.points), 6))
        
        batch_size = 10000
        for i in tqdm(range(0, len(self.coordinates), batch_size), desc="Computing geometric features"):
            end_idx = min(i + batch_size, len(self.coordinates))
            batch_coords = self.coordinates[i:end_idx]
            
            # Find neighbors within radius
            indices = tree.query_radius(batch_coords, r=radius)
            
            for j, neighbors in enumerate(indices):
                if len(neighbors) < 10:  # Skip if not enough neighbors
                    continue
                    
                # Get neighborhood
                neighborhood = self.coordinates[neighbors]
                
                # Height features
                heights = neighborhood[:, 2]
                local_min = np.min(heights)
                local_max = np.max(heights)
                height_range = max(local_max - local_min, 1e-6)
                
                # 1. Height difference from local minimum (normalized)
                geometric_features[i+j, 0] = (self.coordinates[i+j, 2] - local_min) / height_range
                
                # 2. Local height variance (normalized)
                geometric_features[i+j, 1] = np.var(heights) / (height_range**2)
                
                try:
                    # Center the points
                    centered = neighborhood - np.mean(neighborhood, axis=0)
                    
                    # Compute covariance matrix
                    cov = np.cov(centered.T)
                    
                    # Get eigenvalues and eigenvectors
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    
                    # Sort eigenvalues (ascending)
                    idx = eigenvalues.argsort()
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    
                    # Ensure eigenvalues are positive and normalize
                    eigenvalues = np.abs(eigenvalues)
                    sum_eigenvalues = np.sum(eigenvalues) + 1e-8
                    eigenvalues = eigenvalues / sum_eigenvalues
                    
                    # 3. Planarity: (λ2 - λ1) / λ3 (λ1 ≤ λ2 ≤ λ3)
                    geometric_features[i+j, 2] = (eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[2], 1e-6)
                    
                    # 4. Surface variation: λ1 / (λ1 + λ2 + λ3)
                    geometric_features[i+j, 3] = eigenvalues[0] / sum_eigenvalues
                    
                    # 5. Normal vector (smallest eigenvector)
                    normal = eigenvectors[:, 0]
                    
                    # Z component of normal (vertical = ground)
                    geometric_features[i+j, 4] = abs(normal[2])
                    
                    # 6. Angle with vertical (0° = vertical, 90° = horizontal)
                    angle = np.arccos(abs(normal[2])) * 180 / np.pi
                    geometric_features[i+j, 5] = angle / 90.0  # Normalize to [0,1]
                    
                except np.linalg.LinAlgError:
                    # Handle numerical issues
                    geometric_features[i+j, 2:] = 0
        
        return geometric_features
    
    def _create_neighborhood_samples(self, sample_size):
        """Create neighborhood samples for training with balanced ground/non-ground points"""
        from sklearn.neighbors import KDTree
        
        # Build KD-Tree
        tree = KDTree(self.coordinates)
        
        # Determine how many samples to create
        ground_ratio = np.mean(self.labels)
        n_samples = min(5000, len(self.las.points) // sample_size)
        
        print(f"Creating {n_samples} neighborhood samples...")
        
        # Sample center points with balance between ground/non-ground
        ground_indices = np.where(self.labels == 1)[0]
        non_ground_indices = np.where(self.labels == 0)[0]
        
        n_ground = min(n_samples // 2, len(ground_indices))
        n_nonground = min(n_samples - n_ground, len(non_ground_indices))
        
        ground_centers = np.random.choice(ground_indices, n_ground, replace=False)
        nonground_centers = np.random.choice(non_ground_indices, n_nonground, replace=False)
        
        centers = np.concatenate([ground_centers, nonground_centers])
        np.random.shuffle(centers)
        
        # Create neighborhoods
        neighborhoods = []
        for center in tqdm(centers, desc="Creating neighborhood samples"):
            # Get nearest neighbors
            _, indices = tree.query(self.coordinates[center:center+1], k=sample_size)
            neighborhoods.append(indices[0])
        
        return neighborhoods
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        neighborhood = self.neighborhoods[self.sample_indices[idx]]
        
        # Get features and labels
        features = self.all_features[neighborhood]
        labels = self.labels[neighborhood]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'indices': torch.tensor(neighborhood, dtype=torch.long)
        }

# Transformer model for point cloud classification
class GroundTransformer(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=128, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Initial projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_positional_encoding(1024, hidden_dim)
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)  # Binary classification: ground/not-ground
        )
    
    def _create_positional_encoding(self, max_seq_len, d_model):
        """Create sinusoidal positional encodings"""
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, x):
        """Forward pass through the transformer"""
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Classification head
        logits = self.classifier(x)
        
        return logits

# Training function
def train_ground_classifier(model, train_loader, val_loader, device, num_epochs=30):
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_f1 = 0.0
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits.reshape(-1, 2), labels.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(features)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(labels.cpu().numpy().flatten())
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        val_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        history['val_metrics'].append(val_metrics)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Update learning rate based on F1 score
        scheduler.step(f1)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_ground_classifier.pth")
            print(f"  Saved new best model with F1: {f1:.4f}")
    
    # Load best model for inference
    model.load_state_dict(torch.load("best_ground_classifier.pth"))
    
    return model, history

# Prediction function
def classify_ground_points(model, las_file, output_file, device):
    model = model.to(device)
    model.eval()
    
    # Load data
    with laspy.open(las_file, laz_backend=laspy.LazBackend.Lazrs) as f:
        las = f.read()
    
    # Extract and preprocess features
    # (Similar feature extraction as in dataset class)
    coordinates = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Normalize coordinates
    coord_min = np.min(coordinates, axis=0)
    coord_max = np.max(coordinates, axis=0)
    normalized_coords = (coordinates - coord_min) / (coord_max - coord_min + 1e-8)
    
    # Extract basic features
    features = np.zeros((len(las.points), 10))
    features[:, 0] = las.intensity / (np.max(las.intensity) + 1e-8)
    features[:, 1] = las.return_number 
    features[:, 2] = las.number_of_returns
    features[:, 3] = las.scan_direction_flag
    features[:, 4] = las.edge_of_flight_line
    features[:, 5] = (las.scan_angle_rank + 90) / 180
    features[:, 6] = las.user_data
    
    # Add RGB features if available
    if hasattr(las, 'red'):
        max_val = max(np.max(las.red), np.max(las.green), np.max(las.blue))
        features[:, 7] = las.red / max_val
        features[:, 8] = las.green / max_val
        features[:, 9] = las.blue / max_val
    
    # Compute geometric features using KNN for efficiency
    print("Computing geometric features for prediction...")
    from sklearn.neighbors import KDTree
    tree = KDTree(coordinates)
    
    geometric_features = np.zeros((len(las.points), 6))
    
    batch_size = 10000
    for i in tqdm(range(0, len(coordinates), batch_size)):
        end_idx = min(i + batch_size, len(coordinates))
        batch_coords = coordinates[i:end_idx]
        
        # Use k-nearest neighbors for prediction
        _, indices = tree.query(batch_coords, k=20)
        
        for j, neighbors in enumerate(indices):
            # Get neighborhood
            neighborhood = coordinates[neighbors]
            
            # Height features
            heights = neighborhood[:, 2]
            local_min = np.min(heights)
            local_max = np.max(heights)
            height_range = max(local_max - local_min, 1e-6)
            
            geometric_features[i+j, 0] = (coordinates[i+j, 2] - local_min) / height_range
            geometric_features[i+j, 1] = np.var(heights) / (height_range**2)
            
            try:
                # PCA-based features
                centered = neighborhood - np.mean(neighborhood, axis=0)
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                idx = eigenvalues.argsort()
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                eigenvalues = np.abs(eigenvalues)
                sum_eigenvalues = np.sum(eigenvalues) + 1e-8
                eigenvalues = eigenvalues / sum_eigenvalues
                
                geometric_features[i+j, 2] = (eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[2], 1e-6)
                geometric_features[i+j, 3] = eigenvalues[0] / sum_eigenvalues
                
                normal = eigenvectors[:, 0]
                geometric_features[i+j, 4] = abs(normal[2])
                
                angle = np.arccos(abs(normal[2])) * 180 / np.pi
                geometric_features[i+j, 5] = angle / 90.0
                
            except np.linalg.LinAlgError:
                geometric_features[i+j, 2:] = 0
    
    # Combine all features
    all_features = np.hstack([
        normalized_coords,
        features,
        geometric_features
    ])
    
    # Predict in batches using KNN context
    print("Predicting ground points...")
    predictions = np.zeros(len(las.points))
    
    batch_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, len(all_features), batch_size)):
            end_idx = min(i + batch_size, len(all_features))
            batch_features = all_features[i:end_idx]
            
            # Create neighborhood context for each point
            # For prediction, we use a simpler flat-batch approach
            features_tensor = torch.tensor(batch_features, dtype=torch.float32).to(device)
            
            # Predict
            logits = model(features_tensor.unsqueeze(0)).squeeze(0)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # Probability of ground class
            
            predictions[i:end_idx] = probs
    
    # Convert probabilities to binary predictions
    binary_predictions = (predictions > 0.5).astype(np.int32)
    
    # Create output file
    output_las = las.copy()
    
    # Update classification (2 = ground in LAS spec)
    output_las.classification[binary_predictions == 1] = 2
    output_las.classification[binary_predictions == 0] = 1  # Unclassified
    
    # Save file
    output_las.write(output_file)
    
    print(f"Classification complete: {np.sum(binary_predictions == 1)} ground points identified ({np.sum(binary_predictions == 1)/len(binary_predictions)*100:.2f}%)")
    
    return binary_predictions, output_las

# Main function to run the complete pipeline
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # File paths
    input_file = '/Volumes/Fangorn/wetransfer_lidar-samples_2025-04-23_1749/05_grnd_sample/687400_5577000.laz'
    output_file = 'ground_classified.laz'
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = LiDARGroundDataset(input_file, train=True)
    val_dataset = LiDARGroundDataset(input_file, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Input dimension from dataset
    input_dim = train_dataset.all_features.shape[1]
    
    # Create transformer model
    print(f"Creating model with input dimension: {input_dim}...")
    model = GroundTransformer(input_dim=input_dim, hidden_dim=128, num_heads=4, num_layers=3)
    
    # Train model
    print("Training model...")
    trained_model, history = train_ground_classifier(
        model, train_loader, val_loader, device, num_epochs=20
    )
    
    # Classify points
    print("Running inference on input data...")
    predictions, output_las = classify_ground_points(
        trained_model, input_file, output_file, device
    )
    
    # Visualize results
    visualize_results(input_file, output_file, predictions)
    
    return trained_model, history

def visualize_results(input_file, output_file, predictions):
    """Visualize the ground classification results"""
    # Load files
    with laspy.open(input_file, laz_backend=laspy.LazBackend.Lazrs) as f:
        input_las = f.read()
    
    with laspy.open(output_file, laz_backend=laspy.LazBackend.Lazrs) as f:
        output_las = f.read()
    
    # Sample points for visualization
    sample_size = min(20000, len(input_las.points))
    sample_indices = np.random.choice(len(input_las.points), sample_size, replace=False)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Original point cloud colored by height
    plt.subplot(221)
    plt.scatter(
        input_las.x[sample_indices], 
        input_las.y[sample_indices],
        c=input_las.z[sample_indices],
        cmap='terrain',
        s=0.5
    )
    plt.colorbar(label='Elevation (m)')
    plt.title('Original Point Cloud (by Elevation)')
    plt.axis('equal')
    
    # Original classification
    plt.subplot(222)
    if hasattr(input_las, 'classification'):
        plt.scatter(
            input_las.x[sample_indices], 
            input_las.y[sample_indices],
            c=input_las.classification[sample_indices],
            cmap='tab10',
            s=0.5
        )
        plt.colorbar(label='Original Class')
        plt.title('Original Classification')
    else:
        plt.text(0.5, 0.5, 'No classification data available', 
                 ha='center', va='center', transform=plt.gca().transAxes)
    plt.axis('equal')
    
    # New classification
    plt.subplot(223)
    plt.scatter(
        output_las.x[sample_indices], 
        output_las.y[sample_indices],
        c=output_las.classification[sample_indices],
        cmap='tab10',
        s=0.5
    )
    plt.colorbar(label='New Class')
    plt.title('Transformer Classified Points')
    plt.axis('equal')
    
    # 3D view of ground points
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(224, projection='3d')
    ground_indices = sample_indices[output_las.classification[sample_indices] == 2]
    
    # Show only ground points colored by elevation
    ax.scatter(
        output_las.x[ground_indices],
        output_las.y[ground_indices],
        output_las.z[ground_indices],
        c=output_las.z[ground_indices],
        cmap='terrain',
        s=0.5
    )
    ax.set_title('3D View of Predicted Ground Points')
    
    plt.tight_layout()
    plt.savefig('ground_classification_results.png', dpi=300)
    plt.show()
    
    print("Visualization saved to ground_classification_results.png")

if __name__ == "__main__":
    main()