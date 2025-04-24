import laspy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from tqdm import tqdm

# Safer dataset class for LiDAR ground classification
class LiDARGroundDataset(Dataset):
    def __init__(self, las_file, patch_size=20, train=True):
        """
        Dataset for training ground vs. non-ground classification
        """
        # Load the LiDAR file
        with laspy.open(las_file, laz_backend=laspy.LazBackend.Lazrs) as f:
            self.las = f.read()
        
        print(f"Loaded point cloud with {len(self.las.points)} points")
        
        # Get basic point data - these should always exist
        self.x = np.array(self.las.x)
        self.y = np.array(self.las.y)
        self.z = np.array(self.las.z)
        
        # Extract coordinates
        self.coordinates = np.vstack((self.x, self.y, self.z)).transpose()
        
        # Normalize coordinates for better neural network performance
        self.coord_min = np.min(self.coordinates, axis=0)
        self.coord_max = np.max(self.coordinates, axis=0)
        self.normalized_coords = (self.coordinates - self.coord_min) / (self.coord_max - self.coord_min + 1e-8)
        
        # Safely extract other attributes
        self.features = np.zeros((len(self.coordinates), 10))
        
        # Intensity (if available)
        if hasattr(self.las, 'intensity'):
            intensity = np.array(self.las.intensity)
            intensity_max = np.max(intensity) if len(intensity) > 0 else 1
            self.features[:, 0] = intensity / (intensity_max + 1e-8)
        
        # Return number and other attributes (safely check each)
        if hasattr(self.las, 'return_number'):
            self.features[:, 1] = np.array(self.las.return_number)
        
        if hasattr(self.las, 'number_of_returns'):
            self.features[:, 2] = np.array(self.las.number_of_returns)
            
        if hasattr(self.las, 'scan_direction_flag'):
            self.features[:, 3] = np.array(self.las.scan_direction_flag)
            
        if hasattr(self.las, 'edge_of_flight_line'):
            self.features[:, 4] = np.array(self.las.edge_of_flight_line)
            
        if hasattr(self.las, 'scan_angle_rank'):
            angle = np.array(self.las.scan_angle_rank)
            self.features[:, 5] = (angle + 90) / 180  # Normalize
            
        if hasattr(self.las, 'user_data'):
            self.features[:, 6] = np.array(self.las.user_data)
        
        # Add RGB features if available
        rgb_available = all(hasattr(self.las, attr) for attr in ['red', 'green', 'blue'])
        if rgb_available:
            red = np.array(self.las.red)
            green = np.array(self.las.green)
            blue = np.array(self.las.blue)
            max_val = max(np.max(red), np.max(green), np.max(blue)) if len(red) > 0 else 1
            self.features[:, 7] = red / max_val
            self.features[:, 8] = green / max_val
            self.features[:, 9] = blue / max_val
        
        # Create ground truth labels (2 = ground in LAS standard)
        self.labels = np.zeros(len(self.coordinates), dtype=np.int64)
        if hasattr(self.las, 'classification'):
            # Convert to numpy array to avoid laspy recursion issues
            classification = np.array(self.las.classification)
            self.labels[classification == 2] = 1  # 1 = ground, 0 = not ground
            print(f"Found {np.sum(self.labels == 1)} ground points and {np.sum(self.labels == 0)} non-ground points")
        else:
            # If no classification exists, estimate ground based on height
            print("No classification found, generating temporary labels based on height")
            z_sorted = np.sort(self.z)
            z_threshold = z_sorted[int(0.2 * len(z_sorted))]  # Bottom 20% as ground estimate
            self.labels[self.z <= z_threshold] = 1
        
        # Compute simplified geometric features
        self.geometric_features = self._compute_simplified_geometric_features()
        
        # Combine all features
        self.all_features = np.hstack([
            self.normalized_coords,       # 3 features
            self.features,                # 10 features  
            self.geometric_features       # 3 features
        ])
        
        # Create patch indices
        tree = KDTree(self.coordinates)
        
        # Sample some center points
        total_samples = min(5000, len(self.coordinates))
        ground_indices = np.where(self.labels == 1)[0]
        non_ground_indices = np.where(self.labels == 0)[0]
        
        # Balance ground and non-ground samples
        n_ground = min(total_samples // 2, len(ground_indices))
        n_non_ground = min(total_samples - n_ground, len(non_ground_indices))
        
        center_ground = np.random.choice(ground_indices, n_ground, replace=False)
        center_non_ground = np.random.choice(non_ground_indices, n_non_ground, replace=False)
        
        all_centers = np.concatenate([center_ground, center_non_ground])
        np.random.shuffle(all_centers)
        
        # For each center, find neighbors
        self.patches = []
        for center in all_centers:
            _, indices = tree.query(self.coordinates[center:center+1], k=patch_size)
            self.patches.append(indices[0])
        
        # Split into train and validation
        train_idx, val_idx = train_test_split(
            np.arange(len(self.patches)), 
            test_size=0.2, 
            random_state=42
        )
        
        self.indices = train_idx if train else val_idx
    
    def _compute_simplified_geometric_features(self):
        """Compute simplified geometric features (avoiding complex calculations that might cause issues)"""
        # Create a KD-tree for efficient neighbor lookups
        tree = KDTree(self.coordinates)
        
        # Compute 3 simple geometric features:
        # 1. Local height difference (from min height in neighborhood)
        # 2. Local height standard deviation
        # 3. Slope estimate
        features = np.zeros((len(self.coordinates), 3))
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        for i in tqdm(range(0, len(self.coordinates), batch_size), desc="Computing geometric features"):
            end_idx = min(i + batch_size, len(self.coordinates))
            batch_coords = self.coordinates[i:end_idx]
            
            # Find 10 nearest neighbors for each point
            _, indices = tree.query(batch_coords, k=10)
            
            for j, neighbors in enumerate(indices):
                # Get heights of neighbors
                neighbor_heights = self.z[neighbors]
                
                # Feature 1: Height from local minimum
                local_min = np.min(neighbor_heights)
                local_max = np.max(neighbor_heights)
                height_range = max(local_max - local_min, 1e-6)
                features[i+j, 0] = (self.z[i+j] - local_min) / height_range
                
                # Feature 2: Local height variation
                features[i+j, 1] = np.std(neighbor_heights) / height_range
                
                # Feature 3: Simple slope estimate (max height difference / max xy distance)
                neighbor_coords = self.coordinates[neighbors]
                center = self.coordinates[i+j]
                
                # Calculate distances in xy plane
                dx = neighbor_coords[:, 0] - center[0]
                dy = neighbor_coords[:, 1] - center[1]
                dist_xy = np.sqrt(dx**2 + dy**2)
                
                if np.max(dist_xy) > 0:
                    max_slope = height_range / np.max(dist_xy)
                    features[i+j, 2] = np.arctan(max_slope) / (np.pi/2)  # Normalize to [0,1]
                else:
                    features[i+j, 2] = 0
        
        return features
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index
        real_idx = self.indices[idx]
        patch_indices = self.patches[real_idx]
        
        # Get features and labels for this patch
        features = self.all_features[patch_indices]
        labels = self.labels[patch_indices]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'indices': torch.tensor(patch_indices, dtype=torch.long)
        }

# Transformer model for point cloud classification
class GroundTransformer(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Initial projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
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
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)
        x = self.transformer(x)
        logits = self.classifier(x)
        return logits

# Training function
def train_ground_classifier(model, train_loader, val_loader, device, num_epochs=10):
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training loop
    best_acc = 0.0
    
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
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(features)
                preds = torch.argmax(logits, dim=-1)
                
                correct += (preds == labels).sum().item()
                total += labels.numel()
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Validation Accuracy: {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "best_ground_classifier.pth")
            print(f"  Saved new best model with accuracy: {accuracy:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load("best_ground_classifier.pth"))
    return model

# Function to classify new points
def classify_point_cloud(model, las_file, output_file, device):
    model = model.to(device)
    model.eval()
    
    # Load data
    with laspy.open(las_file, laz_backend=laspy.LazBackend.Lazrs) as f:
        las = f.read()
    
    # Get coordinates
    coords = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Create dataset with same preprocessing as training
    features = extract_features(las, coords)
    
    # Create a new LasData for output (compatible with all laspy versions)
    output_las = laspy.LasData(las.header)
    
    # Copy all point records from the original
    output_las.points = las.points.copy()
    
    # Classify in batches
    batch_size = 1024
    all_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(features), batch_size)):
            end = min(i + batch_size, len(features))
            batch = features[i:end]
            
            # Convert to tensor
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            
            # Get predictions
            logits = model(batch_tensor.unsqueeze(0)).squeeze(0)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            
            all_probs.append(probs)
    
    # Combine all probabilities
    all_probs = np.concatenate(all_probs)
    
    # Convert to binary classification (threshold at 0.5)
    ground_mask = all_probs > 0.5
    
    # Update classification
    output_las.classification[ground_mask] = 2  # Ground
    output_las.classification[~ground_mask] = 1  # Non-ground
    
    # Save the output file
    output_las.write(output_file)
    
    print(f"Classification complete: {np.sum(ground_mask)} ground points ({np.sum(ground_mask)/len(ground_mask)*100:.2f}%), {np.sum(~ground_mask)} non-ground points ({np.sum(~ground_mask)/len(ground_mask)*100:.2f}%)")
    return output_las

# Helper function to extract features
def extract_features(las, coords):
    # Normalized coordinates
    coord_min = np.min(coords, axis=0)
    coord_max = np.max(coords, axis=0)
    normalized_coords = (coords - coord_min) / (coord_max - coord_min + 1e-8)
    
    # Basic features
    features = np.zeros((len(coords), 10))
    
    # Safely extract intensity
    if hasattr(las, 'intensity'):
        intensity = np.array(las.intensity)
        max_intensity = np.max(intensity) if len(intensity) > 0 else 1
        features[:, 0] = intensity / (max_intensity + 1e-8)
    
    # Other attributes
    for i, attr in enumerate(['return_number', 'number_of_returns', 
                             'scan_direction_flag', 'edge_of_flight_line']):
        if hasattr(las, attr):
            features[:, i+1] = np.array(getattr(las, attr))
    
    # Scan angle
    if hasattr(las, 'scan_angle_rank'):
        angle = np.array(las.scan_angle_rank)
        features[:, 5] = (angle + 90) / 180
    
    # User data
    if hasattr(las, 'user_data'):
        features[:, 6] = np.array(las.user_data)
    
    # RGB if available
    if all(hasattr(las, attr) for attr in ['red', 'green', 'blue']):
        red = np.array(las.red)
        green = np.array(las.green)
        blue = np.array(las.blue)
        max_val = max(np.max(red), np.max(green), np.max(blue)) if len(red) > 0 else 1
        features[:, 7] = red / max_val
        features[:, 8] = green / max_val
        features[:, 9] = blue / max_val
    
    # Geometric features (simplified)
    geo_features = compute_geometric_features(coords)
    
    # Combine all features
    return np.hstack([normalized_coords, features, geo_features])

# Compute geometric features
def compute_geometric_features(coords):
    tree = KDTree(coords)
    features = np.zeros((len(coords), 3))
    
    batch_size = 10000
    for i in tqdm(range(0, len(coords), batch_size), desc="Computing geometric features"):
        end = min(i + batch_size, len(coords))
        batch = coords[i:end]
        
        # Get 10 nearest neighbors
        _, indices = tree.query(batch, k=10)
        
        for j, neighbors in enumerate(indices):
            # Heights
            neighbor_heights = coords[neighbors, 2]
            local_min = np.min(neighbor_heights)
            local_max = np.max(neighbor_heights)
            height_range = max(local_max - local_min, 1e-6)
            
            # Height difference
            features[i+j, 0] = (coords[i+j, 2] - local_min) / height_range
            
            # Height variation
            features[i+j, 1] = np.std(neighbor_heights) / height_range
            
            # Slope estimate
            dx = coords[neighbors, 0] - coords[i+j, 0]
            dy = coords[neighbors, 1] - coords[i+j, 1]
            dist_xy = np.sqrt(dx**2 + dy**2)
            
            if np.max(dist_xy) > 0:
                max_slope = height_range / np.max(dist_xy)
                features[i+j, 2] = np.arctan(max_slope) / (np.pi/2)
            else:
                features[i+j, 2] = 0
    
    return features

# Main function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # File paths
    input_file = '/Volumes/Fangorn/lidar_analysis/05_grnd_sample/687400_5577000.laz'
    output_file = 'ground_classified.laz'
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = LiDARGroundDataset(input_file, train=True)
    val_dataset = LiDARGroundDataset(input_file, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Get input dimension
    input_dim = train_dataset.all_features.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = GroundTransformer(input_dim=input_dim)
    
    # Train model
    print("Training model...")
    trained_model = train_ground_classifier(
        model, train_loader, val_loader, device, num_epochs=30
    )
    
    # Classify points
    print("Classifying point cloud...")
    output_las = classify_point_cloud(
        trained_model, input_file, output_file, device
    )
    
    # Visualize results
    visualize_results(input_file, output_file)
    
    return trained_model

# Visualization function
def visualize_results(input_file, output_file):
    # Load original and classified point clouds
    with laspy.open(input_file, laz_backend=laspy.LazBackend.Lazrs) as f:
        original = f.read()
    
    with laspy.open(output_file, laz_backend=laspy.LazBackend.Lazrs) as f:
        classified = f.read()
    
    # Sample points for visualization
    sample_size = min(10000, len(original.points))
    sample_indices = np.random.choice(len(original.points), sample_size, replace=False)
    
    # Create figures
    plt.figure(figsize=(15, 10))
    
    # Original elevation
    plt.subplot(221)
    plt.scatter(
        original.x[sample_indices], 
        original.y[sample_indices],
        c=original.z[sample_indices],
        cmap='terrain',
        s=1
    )
    plt.colorbar(label='Elevation')
    plt.title('Original Point Cloud (Elevation)')
    plt.axis('equal')
    
    # Classified points (ground/non-ground)
    plt.subplot(222)
    plt.scatter(
        classified.x[sample_indices], 
        classified.y[sample_indices],
        c=classified.classification[sample_indices],
        cmap='viridis',
        s=1
    )
    plt.colorbar(label='Class')
    plt.title('Ground Classification')
    plt.axis('equal')
    
    # Ground points only
    ground_indices = sample_indices[classified.classification[sample_indices] == 2]
    
    plt.subplot(223)
    if len(ground_indices) > 0:
        plt.scatter(
            classified.x[ground_indices], 
            classified.y[ground_indices],
            c=classified.z[ground_indices],
            cmap='terrain',
            s=1
        )
        plt.colorbar(label='Elevation')
    plt.title('Ground Points Only')
    plt.axis('equal')
    
    # Non-ground points only
    nonground_indices = sample_indices[classified.classification[sample_indices] != 2]
    
    plt.subplot(224)
    if len(nonground_indices) > 0:
        plt.scatter(
            classified.x[nonground_indices], 
            classified.y[nonground_indices],
            c=classified.z[nonground_indices],
            cmap='terrain',
            s=1
        )
        plt.colorbar(label='Elevation')
    plt.title('Non-Ground Points Only')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('ground_classification_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()