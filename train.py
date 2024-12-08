import torch
import sys
import os
import pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from models.GCN_model import *
from data.Graph import *
from data.Preprocessing import *
from utils.Visualization import *


def train_gcn(model, train_loader, val_loader, criterion, optimizer, 
              device, epochs=5, early_stopping_patience=10):
    """
    Training function for the GCN model
    
    Args:
        model (torch.nn.Module): GCN model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimization algorithm
        device (torch.device): Computing device
        epochs (int): Number of training epochs
        early_stopping_patience (int): Epochs to wait for improvement
    
    Returns:
        torch.nn.Module: Trained model
    """
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            batch = batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                val_loss = criterion(outputs, batch.y)
                total_val_loss += val_loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_gcn_model.pth')
        else:
            patience_counter += 1
        
        # Stop training if no improvement
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_gcn_model.pth'))
    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on test data
    
    Args:
        model (torch.nn.Module): Trained GCN model
        test_loader (DataLoader): Test data loader
        device (torch.device): Computing device
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            batch = batch.to(device)
            outputs = model(batch)
            preds = (outputs > 0.5).float()
            
            all_preds.append(preds)
            all_labels.append(batch.y)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    print("First 40 predictions")
    print(all_preds[:40])
    print("First 40 labels")
    print(all_labels[:40])
    # Calculate metrics
    accuracy = (all_preds == all_labels).float().mean()
    precision = (all_preds[all_preds == 1] == all_labels[all_preds == 1]).float().mean()
    recall = (all_preds[all_labels == 1] == all_labels[all_labels == 1]).float().mean()
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def model_infer(model, test_loader, device):
    """
    Evaluate the trained model on test data
    
    Args:
        model (torch.nn.Module): Trained GCN model
        test_loader (DataLoader): Test data loader
        device (torch.device): Computing device
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            batch = batch.to(device)
            outputs = model(batch)
            preds = (outputs > 0.5).float()
            
            all_preds.append(preds)
    
    all_preds = torch.cat(all_preds)
    total_preds = len(all_preds)
    link_percentage = (all_preds.sum() / total_preds) * 100
    non_link_percentage = 100 - link_percentage

    print(f"Percentage of links: {link_percentage:.2f}%")
    print(f"Percentage of non-links: {non_link_percentage:.2f}%")
    return all_preds

# Example usage
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    df_links = preprocessing_links(data_path)

    embeddings_path = os.path.join(data_path, 'embeddings.pkl')
    if os.path.exists(embeddings_path):
        df = pd.read_pickle(embeddings_path)
        embedded_articles = dict(zip(df['Article_Title'], zip(df['Article_Title_embedding'], df['Description_embedding'] )))
    else:
        print("Couldn't find the embeddings")

    if os.path.exists('graph_dataset.pkl') and os.path.exists('candidates_dataset.pkl'):
        with open('graph_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)

        with open('candidates_dataset.pkl', 'rb') as f:
            candidates_dataset = pickle.load(f)

    else:
        print("Creating Graph")
        G=create_graph(embedded_articles, df_links)
        similarities = create_node_similarity_distributions(G)
        candidates, zero_label_non_links = calculate_negative_likelihood_and_labels(G, similarities)

        data_loader = GraphDataLoader(G, candidates, zero_label_non_links)
        dataset, candidates_dataset = data_loader.create_pyg_dataset()
        with open('graph_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        with open('candidates_dataset.pkl', 'wb') as f:
            pickle.dump(candidates_dataset, f)
    

    dataset = dataset.to(device, non_blocking=True)

    print("Making data loaders")
    train_loader, val_loader, test_loader, candidates_loader = create_graph_dataloaders(dataset, candidates_dataset)
    
    # Initialize model
    print("Initializing model")
    model = EdgeClassificationGCNWrapper().to(device)
    
    # Loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print("Starting training")
    # model = train_gcn(
    #     model, 
    #     train_loader, 
    #     val_loader, 
    #     criterion, 
    #     optimizer, 
    #     device
    # )
    
    # If model already trained, load it
    model.load_state_dict(torch.load('best_gcn_model.pth', weights_only=True))

    # Evaluate on test set
    print("evaluating on test set")
    metrics = evaluate_model(model, test_loader, device)
    print("Test Metrics:", metrics)

    print("Evaluating on the candidates set")
    link_predictions = model_infer(model, candidates_loader, device)

if __name__ == '__main__':
    main()
