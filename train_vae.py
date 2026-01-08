import argparse
import torch
import torch.optim as optim
from pathlib import Path
from torchvision import transforms
import wandb

from src.vision_vae import VisionVAE
from src.data import ImageDataLoader


def train_vae(
    data_root: str,
    output_dir: str = "checkpoints",
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    latent_dim: int = 10,
    beta: float = 1.0,
    device: str = None,
    num_workers: int = 0,
    use_wandb: bool = True,
    wandb_project: str = "unipd-vision-vae",
    wandb_entity: str = "frederikdammeier-leipzig-university",
):
    """
    Train a Vision VAE model.
    
    Args:
        data_root: Path to the root directory containing class folders
        output_dir: Directory to save checkpoints
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        latent_dim: Size of the latent space
        beta: Weight for KL divergence term in loss
        device: Device to train on ('cuda' or 'cpu'). Auto-detect if None.
        num_workers: Number of worker processes for data loading
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name (username or team)
    """
    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "latent_dim": latent_dim,
                "beta": beta,
                "num_workers": num_workers,
            },
        )
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create data transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    # normalize beta
    beta_norm = beta / ((64 * 64) / latent_dim)
    print(f"Normalized beta: {beta_norm}")

    # Create dataloader
    print(f"Loading data from {data_root}")
    dataloader = ImageDataLoader.create_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        transform=transform,
    )
    print(f"Loaded {len(dataloader.dataset)} images")
    
    # Create model
    model = VisionVAE(latent_dim=latent_dim, beta=beta_norm).to(device)
    print(f"Created VisionVAE with latent_dim={latent_dim}, beta={beta_norm}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            recon_images, mu, logvar, z = model(images)
            
            # Compute loss
            loss, recon_loss, kl_loss = model.loss_function(recon_images, images, mu, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f})")
        
        # Log to Weights & Biases
        if use_wandb:
            # Convert tensors to images for visualization
            ground_truth_img = transforms.ToPILImage()(images[0].cpu())
            reconstructed_img = transforms.ToPILImage()(recon_images[0].cpu().detach())
            
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "recon_loss": avg_recon_loss,
                "kl_loss": avg_kl_loss,
                "ground_truth": wandb.Image(ground_truth_img),
                "reconstructed": wandb.Image(reconstructed_img),
            })
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(output_dir) / f"vae_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = Path(output_dir) / "vae_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
    
    # Finish W&B run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision VAE")
    parser.add_argument("data_root", type=str, help="Path to data root directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=10, help="Latent dimension")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for KL divergence weighting")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="unipd-vision-vae", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default="frederikdammeier-leipzig-university", help="W&B entity name")
    
    args = parser.parse_args()
    
    train_vae(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        beta=args.beta,
        device=args.device,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
