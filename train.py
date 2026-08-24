"""Train the learnable template through a frozen differentiable video surrogate."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from fot.data import ImageFolderDataset
from template_embedding import TemplateEmbedding


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--output", default="checkpoints/fot.pt")
    p.add_argument("--size", type=int, default=256); p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--lambda-recovery", type=float, default=1.0); args=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TemplateEmbedding(3,args.size,args.size).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    loader=DataLoader(ImageFolderDataset(args.data,args.size),batch_size=args.batch_size,shuffle=True)
    for epoch in range(args.epochs):
        for image in loader:
            image=image.to(device); protected=model(image)
            # VAE-like differentiable compression surrogate for inexpensive pretraining.
            low=torch.nn.functional.interpolate(protected,scale_factor=.125,mode="bilinear",align_corners=False)
            recovered=torch.nn.functional.interpolate(low,size=image.shape[-2:],mode="bilinear",align_corners=False)
            fidelity=torch.nn.functional.mse_loss(protected,image); recovery=torch.nn.functional.l1_loss(recovered,image)
            loss=fidelity+args.lambda_recovery*recovery; opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch={epoch+1} loss={loss.item():.6f}")
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"args":vars(args)},args.output)
if __name__=="__main__": main()
