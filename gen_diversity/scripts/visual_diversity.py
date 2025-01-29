import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ToTensor

from gen_diversity.dataset.gensim import GensimDataset

# images = torch.rand(400, 3, 128, 128)

@torch.no_grad()
def compute_pairwise_similarity(bsize=128):
    # compute the pairwise similarity
    # in an efficient way
    # without blowing up memory
    similarity = torch.zeros((len(images), len(images)))
    for i in tqdm(range(len(images))):
        image_i = images[i].unsqueeze(0).to("cuda")
        
        for j in range(i + 1, len(images), bsize):

            similarity[i, j:min(j + bsize, len(images))] = lpip(image_i, images[j:min(j + bsize, len(images))].to("cuda"))
    # return the lower triangle
    return similarity.triu(diagonal=1)


if __name__ == "__main__":
    group = "all"
    net_type = "vgg"
    print(group, net_type)
    path = f"/localdata/bxu/isaac_lab/GenDiversity/projects/GenSim/data/train/{group}"
    dataset = GensimDataset.load(path, 40, high_level=False, max_episodes=40)
    lpip = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to("cuda")

    torch.manual_seed(0)
    i = torch.randint(len(dataset), (1000,))
    images = dataset.data["image"][i] # [N, C, H, W]
    images = (images[:, :3] / 255) * 2 - 1

    similarity = compute_pairwise_similarity().cpu()
    torch.save(similarity, "similarity_vgg.pt")
    print(similarity.std())
    print(similarity.mean())
