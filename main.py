import argparse
import glob
import numpy as np
import tifffile
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

from data import Dataset
from models import ConvAE

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--path_data',default = "../Images") # path of input images
    parser.add_argument('--n_sample',type = int, default = 1000) # no. samples per image
    parser.add_argument('--size_image',type = int, default = 512)
    parser.add_argument('--size_patch',type = int, default = 64)
    parser.add_argument('--size_batch_train',type = int, default = 64)
    parser.add_argument('--size_batch_infer',type = int, default = 64)
    parser.add_argument('--n_epoch',type = int, default = 10)
    parser.add_argument('--intensity_max_DNA',type = int, default = 4096) # value to divide for DNA channel
    parser.add_argument('--intensity_max_actin',type = int, default = 1024) # value to divide for actin channel
    parser.add_argument('--nn_n_channels',type = int, default = 16) # no. channels for the conv / tconv layers
    parser.add_argument('--nn_dim_latent',type = int, default = 8) # latent dimension
    parser.add_argument('--thres_patch_fg',type = float, default = 0.0625) # the smallest amount of foreground to keep the patch
    parser.add_argument('--thres_patch_mito',type = float, default = 2.0) # max ratio of mean pH3 intensity between patch and image
    return parser.parse_args()


def get_image_filenames(path0):
    img_l = glob.glob(path0+"/*.tif")
    for i in range(len(img_l)):
        img_l[i] = img_l[i][len(path0)+1:].split("-")[0]
    img_l = list(set(img_l))
    img_l.sort()
    return img_l


def prepare_dataset(args):
    img_l = get_image_filenames(args.path_data)
    I = np.zeros([3,len(img_l),args.size_image,args.size_image],dtype = np.uint16)
    for i in range(len(img_l)):
        I[0,i] = tifffile.imread(args.path_data+"/"+img_l[i]+"-DNA.tif")
        I[1,i] = tifffile.imread(args.path_data+"/"+img_l[i]+"-pH3.tif")
        I[2,i] = tifffile.imread(args.path_data+"/"+img_l[i]+"-actin.tif")
    dataset = Dataset(args,I)
    return dataset

def train(args,dataset):
    dataloader_train = DataLoader(dataset,batch_size=args.size_batch_train, shuffle=True, drop_last=True)
    model = ConvAE(args)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True, betas=(0.9, 0.999))
    record_loss = np.zeros([args.n_epoch * len(dataloader_train)],dtype = np.float32)
    for e in range(args.n_epoch):
        for i,data_batch in enumerate(dataloader_train):
            optimizer.zero_grad()
            x,_ = data_batch
            x = x.to(device)
            x_hat = model(x)
            loss_function = nn.MSELoss()
            loss_batch = loss_function(x_hat,x)
            record_loss[e*len(dataloader_train)+i] = loss_batch.item()
            loss_batch.backward()
            optimizer.step()
            print(loss_batch)


    torch.save(model.state_dict(), "model.pt")

    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(record_loss,linewidth=0.25)
    #plt.yscale("log")
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.savefig("loss.pdf", bbox_inches="tight")



def infer(args,dataset):
    dataloader_infer = DataLoader(dataset,batch_size=args.size_batch_infer, shuffle=False, drop_last=True)
    model = ConvAE(args)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    representation = torch.zeros((len(dataloader_infer)*args.size_batch_infer,args.nn_dim_latent), dtype = torch.float32)
    patch_sources  = torch.zeros((len(dataloader_infer)*args.size_batch_infer), dtype = torch.int64)
    with torch.no_grad():
        for i,data_batch in enumerate(dataloader_infer):
            x,image_id = data_batch
            x = x.to(device)
            y = model.encoder(x)
            representation[i*args.size_batch_infer:(i+1)*args.size_batch_infer,:] = y.detach().cpu()
            patch_sources[ i*args.size_batch_infer:(i+1)*args.size_batch_infer] = image_id.detach().cpu()
    torch.save(representation,"representation.pt")
    torch.save(patch_sources,"patch_sources.pt")


def cluster(args):
    representation = torch.load("representation.pt")
    patch_sources = torch.load("patch_sources.pt").numpy()
    y = MiniBatchKMeans(n_clusters=14, random_state=0).fit_predict(representation)
    result = np.zeros([56],dtype=np.int64)
    for i in range(56):
        temp = y[np.where(patch_sources == i)]
        values, counts = np.unique(temp, return_counts=True)
        ind = np.argmax(counts)
        result[i]=values[ind]
    img_l = get_image_filenames(args.path_data)
    print(img_l)
    print(result)



if __name__ == "__main__":
    args = parse_args()

    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    dataset = prepare_dataset(args)
    train(args,dataset)
    infer(args,dataset)
    cluster(args)

