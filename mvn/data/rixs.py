import os

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import radius_graph
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



##############################################################################
## dataset and dataloader class for RIXS. for custom data, put it in the class
## RIXS in the RISXDataset class in dataset_name. Load in self.load using np.load
## (see below for structure) in the __init__ and then accordingly in the 
##  __getitem__ . 
##############################################################################



DATAROOT = os.environ["DATAROOT"]

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        keylist = ["spectrum"]
        if key in keylist:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)



class RIXSDataset:
    """
    NBodyDataset

    """

    def __init__(
        self, dataroot, partition="train", num_sample=1e8, dataset_name=None, cutoff=5.0, eps=1e-12, 
        logtransform=True, dopca=False
    ):
        self.partition = partition
        self.cutoff = cutoff
        self.dataroot = dataroot
        self.num_samples = num_sample
        self.eps=eps
        self.logtrans = logtransform
        self.dopca=dopca
        
        if self.logtrans and self.dopca:
            raise ValueError("Not implemented")
 #       self.dataset_name = dataset_name
  
#        self.max_samples = int(max_samples)
        self.dataset_name = [dataset + "_" + self.partition for dataset in dataset_name]
        self.atomnum, self.coords, self.spectra, self.mulliken = self.load()

    def load(self):
       
        atomnum = np.load(os.path.join(self.dataroot, self.dataset_name[0] + ".npy"))
        coords = np.load(os.path.join(self.dataroot, self.dataset_name[1] + ".npy"))
        spectra = np.load(os.path.join(self.dataroot, self.dataset_name[2] + ".npy"))
        mullikencharges = np.load(os.path.join(self.dataroot, self.dataset_name[3] + ".npy"))
        
        # charges = np.load('n_body_system/dataset/charges_' + self.sufix + '.npy')
#        spectra = max_normalize(torch.Tensor(Espectra[:,:,1]))
        if self.logtrans:
            spectra = torch.log(torch.Tensor(spectra) + self.eps)

        elif self.dopca:
            
            if self.partition=="train":
                self.n_components = GetPCAcomponents(spectra)
                #            print("n components", self.n_components)
                spectra_std,  spectra_scaler = standardscaler(spectra, scalar=None)
                spectra,  spectra_pca = transform2PCAbasis(spectra_std, self.n_components, pca=None)
                savetransformation(spectra_scaler,spectra_pca)
                spectra=torch.Tensor(spectra)
                spectra, mu, sigma = standardizepcascores(spectra)
                savestat1(mu,sigma)
                
                # print("shape of spectra_transforemd",spectra_transfrmd.shape)
                # print("training", spectra_transfrmd[0])
            else:
                scalar, pca = loadtransformation()
                spectra_std,  _ = standardscaler(spectra, scalar=scalar)
                spectra,  _ = transform2PCAbasis(spectra_std, pca.n_components_ , pca=pca)
                spectra=torch.Tensor(spectra)
                mu, sigma = loadstat()
                spectra, _, _ = standardizepcascores(spectra, mean=mu, std=sigma)
            
                
        else:
            spectra=torch.Tensor(spectra)
                
#        print("spectra from sample 1", spectra[0])
        return torch.Tensor(atomnum).unsqueeze(-1), torch.Tensor(coords), spectra, torch.Tensor(mullikencharges).unsqueeze(-1)

    # def preprocess(self, loc, vel, edges, charges):
    #     # cast to torch and swap n_nodes <--> n_features dimensions
    #     loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
    #     print("shape of loc, vel", loc.shape, vel.shape)
    #     n_nodes = loc.size(2)
    #     loc = loc[0 : self.max_samples, :, :, :]  # limit number of samples
    #     vel = vel[0 : self.max_samples, :, :, :]  # speed when starting the trajectory
    #     charges = charges[0 : self.max_samples]
    #     edge_attr = []

    #     # Initialize edges and edge_attributes
    #     rows, cols = [], []
    #     for i in range(n_nodes):
    #         for j in range(n_nodes):
    #             if i != j:
    #                 edge_attr.append(edges[:, i, j])
    #                 rows.append(i)
    #                 cols.append(j)
    #     edges = [rows, cols]
    #     edge_attr = (
    #         torch.from_numpy(np.array(edge_attr)).float().transpose(0, 1).unsqueeze(2)
    #     )  # swap n_nodes <--> batch_size and add nf dimension
    #     return (
    #         torch.Tensor(loc),
    #         torch.Tensor(vel),
    #         torch.Tensor(edge_attr),
    #         torch.Tensor(edges).long(),
    #         torch.Tensor(charges),
    #     )

    # def set_max_samples(self, max_samples):
    #     self.max_samples = int(max_samples)
    #     self.data, self.edges = self.load()

    # def get_n_nodes(self):
    #     return self.data[0].size(1)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
  
        atomnum, coords, spec, charges = self.atomnum[i], self.coords[i], self.spectra[i], self.mulliken[i]
        edge_index = radius_graph(coords, r=self.cutoff, loop=False)
        # print("edge index", edge_index)
        # print("edge index shape", edge_index.shape)
        # print("atomnum", atomnum)
        # print("coords", coords)
        # print("spec", spec)
        # exit()
#        print("charges", charges)
#        print("shapes of the data tensors", atomnum.shape,coords.shape,spec.shape,charges.shape,edge_index.shape)

        graph_data = MyData(
            z=atomnum,
            mulkncharges=charges,
            pos=coords,
            spectrum=spec,
            edge_index=edge_index,
            num_nodes=25
        )

        return graph_data

    # def __len__(self):
    #     return len(self.data[0])

    # def get_edges(self, batch_size, n_nodes):
    #     edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
    #     if batch_size == 1:
    #         return edges
    #     elif batch_size > 1:
    #         rows, cols = [], []
    #         for i in range(batch_size):
    #             rows.append(edges[0] + n_nodes * i)
    #             cols.append(edges[1] + n_nodes * i)
    #         edges = [torch.cat(rows), torch.cat(cols)]
    #     return edges

    



class RIXS:
    def __init__(
        self,
        cutoff_radii=5,
        batch_size=10,
        pathroot=None,
        filename=None,
        n_train=1,
        n_val=1,
        n_test=1,
        logtransform=True,
        pca=False
    ):
        
        if pathroot is not None:
            dataroot = os.path.join(os.environ["DATAROOT"], pathroot)
        else:
            raise ValueError("pathroot cannot be None")

        
        self.train_dataset = RIXSDataset(dataroot,
                                         partition="train", num_sample=n_train, dataset_name=["atmnum", "coords", "spectra", "mullikencharges"],
                                         cutoff=cutoff_radii, logtransform=logtransform, dopca=pca)
        self.valid_dataset = RIXSDataset(dataroot,
                partition="val", num_sample=n_val, dataset_name=["atmnum", "coords", "spectra", "mullikencharges"],
                                    cutoff=cutoff_radii,logtransform=logtransform, dopca=pca)

        self.test_dataset = RIXSDataset(dataroot,
                partition="test", num_sample=n_test, dataset_name=["atmnum", "coords", "spectra","mullikencharges"],
                                        cutoff=cutoff_radii,logtransform=logtransform, dopca=pca)


        # self.traindebug_dataset = RIXSDataset(dataroot,
        #         partition="train2debug", max_samples=num_samples, dataset_name=["atmnum", "coords", "spectra"]
        #     )
            
        self.batch_size = batch_size

    def train_loader(self):
        
            return PyGDataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )

    def val_loader(self):

            return PyGDataLoader(
                self.valid_dataset, batch_size=self.batch_size, shuffle=False
            )


    def test_loader(self):
 
            return PyGDataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False
            )

    # def traindebug_loader(self):
    #       extra_gen = torch.Generator()
    #       extra_gen.manual_seed(42)
    #       return PyGDataLoader(
    #               self.traindebug_dataset, batch_size=self.batch_size, shuffle=False, generator=extra_gen
    #           )  

def max_normalize(y, eps=1e-12):
    """
    y: (200,) or (batch, 200)
    """
    max_val = y.max(dim=1, keepdim=True).values
    return y / (max_val + eps)

def fit(X,  eps=1e-8, mask_threshold=1e-8):
    
    mean = X.mean(dim=0)

    std = X.std(dim=0,unbiased=False)
    
    mask = std > mask_threshold
    
    k = int((~mask).sum().item())
#    print("k", k)
#    std = torch.clamp(std, min=eps)
    

    return mean, std, mask, k

def savestat1(mean, std):
    
    torch.save({
        "mu": mean,
        "sigma": std,
        },"spectral_stats.pt")

def savestat(mean,std, mask, ncomp):
    
    torch.save({
        "mu": mean,
        "sigma": std,
        "mask":mask,
        "ncomp": ncomp
        },"spectral_stats.pt")
    
def savetransformation(scaler,pca):
    
    joblib.dump(scaler, "y_scaler.pkl")
    joblib.dump(pca, "pca_y.pkl")

def loadtransformation():
    scaler = joblib.load("y_scaler.pkl")
    pca = joblib.load("pca_y.pkl")
    return scaler,pca

# def loadstat():
    
#     ckpt = torch.load("spectral_stats.pt")   
    
#     return ckpt["mu"], ckpt["sigma"], ckpt["mask"], ckpt["ncomp"]

def loadstat():
    
    ckpt = torch.load("spectral_stats.pt")   
    
    return ckpt["mu"], ckpt["sigma"]

def transform(X, mu, sigma):
    return  (X - mu) / sigma   

def standardscaler(dataset, scalar=None):
    
    if scalar is None:
        dataset_scaler = StandardScaler()
        dataset_scaled = dataset_scaler.fit_transform(dataset)
    else:  
        dataset_scaler = scalar
        dataset_scaled = dataset_scaler.transform(dataset)
        
        
    return dataset_scaled, dataset_scaler   

def transform2PCAbasis(dataset, n, pca=None):
    
       
    if pca is None:
        pca_y = PCA(n_components=n)
        dataset_pca = pca_y.fit_transform(dataset)
    else:
        pca_y = pca
        dataset_pca = pca_y.transform(dataset)
        
    return dataset_pca,  pca_y   


def GetPCAcomponents(dataset, min_var=0.95,max_var=0.999,step=0.01,mse_threshold=1e-6):

      data_scaler = StandardScaler()

      dataset_scaled = data_scaler.fit_transform(dataset)

      return  find_pca_components_rse_spectra(dataset_scaled, 
                                              data_scaler, 
                                              start_var=min_var,
                                              max_var=max_var,
                                              step=step,
                                              mse_threshold=mse_threshold)  


def find_pca_components_rse_spectra(y_train_s,
                                   y_scaler,
                                   start_var=0.95,
                                   max_var=0.999,
                                   step=0.01,
                                   mse_threshold=1e-6):
    
    pca_full = PCA()
    pca_full.fit(y_train_s)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    var_threshold = start_var
    
    while var_threshold <= max_var:
        
        n_components = np.argmax(cumulative_variance >= var_threshold) + 1
        
        pca = PCA(n_components=n_components)
        y_train_pca = pca.fit_transform(y_train_s)
        y_reconstructed_s = pca.inverse_transform(y_train_pca)
        
        # go to original spectra
        y_true = y_scaler.inverse_transform(y_train_s)
        y_pred = y_scaler.inverse_transform(y_reconstructed_s)
        
        mse = compute_mse_spectra(y_true, y_pred)
        
        print(f"Var={var_threshold:.3f} | Comp={n_components} | MSE={mse:.6f}")
        
#        save_spectra_pdfs(y_pred,y_true)
        
        if mse <= mse_threshold:
            print("\n✅ Selected configuration:")
            print(f"Variance: {var_threshold}")
            print(f"Components: {n_components}")
            print(f"Relative Spectra Error: {mse}")
            return n_components
        
        var_threshold += step
    
    print("\n⚠️ Threshold not reached, using max variance")
    return n_components    

def compute_mse_spectra(y_true, y_pred, eps=1e-8):
            
    return np.mean((y_true - y_pred)**2)

def standardizepcascores(X, mean=None, std=None, eps=1e-8):
    
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        
    X_std = (X - mean) / (std + eps)

    return X_std, mean, std