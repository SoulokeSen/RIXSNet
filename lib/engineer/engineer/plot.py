#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 12:23:06 2025

@author: souloke
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os
import joblib, shutil
from matplotlib.backends.backend_pdf import PdfPages

def save_spectra_pdfs(preds, targets, output_dir="spectra_plots"):
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    n_samples = preds.shape[0]
    with PdfPages(output_dir / "all_plots.pdf") as pdf:
        for i in range(n_samples):
            plt.figure()

            plt.plot(targets[i], label="Target")
            plt.plot(preds[i], label="Prediction")

            plt.xlabel("Spectra index / Energy")
            plt.ylabel("Intensity")
            plt.ylim(-0.1, 1.0)
            plt.title(f"Sample {i}")
            plt.legend()

            filename = os.path.join(output_dir, f"sample_{i}.pdf")
#            plt.savefig(filename, bbox_inches="tight")
            pdf.savefig()
            plt.close()

    print(f"Saved {n_samples} PDF files in '{output_dir}'")    
#    plt.show()

# def loadstat():
    
#     ckpt = torch.load("spectral_stats.pt")   
    
#     return ckpt["mu"], ckpt["sigma"], ckpt["mask"], ckpt["ncomp"]

def loadstat():
    
    ckpt = torch.load("spectral_stats.pt")   
    
    return ckpt["mu"], ckpt["sigma"]

def loadtransformation():
    scaler = joblib.load("y_scaler.pkl")
    pca = joblib.load("pca_y.pkl")
    return scaler,pca

def dumpandplot(loss, spectra_list, total_loss_batch, prefix, rundir, log, pca):
    
##dump errors for 2rdm
    outputdir = Path(rundir)
    name1 = "losses"+prefix+".txt"
    np.savetxt(outputdir / name1, loss.cpu().numpy(), fmt="%.8f")

    name2 = "total_loss_fullbatch_spectra_"+prefix+".txt"
    loss_batch = np.array(total_loss_batch)
    np.savetxt(outputdir / name2, loss_batch, fmt="%.8f")
    
    mean = loss_batch.mean()
    median = np.median(loss_batch)
    max_val = loss_batch.max()
    max_index = np.argmax(loss_batch)
#    p90 = np.percentile(loss, 90)

    threshold = np.percentile(loss_batch, 95)   # top 5%
    spike_indices = np.where(loss_batch >= threshold)[0]

    name3 = "errors_spectra_"+prefix+".txt"
    with open(outputdir / name3, "w") as f:
        f.write(f"Mean: {mean:.8f}\n")
        f.write(f"Median: {median:.8f}\n")
        f.write(f"Max: {max_val:.8f}\n")
        f.write(f"Max_index: {max_index}\n")
        f.write(f"Spike Indices: {spike_indices.tolist()}\n")


##dump errors for energy

    n_batch = len(spectra_list)
    spectra_pred=[]
    spectra_rixs=[]
    for i in range(n_batch):
        spectra_pred.append(spectra_list[i][0])
        spectra_rixs.append(spectra_list[i][1])
        
    spectra_pred_2plot = torch.cat(spectra_pred, dim=0)
    spectra_rixs_2plot = torch.cat(spectra_rixs, dim=0)  
    
    if log :
        spectra_pred_2plot = torch.exp(spectra_pred_2plot).cpu().numpy()
        spectra_rixs_2plot = torch.exp(spectra_rixs_2plot).cpu().numpy()
    elif pca:
        scalar, pca = loadtransformation()
        mu, sigma = loadstat()
        spectra_pred_2plot = spectra_pred_2plot * sigma + mu
        spectra_rixs_2plot = spectra_rixs_2plot * sigma + mu
        
        spectra_pred_2plot = pca.inverse_transform(spectra_pred_2plot.cpu().numpy())
        spectra_pred_2plot = scalar.inverse_transform(spectra_pred_2plot)
         
        spectra_rixs_2plot = pca.inverse_transform(spectra_rixs_2plot.cpu().numpy())
        spectra_rixs_2plot = scalar.inverse_transform(spectra_rixs_2plot)    
         
    else:
        spectra_pred_2plot = spectra_pred_2plot.cpu().numpy()
        spectra_rixs_2plot = spectra_rixs_2plot.cpu().numpy()



    
    # Create a figure
    plt.figure(figsize=(8, 6))

    # spectra_error_mae = torch.abs(spectra_pred_2plot-spectra_rixs_2plot)

    # name4 = "total_loss_fullbatch_spectra_"+prefix+".txt"
    # loss_batch_spectra = spectra_error_mae.cpu().numpy()
    # np.savetxt(outputdir / name4, loss_batch_spectra, fmt="%.8f")
    
#     mean = loss_batch_energy.mean()
#     median = np.median(loss_batch_energy)
#     max_val = loss_batch_energy.max()
#     max_index = np.argmax(loss_batch_energy)
# #    p90 = np.percentile(loss, 90)

#     threshold = np.percentile(loss_batch_energy, 95)   # top 5%
#     spike_indices = np.where(loss_batch_energy >= threshold)[0]

#     name5 = "errors_energy_"+prefix+".txt"
#     with open(outputdir / name5, "w") as f:
#         f.write(f"Mean: {mean:.8f}\n")
#         f.write(f"Median: {median:.8f}\n")
#         f.write(f"Max: {max_val:.8f}\n")
#         f.write(f"Max_index: {max_index}\n")
#         f.write(f"Spike Indices: {spike_indices.tolist()}\n")




#     mae = energy_error_mae.mean()
#     name6 = "mae_energy_"+prefix+".txt"
#     np.savetxt(outputdir / name6, np.atleast_1d(mae.cpu().numpy()), fmt="%.8f")    


#calculate mean energy error and write to file
    plt.imshow(spectra_pred_2plot - spectra_rixs_2plot, aspect="auto", cmap="coolwarm")
    plt.colorbar()
    plt.title("Error map")
    plt.xlabel("Spectra dimension")
    plt.ylabel("Samples")
    name6 = "error_heatmap_"+prefix+".png"
    plt.savefig(outputdir / name6, dpi=300, bbox_inches="tight")
    plt.close()
    name7 = "spectra_plots_"+prefix
    save_spectra_pdfs(spectra_pred_2plot, spectra_rixs_2plot, output_dir= outputdir / name7)