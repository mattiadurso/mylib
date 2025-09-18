import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .conversions import to_torch
from .geometry import compute_epipolar_lines_to_plot_from_F, compute_fundamental_from_essential
from .metrics import check_epipolar_constraint, compute_sampson_error


def plot_imgs(images):
    fig, ax = plt.subplots(1,len(images), figsize=(5*len(images),5))
    for i,img in enumerate(images):
        ax[i].title.set_text(f'Image {i+1}')
        ax[i].imshow(img)
        ax[i].axis('off')

    # tight layout
    plt.tight_layout()
    plt.show()


def plot_from_dataset(imgs, ids, title=None, save_path=None):
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img/255)
        plt.axis('off')
        plt.title(f"Frame {ids[i]}")
    plt.tight_layout()
    if title:
        plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_imgs_and_kpts_old(img1, img2, kpt1, kpt2, space=50, matches=True, index=False, sample_points=32):
    # """
    # Plot two images side by side with keypoints overlayed and matches if specified.
    # """
    # #assert (img1-img2).sum() != 0, "Images must be different"
    # #assert not torch.allclose(kpt1,kpt2), "Keypoints must be different"

    # white = torch.ones((img1.shape[0], space, 3)).int()*255
    # concat = torch.cat((img1, white, img2), dim=1).int()
    
    # plt.figure(figsize=(15, 8))
    # plt.imshow(concat)

    # sample_points = min(sample_points, len(kpt1))
    # if sample_points and sample_points < len(kpt1):
    #     kpt1 = kpt1[::len(kpt1)//sample_points]
    #     kpt2 = kpt2[::len(kpt2)//sample_points]
    
    # if index:
    #     for i,(x,y) in enumerate(kpt1):
    #         plt.text(x, y, c="w", s=str(i), fontsize=8, ha='center', va='center')
    #     for i,(x,y) in enumerate(kpt2):
    #         plt.text(x + img1.shape[1] + space, y, c="w", s=str(i), fontsize=8,ha='center', va='center')

    # plt.scatter(kpt1[:, 0],                         kpt1[:, 1], c="r", s=13)
    # plt.scatter(kpt2[:, 0] + img1.shape[1] + space, kpt2[:, 1], c="r", s=13)

    # if matches:
    #     for i in range(kpt1.shape[0]):
    #         plt.plot([kpt1[i, 0], kpt2[i, 0] + img1.shape[1] + space], [kpt1[i, 1], kpt2[i, 1]], c="g", linewidth=1, alpha=0.5)

    # plt.title("Image 1                 Image 2")
    # plt.xlim([0, img1.shape[1] + img2.shape[1] + space])
    # plt.ylim([img1.shape[0], 0])
    print("plot_imgs_and_kpts_old is deprecated, use plot_imgs_and_kpts instead.")



def plot_imgs_and_kpts(
    img1,
    img2,
    kpt1,
    kpt2,
    space=50,
    matches=True,
    index=False,
    sample_points=32,
    pad_color=(255, 255, 255),
):
    """
    Plot two images side by side with keypoints overlaid and matches if specified.
    Handles differing heights (e.g., landscape vs portrait) by padding vertically
    so both images share the same height, and adjusts keypoints for that padding.

    img1, img2: torch tensors of shape (H, W, 3), expected in [0,255]
    kpt1, kpt2: tensors of shape (N,2) with (x,y) coordinates
    pad_color: tuple of 3 ints for RGB padding (default is white)
    """
    def pad_to_height(img, target_h, pad_color):
        h, w, c = img.shape
        if h >= target_h:
            return img, 0
        total_pad = target_h - h
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top
        # build pad canvas with desired color
        color_tensor = torch.tensor(pad_color, dtype=img.dtype, device=img.device).view(1, 1, 3)
        padded = color_tensor.expand(target_h, w, 3).clone()
        padded[pad_top : pad_top + h] = img
        return padded, pad_top

    # Determine target height and pad both images
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    target_h = max(h1, h2)

    img1_padded, offset1 = pad_to_height(img1, target_h, pad_color)
    img2_padded, offset2 = pad_to_height(img2, target_h, pad_color)

    # Adjust keypoints for vertical padding (y coordinate)
    kpt1_adj = kpt1.clone().float()
    kpt2_adj = kpt2.clone().float()
    kpt1_adj[:, 1] += offset1
    kpt2_adj[:, 1] += offset2

    # Sample if requested (keeping correspondence)
    max_samples = min(kpt1_adj.shape[0], kpt2_adj.shape[0])
    sample_points = min(sample_points, max_samples)
    if sample_points and sample_points < max_samples:
        step = max(1, kpt1_adj.shape[0] // sample_points)
        kpt1_adj = kpt1_adj[::step][:sample_points]
        kpt2_adj = kpt2_adj[::step][:sample_points]

    # Build concatenated image with white separator
    separator = torch.ones((target_h, space, 3), dtype=img1_padded.dtype, device=img1_padded.device) * 255
    concat = torch.cat((img1_padded, separator, img2_padded), dim=1)
    concat_np = concat.cpu().numpy().astype(np.uint8)

    # Plot
    plt.figure(figsize=(15, 8))
    plt.imshow(concat_np)
    x_offset = w1 + space

    if index:
        for i, (x, y) in enumerate(kpt1_adj):
            plt.text(x.item(), y.item(), str(i), color="w", fontsize=8, ha="center", va="center")
        for i, (x, y) in enumerate(kpt2_adj):
            plt.text(x.item() + x_offset, y.item(), str(i), color="w", fontsize=8, ha="center", va="center")

    plt.scatter(kpt1_adj[:, 0].cpu(), kpt1_adj[:, 1].cpu(), c="r", s=13)
    plt.scatter(kpt2_adj[:, 0].cpu() + x_offset, kpt2_adj[:, 1].cpu(), c="r", s=13)

    if matches:
        for i in range(kpt1_adj.shape[0]):
            x1, y1 = kpt1_adj[i]
            x2, y2 = kpt2_adj[i]
            plt.plot(
                [x1.item(), x2.item() + x_offset],
                [y1.item(), y2.item()],
                c="g",
                linewidth=1,
                alpha=0.5,
            )

    # plt.title("Image 1                 Image 2")
    plt.xlim([0, w1 + w2 + space])
    plt.ylim([target_h, 0])
    plt.axis("off")
    plt.tight_layout()
    plt.show()



def plot_imgs_and_epipolar_lines(img1, img2, points1, points2, E12, K1=None, K2=None,  sample_points=12, skip_left=False, verbose=True):
    """
    Compute and plot epipolar lines between two images. Points must be in pixel space. 
    NOTE:
    Provide intrisic matrices if working with Essential matrix.
    Args:
        img1, img2: images
        points1, points2: numpy (N,2)
        E21_in: Essential matrix from img1 to img2
        K1: intrinsics matrix. If not provided, E12 is assumed to be the fundamental matrix.
        K2: intrinsics matrix. If not provided, K1 is assumed to be the intrinsics matrix of the second image.
        sample_points: points to display.
    """
    points1 = to_torch(points1, b=False)
    points2 = to_torch(points2, b=False) if points2 is not None else torch.zeros_like(points1)-1
    E12 = to_torch(E12)

    if K1 is not None:
        K1 = to_torch(K1, b=False)
    if K2 is not None:
        K2 = to_torch(K2, b=False)
    else:
        K2 = K1

    # if working with Essential matrix, E becames F
    if K1 is not None:
        E12 = compute_fundamental_from_essential(E12, K1, K2)
    E21 = E12.permute(0,2,1)

    # Some stats
    if verbose:
        print("Epipolar constraint (all points) x'Fx=0:", check_epipolar_constraint(E12, points1, points2))
        print("Sampson error (all points):             ", compute_sampson_error(E12, points1, points2))
        print(f"\nTotal points: {len(points1)}, displaying: {min(sample_points,len(points1))} and {min(sample_points,len(points2))} points.")
        
    if sample_points and sample_points < len(points1):
        points1 = points1[::len(points1)//sample_points]
        points2 = points2[::len(points2)//sample_points]

    fig, ax = plt.subplots(1,2, figsize=(15,8))
    ax[0].title.set_text('Image 1')
    ax[0].imshow(img1)
    ax[0].scatter(points1[:,0], points1[:,1], c='r', s=5, marker='x')
    ax[1].title.set_text('Image 2')
    ax[1].imshow(img2)
    ax[1].scatter(points2[:,0], points2[:,1], c='r', s=5, marker='x')
  

    # compute and plot epipolar lines 1 -> 2
    lines_points2 = compute_epipolar_lines_to_plot_from_F(img2, E12, points1)
    for i in range(len(lines_points2)):
        ax[1].plot(lines_points2[i][0], lines_points2[i][1],linewidth=1)

    if not skip_left:
        # compute and plot epipolar lines 2 -> 1
        lines_points1 = compute_epipolar_lines_to_plot_from_F(img1, E21, points2)
        for i in range(len(lines_points1)):
            ax[0].plot(lines_points1[i][0], lines_points1[i][1],linewidth=1)
    
    ax[0].set_ylim([img1.shape[0], 0])
    ax[0].set_xlim([0, img1.shape[1]])
    ax[1].set_ylim([img2.shape[0], 0])
    ax[1].set_xlim([0, img2.shape[1]])
    plt.tight_layout()
    plt.show()


def plot_bar_from_df(df, df2=None, models=None, xlabel=None, ylabel=None, xticks=None, yticks=None, title=None, 
                     ylim=[0, 105], xlim=None, size=(8, 6), bar_width=0.5, legend_title=None):
    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=size)
    
    # Plot the second DataFrame, if provided
    if df2 is not None:
        df2.plot(kind='bar', width=bar_width, position=0, alpha=0.4, ax=ax)
        df.plot(kind='bar', width=bar_width, position=0, alpha=1.0, ax=ax)
    else:
        df.plot(kind='bar', width=bar_width, position=0, alpha=1.0, ax=ax)

    # Adjust y-axis limits
    ax.set_ylim(ylim[0], ylim[1])

    # Adding labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Center the x-ticks and rotate
    ax.set_xticks(np.arange(len(df))+.25)  # Set ticks at the center of each bar group
    ax.set_xticklabels(df.index, rotation=45, ha='right', rotation_mode="anchor")  # Rotate labels for better readability

    # if len(df) > 10: plot y also on the right
    if len(df) > 10:
        # Create a secondary y-axis on the right
        ax_right = ax.twinx()
        ax_right.set_yticks(ax.get_yticks())  # Set the same label on the right
        ax_right.yaxis.set_ticks_position('right')  # Set ticks on the right side
        ax_right.set_ylim(ylim[0], ylim[1])

    # Display the legend outside the plot
    if models is None:
        models = df.columns if df2 is None else df2.columns.tolist() + df.columns.tolist()
    ax.legend(models, title=legend_title, loc='center left', bbox_to_anchor=(1.02, 0.5))

    # Draw a vertical line if the last index is "Mean"
    if df.index[-1] == "Mean" or df.index[-1] == "mean":
        n = len(df)
        ax.axvline(x=n - (1+bar_width/2), ymin=0., ymax=0.98, color='black', linewidth=1, linestyle='--')

    
    

    plt.tight_layout()
    plt.show()
