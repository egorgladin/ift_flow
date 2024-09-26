import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_losses(fnames, labels, save_every=50):
    plt.figure(figsize=(8, 6))

    for fname, lbl in zip(fnames, labels):
        with open(f'losses_pkl/{fname}.pkl', 'rb') as file:
            losses = pickle.load(file)

        mean_curve = np.mean(losses, axis=0)
        std_curve = np.std(losses, axis=0)
        save_every_ = 2 * save_every if fname.startswith('ift') else save_every
        iterations = np.arange(1, len(mean_curve) * save_every_, save_every_)
        p = plt.plot(iterations, mean_curve, label=lbl)
        plt.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve, color=p[0].get_color(), alpha=0.2)

    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('MMD')
    plt.yscale('log')
    plt.grid()
    im_name = f'losses_semi_final11.pdf'
    plt.savefig(im_name, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {im_name}")


if __name__ == "__main__":
    fnames = ['mmd_10000_None_0_None_None', 'mmd_10000_2_4000_None_None',
              'ift_10000_None_0_0.001_MMD', 'ift_10000_None_0_1.0_KL']
    labels = ['MMD flow', 'MMD flow + noise', 'IFT particle GD', 'IFT particle GD with KL step']
    plot_losses(fnames, labels)
