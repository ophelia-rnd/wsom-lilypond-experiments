import matplotlib.pyplot as plt

INTRINSIC_POSITION_CMAP = "plasma"

def visualize_manifold(X, y, title="Manifold in ambient space", cmap=INTRINSIC_POSITION_CMAP, plot_style_args=None, n_tick_labels=5, ax=None, hold_on=False):
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    plot_style = {"alpha": 0.8}
    plot_style.update(plot_style_args or {})

    ax.scatter(*X.T, c=y, s=50, cmap=cmap, **plot_style)
    ax.view_init(azim=-66, elev=12)

    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=n_tick_labels))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=n_tick_labels))
    ax.zaxis.set_major_locator(plt.MaxNLocator(nbins=n_tick_labels))

    ax.set_title(title)
    plt.tight_layout()

    if not hold_on:
        plt.show()
        return fig, ax

    else:
        return fig, ax

def place_som_lattice(ax, node_weights, node_weights_flat):
    n_rows, n_cols = node_weights.shape[:2]
    W = node_weights.reshape(n_rows, n_cols, -1)

    # nodes
    ax.scatter(*node_weights_flat.T, edgecolor="red", color="coral", s=10, marker="s", alpha=0.5)

    # horizontal connections
    for i in range(n_rows):
        for j in range(n_cols - 1):
            ax.plot(*W[i, j:j+2].T, color="red", alpha=0.8, linewidth=1)

    # vertical connections
    for i in range(n_rows - 1):
        for j in range(n_cols):
            ax.plot(*W[i:i+2, j].T, color="red", alpha=0.8, linewidth=1)
