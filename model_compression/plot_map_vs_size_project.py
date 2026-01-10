import os
import matplotlib.pyplot as plt
from adjustText import adjust_text

results = [
    {"name": "PicoSAM2", "coco_miou": 0.3337, "coco_map": 0.0970, "size": 1.22},
    {"name": "PicoSAM2 QAT", "coco_miou": 0.3308, "coco_map": 0.0997, "size": 1.22},
    {"name": "U-Net Model", "coco_miou": 0.3970, "coco_map": 0.0903, "size": 6},
]


colors = ['tab:pink', 'tab:blue', 'tab:green']
markers = ['o', 's', '^']

def create_plot(results, x_key, x_label, y_label, title, filename):
    plt.figure(figsize=(9, 6))
    texts = []
    for i, r in enumerate(results):
        x_value = r[x_key]
        y_value = r["size"]
        if x_value is None:
            continue
        plt.scatter(x_value, y_value, color=colors[i], marker=markers[i], s=80, edgecolor='black', label=r["name"])
        # Default label placement
        ha, va = 'left', 'bottom'
        dx, dy = 0, 0.04
        # Special case: U-Net Model
        if r["name"] == "U-Net Model":
            ha, va = 'left', 'center'
            dx, dy = 0.0004, 0
        if r["name"] == "PicoSAM2":
            ha, va = 'center', 'bottom'
            dy = 0.04  # small horizontal offset
        text = plt.text(
            x_value + dx, y_value + dy, r["name"],
            fontsize=12, ha=ha, va=va,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )
        texts.append(text)

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True, which='both', ls="--", alpha=0.7)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=10)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

create_plot(results, "coco_map", "COCO mAP@[0.5:0.95]", "Model Size (MB)", "Model Size vs COCO mAP", "images/coco_map_vs_size_log.png")


def bubble_plot(results, filename):
    plt.figure(figsize=(8, 4.5))

    # Extract values
    x = [r["coco_map"] for r in results]
    y = [r["coco_miou"] for r in results]
    sizes = [r["size"] * 300 for r in results]  # scale factor for visibility

    # Bubble plot
    plt.scatter(
        x, y,
        s=sizes,
        alpha=0.6,
        edgecolor='black'
    )

    # Direct labels
    for r in results:
        # Default label placement
        ha, va = 'center', 'bottom'
        dx, dy = 0, 0.0035
        # Special cases
        if r["name"] == "U-Net Model":
            ha, va = 'left', 'center'
            dx, dy = 0.001, -0.0005
        plt.text(
            r["coco_map"] + dx,
            r["coco_miou"] + dy,
            r["name"],
            ha=ha,
            va=va,
            fontsize=11,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )

    plt.xlabel("COCO mAP@[0.5:0.95]", fontsize=13)
    plt.ylabel("COCO mIoU", fontsize=13)
    plt.ylim(0.325, 0.405)
    plt.title("Accuracy Trade-offs vs Model Size", fontsize=15)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

bubble_plot(results, "images/bubble_accuracy_vs_size.png")
