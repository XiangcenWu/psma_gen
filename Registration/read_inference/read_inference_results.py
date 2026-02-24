import ast
import numpy as np


def load_registration_results(filename):
    """
    Load registration results from text file.

    Returns:
        masks_names,
        dice_before_lists,
        dice_after_lists,
        tre_before_lists,
        tre_after_lists
    """

    def parse_list_line(line):
        parts = line.strip().split(";")
        parsed = []
        for p in parts:
            p = p.strip()

            # try to interpret as python literal (list, float, etc.)
            try:
                parsed.append(ast.literal_eval(p))
            except Exception:
                parsed.append(p)

        return parsed

    with open(filename, "r") as f:
        lines = f.readlines()

    masks_names = parse_list_line(lines[0])
    dice_before_lists = parse_list_line(lines[1])
    dice_after_lists = parse_list_line(lines[2])
    tre_before_lists = parse_list_line(lines[3])
    tre_after_lists = parse_list_line(lines[4])

    return (
        masks_names,
        dice_before_lists,
        dice_after_lists,
        tre_before_lists,
        tre_after_lists,
    )


def get_metrics_for_masks(mask_names_input, masks_names, dice_before_lists, dice_after_lists, tre_before_lists, tre_after_lists):
    """
    Returns the registration metrics for the given list of mask names.

    Parameters:
        mask_names_input (list of str): List of mask names to query.
        masks_names (list of str): Full list of all mask names.
        dice_before_lists (list of float): Dice scores before registration.
        dice_after_lists (list of float): Dice scores after registration.
        tre_before_lists (list of float): TRE values before registration.
        tre_after_lists (list of float): TRE values after registration.

    Returns:
        dict: Dictionary where keys are mask names and values are dicts of metrics.
    """
    results = {}
    for name in mask_names_input:
        if name in masks_names:
            idx = masks_names.index(name)
            results[name] = {
                "dice_before": dice_before_lists[idx],
                "dice_after": dice_after_lists[idx],
                "tre_before": tre_before_lists[idx],
                "tre_after": tre_after_lists[idx]
            }
        else:
            results[name] = {
                "dice_before": None,
                "dice_after": None,
                "tre_before": None,
                "tre_after": None
            }
    return results

if __name__ == "__main__":


    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def plot_organ_metrics_single_row(masks_names, metric_lists, legend_names, selected_organs, title):
        """
        Plot multiple metrics in a single row grouped by organ.
        
        Parameters:
        -----------
        masks_names : list of str
            Names of all organs
        metric_lists : list of lists of lists
            List of metric data. Each element is a list of lists containing scores for each organ.
            Example: [dice_after_lists, tre_after_lists, another_metric_lists]
        legend_names : list of str
            Names for each metric to display in legend.
            Example: ['DICE', 'TRE', 'Another Metric']
        selected_organs : list of str
            List of organ names to plot (e.g., ['liver', 'heart', 'kidney_right'])
        """
        # Find indices of selected organs
        selected_indices = [masks_names.index(organ) for organ in selected_organs]
        
        # Extract data for selected organs for each metric
        selected_metrics = []
        for metric_list in metric_lists:
            selected_metrics.append([metric_list[i] for i in selected_indices])
        
        n_organs = len(selected_organs)
        n_metrics = len(metric_lists)
        
        # Calculate positions for each boxplot
        group_width = n_metrics * 1.2
        positions_base = np.arange(n_organs) * (group_width + 1)
        
        # Define colors for different metrics
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lavender']
        
        fig, ax = plt.subplots(figsize=(max(10, n_organs * 2), 6))
        
        # Create boxplots for each metric
        bps = []
        for i, (selected_data, legend_name) in enumerate(zip(selected_metrics, legend_names)):
            offset = (i - (n_metrics - 1) / 2) * 0.6
            positions = positions_base + offset

            selected_data = [np.asarray(d, dtype=float) for d in selected_data]
            
            bp = ax.boxplot(selected_data, positions=positions, widths=0.5,
                        patch_artist=True, label=legend_name,
                        flierprops=dict(marker='o', markersize=3, linestyle='none', 
                                    markerfacecolor='black', alpha=0.5))

            


            
            # Color the boxes
            color = colors[i % len(colors)]
            for patch in bp['boxes']:
                patch.set_facecolor(color)
            
            bps.append(bp)
        
        ax.set_xticks(positions_base)
        ax.set_xticklabels(selected_organs, rotation=90)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


    def plot_separate_legend(legend_names, colors, ncol=None, figsize=(4, 1.2), fontsize=10, frameon=False):
        handles = [Patch(facecolor=colors[i], edgecolor='black') for i in range(len(legend_names))]

        fig_leg = plt.figure(figsize=figsize)
        fig_leg.legend(
            handles, legend_names,
            loc="center",
            ncol=(len(legend_names) if ncol is None else ncol),
            fontsize=fontsize,
            frameon=frameon,
            columnspacing=1.2,
            handlelength=1.5,
            handletextpad=0.6
        )
        plt.axis("off")
        fig_leg.tight_layout()
        plt.show()
        return fig_leg

    def plot_organ_metrics_single_row_horizontal(
        masks_names, metric_lists, legend_names, selected_organs, title,
        box_thickness=0.9,   # <<< control thickness here
        metric_spacing=0.9,  # <<< controls separation between metrics in a group
        yaxis=False
    ):
        selected_indices = [masks_names.index(organ) for organ in selected_organs]

        selected_metrics = []
        for metric_list in metric_lists:
            selected_metrics.append([metric_list[i] for i in selected_indices])

        n_organs = len(selected_organs)
        n_metrics = len(metric_lists)

        group_height = n_metrics * 1.2
        positions_base = np.arange(n_organs) * (group_height + 1)

        colors = ['tab:blue', 'tab:gray', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        plot_separate_legend(legend_names, colors, ncol=len(legend_names), figsize=(6, 1.2))

        # fig, ax = plt.subplots(figsize=(10, max(6, n_organs * 0.6)))
        fig, ax = plt.subplots(figsize=(6, 4))

        for i, (selected_data, legend_name) in enumerate(zip(selected_metrics, legend_names)):
            offset = (i - (n_metrics - 1) / 2) * metric_spacing
            positions = positions_base + offset

            selected_data = [np.asarray(d, dtype=float) for d in selected_data]

            bp = ax.boxplot(
                selected_data,
                positions=positions,
                widths=box_thickness,   # <<< thicker boxes
                vert=False,
                patch_artist=True,
                labels=None,
                showfliers=False,
                flierprops=dict(
                    marker='o',
                    markersize=3,
                    linestyle='none',
                    markerfacecolor='black',
                    alpha=0.5
                )
            )

            color = colors[i % len(colors)]
            for patch in bp['boxes']:
                patch.set_facecolor(color)

            ax.plot([], [], color=color, label=legend_name)
        if yaxis == True:
            ax.set_yticks(positions_base)
            clean_organs = [o.replace("_", " ") for o in selected_organs]
            ax.set_yticklabels(clean_organs)
        else:
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False)

        

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)

        # ax.set_xlabel('Score', fontsize=12)
        # ax.set_title(title, fontsize=14, fontweight='bold')
        # ax.legend()
        ax.grid(False)

        plt.tight_layout()
        plt.show()
    selected_organs = [
        "brain",
        "skull",
        "heart",
        "aorta",
        "trachea",
        "esophagus",
        "liver",
        "spleen",
        "pancreas",
        "stomach",
        "kidney_left",
        "kidney_right",
        "colon",
        "urinary_bladder",
        "prostate",
        "spinal_cord",
        "vertebrae_L3",
    ]


# Example usage:
    s = 4500
    margin = 3000
    # plot_organ_metrics_single_row(masks_names, dice_after_lists, tre_after_lists, selected_organs)
    ctsmoothness_l12000_k10_mar3000_gam1 = fr"C:\Users\Sam\Downloads\pet_reg_results\ctsmoothness_l{s}_k10_mar{margin}_gam1.0.txt"
    ctsmoothness_l12000_k10_mar3000_gam2 = fr"C:\Users\Sam\Downloads\pet_reg_results\ctsmoothness_l{s}_k10_mar{margin}_gam2.0.txt"
    baseline_l12000_k10 = fr"C:\Users\Sam\Downloads\pet_reg_results\baseline_l{s}_k10.txt"

    masks_names, dice_before_lists, dice_after_lists_0, tre_before_lists, tre_after_lists_0 = load_registration_results(ctsmoothness_l12000_k10_mar3000_gam1)
    _, dice_before_lists, dice_after_lists_1, tre_before_lists, tre_after_lists_1 = load_registration_results(ctsmoothness_l12000_k10_mar3000_gam2)
    _, dice_before_lists, dice_after_lists_2, tre_before_lists, tre_after_lists_2 = load_registration_results(baseline_l12000_k10)

    print(len(dice_after_lists_0))
   
    plot_organ_metrics_single_row_horizontal(masks_names, \
        [dice_after_lists_0, dice_after_lists_1, dice_after_lists_2], \
        [r'$\gamma=1$', r'$\gamma=2$', 'baseline'], selected_organs, title = 'dice' + str(s) + str(margin), yaxis=True)
    plot_organ_metrics_single_row_horizontal(masks_names, \
        [tre_after_lists_0, tre_after_lists_1, tre_after_lists_2], \
        [r'$\gamma=1$', r'$\gamma=2$', 'baseline'], selected_organs, title = 'tre' + str(s) + str(margin), yaxis=False)


    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Labels and colors (edit as needed)
    legend_names = ["p < 0.05", "p < 0.001", "insignificant"]
    colors = ['tab:purple', 'tab:red', 'tab:brown']

    # Create dot handles
    handles = [
        Line2D([0], [0],
            marker="o", linestyle="None",
            markerfacecolor=c, markeredgecolor=c,
            markersize=12)
        for c in colors
    ]

    # Legend-only figure
    fig_leg = plt.figure(figsize=(6, 1.2))
    fig_leg.legend(
        handles, legend_names,
        loc="center",
        ncol=len(legend_names),
        frameon=False,
        fontsize=10,
        columnspacing=1.5,
        handletextpad=0.6
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


    # selected_organs = [
    #     "brain",
    #     "skull",
    #     "lung_upper_lobe_left",
    #     "liver",
    #     "hip_left",
    #     "prostate",
    #     "vertebrae_L3"
    # ]
    # baseline_l5000_k10 = fr"C:\Users\Sam\Downloads\pet_reg_results\baseline_l4000_k10.txt"
    # baseline_l6000_k10 = fr"C:\Users\Sam\Downloads\pet_reg_results\baseline_l4500_k10.txt"
    # baseline_l7000_k10 = fr"C:\Users\Sam\Downloads\pet_reg_results\baseline_l5000_k10.txt"
    # baseline_l8000_k10 = fr"C:\Users\Sam\Downloads\pet_reg_results\baseline_l5500_k10.txt"
    # baseline_l10000_k10 = fr"C:\Users\Sam\Downloads\pet_reg_results\baseline_l6000_k10.txt"
    # baseline_l12000_k10 = fr"C:\Users\Sam\Downloads\pet_reg_results\baseline_l7000_k10.txt"

    # masks_names, dice_before_lists, dice_after_lists_0, tre_before_lists, tre_after_lists_0 = load_registration_results(baseline_l5000_k10)
    # masks_names, dice_before_lists, dice_after_lists_1, tre_before_lists, tre_after_lists_1 = load_registration_results(baseline_l6000_k10)
    # masks_names, dice_before_lists, dice_after_lists_2, tre_before_lists, tre_after_lists_2 = load_registration_results(baseline_l7000_k10)
    # _, dice_before_lists, dice_after_list_3, tre_before_lists, tre_after_lists_3 = load_registration_results(baseline_l8000_k10)
    # _, dice_before_lists, dice_after_list_4, tre_before_lists, tre_after_lists_4 = load_registration_results(baseline_l10000_k10)
    # _, dice_before_lists, dice_after_list_5, tre_before_lists, tre_after_lists_5 = load_registration_results(baseline_l12000_k10)

    # plot_organ_metrics_single_row(masks_names, \
    #     [dice_after_lists_0, dice_after_lists_1, dice_after_lists_2, dice_after_list_3, dice_after_list_4, dice_after_list_5], \
    #     ['4000', '4500', '5000', '5500', '6000', '7000'], selected_organs, title = str(s))

    # plot_organ_metrics_single_row(masks_names, \
    #     [tre_after_lists_0, tre_after_lists_1, tre_after_lists_2, tre_after_lists_3, tre_after_lists_4, tre_after_lists_5], \
    #     ['4000', '4500', '5000', '5500', '6000', '7000'], selected_organs, title = str(s))
