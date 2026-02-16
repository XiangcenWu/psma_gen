import ast


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
    import numpy as np

    def plot_organ_metrics_single_row(masks_names, metric_lists, legend_names, selected_organs):
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
        ax.set_title('Metrics by Organ', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
# Example usage:
    
# plot_organ_metrics_single_row(masks_names, dice_after_lists, tre_after_lists, selected_organs)
    ctsmoothness_l8000_k10_mar3000_gam1d2 = r"C:\Users\Sam\Downloads\pet_reg_results\ctsmoothness_l8000_k10_mar3000_gam1.2.txt"
    ctsmoothness_l8000_k10_mar3000_gam1d5 = r"C:\Users\Sam\Downloads\pet_reg_results\ctsmoothness_l8000_k10_mar3000_gam1.5.txt"
    ctsmoothness_l8000_k10_mar3000_gam2 = r"C:\Users\Sam\Downloads\pet_reg_results\ctsmoothness_l8000_k10_mar3000_gam2.0.txt"

    masks_names, dice_before_lists, dice_after_lists_0, tre_before_lists, tre_after_lists_0 = load_registration_results(ctsmoothness_l8000_k10_mar3000_gam1d2)
    _, dice_before_lists, dice_after_lists_1, tre_before_lists, tre_after_lists_1 = load_registration_results(ctsmoothness_l8000_k10_mar3000_gam1d5)
    _, dice_before_lists, dice_after_list2, tre_before_lists, tre_after_lists_2 = load_registration_results(ctsmoothness_l8000_k10_mar3000_gam2)
    # print(masks_names)
    selected_organs = ['spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver', 'stomach', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left', 'lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'esophagus', 'trachea', 'thyroid_gland', 'small_bowel', 'duodenum', 'colon', 'urinary_bladder', 'prostate']
    plot_organ_metrics_single_row(masks_names, \
        [dice_after_lists_0, dice_after_lists_1, dice_after_list2], \
        ['1.2', '1.5', '2'], selected_organs)

    plot_organ_metrics_single_row(masks_names, \
        [tre_after_lists_0, tre_after_lists_1, tre_after_lists_2], \
        ['1.2', '1.5', '2'], selected_organs)
    