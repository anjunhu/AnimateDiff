import os
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from typing import List
from scipy.spatial import distance
import tempfile
import difflib
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve

def extract_info_from_filename(filename):
    """Extract index and prompt information from filename."""
    # Example: 0000_0__3D_scan_Wooden_statue_of_a_bigger_cute_bear_bear_wooden_statue_3dscan_3d_asset_0.png
    parts = os.path.basename(filename).split('__', 1)
    
    if len(parts) < 2:
        return None, None, None
    
    prefix = parts[0]  # e.g., "0000_0"
    index_parts = prefix.split('_')
    
    if len(index_parts) < 2:
        return None, None, None
    
    index = int(index_parts[0])
    label = int(index_parts[1])
    
    # Extract the safe prompt part (everything between __ and the last _number)
    safe_prompt = parts[1]
    safe_prompt = re.sub(r'_traj_\d+\.json$|_\d+\.png$|_noise_mag_\d+\.png$', '', safe_prompt)
    
    return index, label, safe_prompt




def get_file_pairs(input_files):
    """Get all the file pairs in the directory."""
    # Find all json files with traj pattern
    json_files = [f for f in input_files if f.endswith('.json') and '_traj' in f]
    
    pairs = []
    for json_file in json_files:
        try:
            # Handle both relative and absolute paths
            json_dir = os.path.dirname(json_file)
            json_basename = os.path.basename(json_file)
            
            # Remove the "_traj.json" suffix to get the base name
            base_name = json_basename.replace("_traj.json", "")
            
            # Search for matching image files using the base pattern
            if json_dir:
                img_pattern = os.path.join(json_dir, f"{base_name}*.png")
                noise_pattern = os.path.join(json_dir, f"{base_name}*noise*.png")
            else:
                img_pattern = f"{base_name}*.png"
                noise_pattern = f"{base_name}*noise*.png"
            
            img_files = glob.glob(img_pattern)
            noise_img_files = glob.glob(noise_pattern)
            
            # Filter out noise files from regular image files
            img_files = [f for f in img_files if "noise" not in os.path.basename(f)]
            
            if img_files and noise_img_files:
                # Use the first matching files
                pairs.append((json_file, img_files[0], noise_img_files[0]))
                print(f"Found pair: {json_file} -> {img_files[0]} + {noise_img_files[0]}")
            else:
                print(f"No matching image files found for {json_file}")
                print(f"  Searched patterns: {img_pattern} and {noise_pattern}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return pairs

def load_json_data(json_file):
    """Load JSON data from file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def save_json_data(json_file, data):
    """Save JSON data to file."""
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

def update_memorization_status(json_file, status):
    """Update the memorization status in the JSON file."""
    data = load_json_data(json_file)
    data["memorized"] = status
    save_json_data(json_file, data)
    return "JSON updated successfully!"




def calculate_rocauc(inputs):
    """Calculate ROCAUC for average noise norm vs memorization status."""
    json_files = inputs 
    
    # Store the average norms and their memorization labels
    avg_norms = []
    labels = []
    dataset_info = []  # Store which dataset each sample belongs to
    
    for json_file in json_files:
        try:
            data = load_json_data(json_file)
            
            # Check which noise norm field to use
            norm_field = "text_noise_norms"
            if "noise_diff_norms" in data:
                norm_field = "noise_diff_norms"  # Use this if available
            
            # Skip if no noise norms or not labeled
            if norm_field not in data or "memorized" not in data:
                continue
                
            # Calculate the average norm across timesteps
            avg_norm = np.mean(data[norm_field])
            avg_norms.append(avg_norm)
            labels.append(1 if data["memorized"] else 0)
            
            # Determine dataset
            if "objaverse" in str(json_file):
                dataset = "Objaverse"
            else:
                dataset = "LAION"
            dataset_info.append(dataset)
                
        except (KeyError, FileNotFoundError) as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if not avg_norms or not labels:
        return None, None, None, None, None
    
    # Calculate overall ROCAUC
    fpr, tpr, thresholds = roc_curve(labels, avg_norms)
    roc_auc = auc(fpr, tpr)
    
    # Find the optimal threshold using Youden's J statistic (maximizing sensitivity + specificity - 1)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
    # Calculate dataset-specific ROCAUCs
    dataset_results = {}
    for dataset_name in set(dataset_info):
        # Skip if it's just non-memorized samples (they're all treated the same)
        if sum(1 for i, l in enumerate(labels) if l == 1 and dataset_info[i] == dataset_name) == 0:
            continue
            
        # For each dataset, we compare memorized samples from this dataset vs all non-memorized
        dataset_labels = []
        dataset_norms = []
        
        for i, (norm, label) in enumerate(zip(avg_norms, labels)):
            # Include all non-memorized samples (label=0) regardless of dataset
            # For memorized samples (label=1), only include those from this dataset
            if label == 0 or (label == 1 and dataset_info[i] == dataset_name):
                dataset_labels.append(label)
                dataset_norms.append(norm)
                
        # Calculate ROCAUC for this dataset
        try:
            ds_fpr, ds_tpr, ds_thresholds = roc_curve(dataset_labels, dataset_norms)
            ds_roc_auc = auc(ds_fpr, ds_tpr)
            
            # Find optimal threshold
            ds_j_scores = ds_tpr - ds_fpr
            ds_best_idx = np.argmax(ds_j_scores)
            ds_optimal_threshold = ds_thresholds[ds_best_idx]
            
            dataset_results[dataset_name] = {
                'auc': ds_roc_auc,
                'fpr': ds_fpr,
                'tpr': ds_tpr,
                'threshold': ds_optimal_threshold
            }
        except Exception as e:
            print(f"Error calculating ROCAUC for {dataset_name}: {e}")
    
    return roc_auc, fpr, tpr, optimal_threshold, dataset_results

def plot_roc_curve(fpr, tpr, roc_auc, output_file="roc_curve.png", dataset_results=None):
    """Create and save ROC curve plot with per-dataset breakdowns."""
    plt.figure(figsize=(10, 8))
    
    # Plot overall ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Overall ROC (AUC = {roc_auc:.3f})')
    
    # Plot per-dataset ROC curves if available
    if dataset_results:
        colors = {'LAION': 'red', 'Objaverse': 'black'}
        for dataset_name, result in dataset_results.items():
            plt.plot(
                result['fpr'], 
                result['tpr'], 
                color=colors.get(dataset_name, 'purple'), 
                lw=1.5, 
                linestyle='-',
                label=f'{dataset_name} ROC (AUC = {result["auc"]:.3f})'
            )
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AnimateDiff - ROC Curves: Average Noise Norm as Memorization Predictor')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return output_file

def plot_score_distributions(inputs, optimal_threshold=None, output_file="score_distribution.png"):
    """Plot distributions of average noise norms for memorized vs non-memorized samples."""
    json_files = inputs[:700]
    
    # Separate scores by dataset and memorization status
    objaverse_memorized_scores = []
    laion_memorized_scores = []
    not_memorized_scores = []  # All non-memorized (combined across all datasets)
    
    for json_file in json_files:
        try:
            data = load_json_data(json_file)
            is_objaverse = "objaverse" in str(json_file)
            
            # Check which norm field to use
            norm_field = "text_noise_norms"
            if "noise_diff_norms" in data:
                norm_field = "noise_diff_norms"  # Use this if available
            
            # Skip if no noise norms or not labeled
            if norm_field not in data or "memorized" not in data:
                continue
                
            # Calculate the average norm across timesteps
            avg_norm = np.mean(data[norm_field])
            
            # Group all non-memorized samples together, regardless of dataset
            if not data["memorized"]:
                not_memorized_scores.append(avg_norm)
            else:
                # For memorized samples, maintain dataset distinction
                if is_objaverse:
                    objaverse_memorized_scores.append(avg_norm)
                else:
                    laion_memorized_scores.append(avg_norm)
                
        except (KeyError, FileNotFoundError):
            continue
    
    if not (objaverse_memorized_scores or laion_memorized_scores) and not not_memorized_scores:
        return None
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    max_value = max(
        max(objaverse_memorized_scores, default=0),
        max(laion_memorized_scores, default=0),
        max(not_memorized_scores, default=0)
    ) + 10
    bins = np.linspace(0, max_value, 30)
    
    # Plot all non-memorized samples together
    plt.hist(not_memorized_scores, bins=bins, alpha=0.5, label=f'Not Memorized (n={len(not_memorized_scores)})', color='green')
    
    # Plot memorized samples by dataset
    if laion_memorized_scores:
        plt.hist(laion_memorized_scores, bins=bins, alpha=0.5, label=f'LAION Memorized (n={len(laion_memorized_scores)})', color='red')
    if objaverse_memorized_scores:
        plt.hist(objaverse_memorized_scores, bins=bins, alpha=0.5, label=f'Objaverse Memorized (n={len(objaverse_memorized_scores)})', color='black')
    
    # Add vertical line for optimal threshold if provided
    if optimal_threshold is not None:
        plt.axvline(x=optimal_threshold, color='blue', linestyle='--', 
                   label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    plt.xlabel('Average Noise Norm')
    plt.ylabel('Frequency')
    plt.title('AnimateDiff - Distribution of Average Noise Norms by Memorization Status')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return output_file

def plot_precision_recall_curve(inputs, output_file="precision_recall.png"):
    """Create and save precision-recall curve."""
    json_files = inputs[:700]
    
    # Store the average norms and their memorization labels
    avg_norms = []
    labels = []
    
    for json_file in json_files:
        try:
            data = load_json_data(json_file)
            
            # Check which norm field to use
            norm_field = "text_noise_norms"
            if "noise_diff_norms" in data:
                norm_field = "noise_diff_norms"
                
            # Skip if no noise norms or not labeled
            if norm_field not in data or "memorized" not in data:
                continue
                
            # Calculate the average norm across timesteps
            avg_norm = np.mean(data[norm_field])
            avg_norms.append(avg_norm)
            labels.append(1 if data["memorized"] else 0)
                
        except (KeyError, FileNotFoundError):
            continue
    
    if not avg_norms or not labels or len(set(labels)) < 2:
        return None
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(labels, avg_norms)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    
    # Add labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AnimateDiff - Precision-Recall Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()
    
    return output_file




def create_aggregated_plot(inputs, output_file="aggregated_plot.png"):
    json_files = inputs[:700]
    
    # Create four separate groups based on memorization status and path
    second_stage_memorized_norms = []
    second_stage_not_memorized_norms = []
    standard_memorized_norms = []
    standard_not_memorized_norms = []
    
    labeled_count = 0
    unlabeled_count = 0
    
    for json_file in json_files:
        try:
            data = load_json_data(json_file)
            is_second_stage = "clusters" in str(json_file)
            
            if "memorized" in data:
                labeled_count += 1
                if data["memorized"]:
                    if is_second_stage:
                        second_stage_memorized_norms.append(data["noise_diff_norms"])
                    else:
                        standard_memorized_norms.append(data["noise_diff_norms"])
                else:
                    if is_second_stage:
                        second_stage_not_memorized_norms.append(data["noise_diff_norms"])
                    else:
                        standard_not_memorized_norms.append(data["noise_diff_norms"])
            else:
                # unlabeled_count += 1
                labeled_count += 1
                if is_second_stage:
                    second_stage_not_memorized_norms.append(data["noise_diff_norms"])
                else:
                    standard_not_memorized_norms.append(data["noise_diff_norms"])

        except (KeyError, FileNotFoundError):
            continue
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Only create plot if we have data
    if any([second_stage_memorized_norms, second_stage_not_memorized_norms, 
            standard_memorized_norms, standard_not_memorized_norms]):
        # Get timesteps from the first available trajectory
        timesteps = None
        for norm_list in [second_stage_memorized_norms, second_stage_not_memorized_norms, 
                         standard_memorized_norms, standard_not_memorized_norms]:
            if norm_list:
                timesteps = list(range(len(norm_list[0])))
                break
        
        if timesteps is None:
            timesteps = []
        
        # Plot each group with its specific color
        legend_handles = []

        # LAION not memorized (green)
        if standard_not_memorized_norms:
            line = plt.plot(timesteps, standard_not_memorized_norms[0], color="green", alpha=0.1, linewidth=1)[0]
            for traj in standard_not_memorized_norms[1:]:
                plt.plot(timesteps, traj, color="green", alpha=0.1, linewidth=1)
            legend_handles.append((line, f"LAION Not Memorized (green)"))
        
        # WebVid10M not memorized (green)
        if second_stage_not_memorized_norms:
            line = plt.plot(timesteps, second_stage_not_memorized_norms[0], color="green", alpha=0.1, linewidth=1)[0]
            for traj in second_stage_not_memorized_norms[1:]:
                plt.plot(timesteps, traj, color="green", alpha=0.1, linewidth=1)
            legend_handles.append((line, f"WebVid10M Not Memorized (green)"))

        # LAION memorized (red)
        if standard_memorized_norms:
            line = plt.plot(timesteps, standard_memorized_norms[0], color="red", alpha=0.2, linewidth=1)[0]
            for traj in standard_memorized_norms[1:]:
                plt.plot(timesteps, traj, color="red", alpha=0.2, linewidth=1)
            legend_handles.append((line, f"LAION Memorized (red, {len(standard_memorized_norms)})"))

        # WebVid10M memorized (black)
        if second_stage_memorized_norms:
            line = plt.plot(timesteps, second_stage_memorized_norms[0], color="black", alpha=0.8, linewidth=1)[0]
            for traj in second_stage_memorized_norms[1:]:
                plt.plot(timesteps, traj, color="black", alpha=0.8, linewidth=1)
            legend_handles.append((line, f"WebVid10M Memorized (black)"))
            
        plt.title(f"AnimateDiff - Aggregated C-UC Noise Norms by Memorization Status and Dataset\n"
                # f"(Labeled: {labeled_count}, Unlabeled: {unlabeled_count})"
                )
        plt.xlabel("Denoising Step")
        plt.ylabel("Text Noise Norm")
        plt.ylim(0, 40)
        plt.grid(True)

        # Add the legend with counts
        plt.legend([h for h, l in legend_handles], 
                  [l for h, l in legend_handles], 
                  loc="upper right")

    else:
        plt.text(0.5, 0.5, "No labeled data available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
    
    plt.tight_layout()
    
    # Save to disk
    plt.savefig(output_file)
    plt.close()
    
    # Print statistics
    print(f"Aggregated plot saved to {output_file}")
    print(f"Statistics:")
    print(f"  - WebVid10M Memorized: {len(second_stage_memorized_norms)}")
    print(f"  - WebVid10M Not Memorized: {len(second_stage_not_memorized_norms)}")
    print(f"  - LAION Memorized: {len(standard_memorized_norms)}")
    print(f"  - LAION Not Memorized: {len(standard_not_memorized_norms)}")
    print(f"Total: {labeled_count} labeled, {unlabeled_count} unlabeled")
    
    # Add ROC AUC analysis after aggregated plot
    roc_auc, fpr, tpr, optimal_threshold, dataset_results = calculate_rocauc(inputs)
    
    # Create additional ROC analysis plots and export data
    results = {}
    
    if roc_auc is not None:
        # Create ROC curve plot
        roc_plot_path = plot_roc_curve(fpr, tpr, roc_auc, 
                                       output_file="roc_curve.png", 
                                       dataset_results=dataset_results)
        results["roc_plot"] = roc_plot_path
        
        # Create distribution plot
        dist_plot_path = plot_score_distributions(inputs, optimal_threshold, 
                                                 output_file="score_distribution.png")
        results["distribution_plot"] = dist_plot_path
        
        # Generate precision-recall curve
        if len(set([1 if data["memorized"] else 0 for data in 
                   [load_json_data(file) for file in json_files if "memorized" in load_json_data(file)]])) > 1:
            precision_recall_path = plot_precision_recall_curve(inputs, 
                                                              output_file="precision_recall.png")
            results["precision_recall_plot"] = precision_recall_path
    
    return output_file, roc_auc, optimal_threshold, results



def check_prompt_consistency(filename, json_data):
    """Check if the prompt in the JSON matches the filename (accounting for safe_prompt transformations)."""
    _, _, safe_prompt_from_filename = extract_info_from_filename(filename)
    
    if "prompt" not in json_data:
        return "No prompt field in JSON", ""
    
    original_prompt = json_data["prompt"]
    
    # Convert original prompt to what would be a safe_prompt
    expected_safe_prompt = re.sub(r'\W+', '_', original_prompt)
    
    if safe_prompt_from_filename and safe_prompt_from_filename in expected_safe_prompt:
        return f"Prompt matches filename: {original_prompt}", original_prompt
    else:
        return f"WARNING: Prompt mismatch! {safe_prompt_from_filename} != {expected_safe_prompt}", original_prompt

def save_mesh_as_obj(mesh, output_file=None):
    """Save mesh as OBJ file for Gradio's Model3D component."""
    if output_file is None:
        # Create a temporary file if no output file is specified
        temp_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
        output_file = temp_file.name
        temp_file.close()
    
    # Export the mesh as OBJ
    mesh.export(output_file, file_type='obj')
    
    return output_file

def find_original_asset_path(uid):
    """Find the original asset path in the WebVid10M cache."""
    cache_dir = find_second_stage_cache_dir()
    if not cache_dir:
        return None
    
    # The asset might be stored with its UID as the filename
    potential_paths = glob.glob(os.path.join(cache_dir, f"{uid}.*"))
    potential_paths += glob.glob(os.path.join(cache_dir, uid, "*.*"))
    
    for path in potential_paths:
        if os.path.exists(path) and path.lower().endswith(('.obj', '.glb', '.gltf')):
            return path
    
    return None

def app(input_files, no_3d=True):
    pairs = get_file_pairs(input_files)
    if not pairs:
        return gr.Markdown("No valid file pairs found.")
    
    current_index = 0
    current_3d_model = None
    
    def load_current_pair():
        nonlocal current_3d_model
        
        if current_index >= len(pairs):
            plot_path = create_aggregated_plot(input_files)
            return (
                None, None, "All files processed! Aggregated plot created.", 
                gr.update(value=plot_path, visible=True), 
                gr.update(visible=False), 
                gr.update(visible=False), 
                gr.update(visible=True),
                None, None
            )
        
        json_file, img_file, noise_img_file = pairs[current_index]
        data = load_json_data(json_file)
        
        index, label, safe_prompt = extract_info_from_filename(json_file)
        prompt_message, original_prompt = check_prompt_consistency(json_file, data)

        model_path = None
        asset_info = "3D asset loading disabled" if no_3d else "No 3D asset information available"

        has_memorized_field = "memorized" in data
        default_value = False
        if has_memorized_field:
            default_value = "1 (Memorized)" if data["memorized"] else "0 (Not Memorized)"

        # Gather other attributes
        other_info = "\n".join(
            [f"- **{k}**: {v}" for k, v in data.items() if k not in ["prompt", "memorized", "noise_diff_norms"]]
        )

        file_info_text = f"""
        **File {current_index + 1}/{len(pairs)}**: `{os.path.basename(json_file)}`

        **Prompt Check**: {prompt_message}

        **Other JSON Attributes**:   
        {other_info if other_info else "No additional metadata"}
        """

        return (
            img_file, 
            noise_img_file, 
            file_info_text, 
            gr.update(value=None, visible=False),
            gr.update(value=default_value, visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            model_path if not no_3d else None,
            gr.update(visible=bool(model_path) if not no_3d else False)
        )
    
    def submit_label(choice):
        nonlocal current_index
        json_file = pairs[current_index][0]
        update_memorization_status(json_file, choice)
        current_index += 1
        return load_current_pair()
    
    def skip():
        nonlocal current_index
        current_index += 1
        return load_current_pair()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Memorization Labeling App with 3D Asset Visualization")
        gr.Markdown("Label images as memorized (1) or not memorized (0) and compare with 3D assets")
        
        with gr.Row():
            file_info = gr.Markdown("Loading...")
        
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Image", show_label=True)
                noise_image = gr.Image(label="C-UC Norm", show_label=True)
        
        with gr.Row():
            choice = gr.Radio(["0 (Not Memorized)", "1 (Memorized)"], label="Memorization Status")
        
        with gr.Row():
            submit_btn = gr.Button("Submit")
            skip_btn = gr.Button("Skip")
        
        with gr.Row():
            with gr.Column(scale=1):
                plot_display = gr.Image(label="Aggregated Plot", visible=False)
            
            with gr.Column(scale=1):
                asset_final = gr.Markdown(visible=False)
        
        # Initialize the UI with the first pair
        demo.load(
            load_current_pair, 
            [], 
            [image, noise_image, file_info, plot_display, choice, submit_btn, plot_display, noise_image, noise_image]
        )
        
        # Set up event handlers
        submit_btn.click(
            fn=lambda c: submit_label(1 if c == "1 (Memorized)" else 0), 
            inputs=[choice], 
            outputs=[image, noise_image, file_info, plot_display, choice, submit_btn, plot_display, plot_display, plot_display]
        )
        
        skip_btn.click(
            fn=skip, 
            inputs=[], 
            outputs=[image, noise_image, file_info, plot_display, choice, submit_btn, plot_display, plot_display, plot_display]
        )
    
    return demo

def main():
    parser = argparse.ArgumentParser(description="Memorization Labeling App with 3D Asset Visualization")
    parser.add_argument("--inputs", "-i", nargs='+',
                        type=str, help="Regex of saved json files from attacks", 
                        default="/scratch/local/ssd/anjun/memorization/AnimateDiff/samples/1_1_animate_RealisticVision-2025-04-16T14-44-40/*/mse/*.json")
    parser.add_argument("--port", "-d", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--output", "-o", type=str, default="aggregated_plot.png", help="Output file for the aggregated plot")
    parser.add_argument("--plot-only", action="store_true", help="Only generate the aggregated plot without launching the UI")
    parser.add_argument("--no-3d", action="store_false", default=True, help="Disable 3D model display")
    
    args = parser.parse_args()
    
    # Check if only plotting is requested
    if args.plot_only:
        create_aggregated_plot(args.inputs, args.output)
        return
    
    # Otherwise, launch the labeling app
    demo = app(args.inputs)
    demo.launch(server_port=args.port)

if __name__ == "__main__":
    main()