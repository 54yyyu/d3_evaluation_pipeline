import numpy as np
import torch
from datetime import datetime
import pickle

def run_attribution_consistency_analysis(deepstarr, sample_seqs, X_test, output_dir="."):
    """
    Run attribution consistency analysis on sample sequences and test data.
    
    Args:
        deepstarr: The DeepSTARR model
        sample_seqs: Sample sequences tensor (N, L, A)
        X_test: Test sequences tensor (N, L, A)  
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with entropic information
    """
    # Import required functions from utils
    from utils.seq_evals_improved import gradient_shap, process_attribution_map, unit_mask
    from utils.seq_evals_improved import spherical_coordinates_process_2_trad, initialize_integration_2, calculate_entropy_2
    
    # Get current timestamp
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Ensure tensors are on the same device as the model
    device = next(deepstarr.parameters()).device
    sample_seqs = sample_seqs.to(device)
    X_test = X_test.to(device)
    
    # Top 2,000 functional activity sampled sequence
    activity_sample_seqs = deepstarr(sample_seqs.permute(0,2,1))
    samples_total_activity = activity_sample_seqs.sum(dim=1)
    sorted_indices = torch.argsort(samples_total_activity, descending=True)
    top_sampled_seqs = sample_seqs[sorted_indices[:2000]]
    
    # SHAP score for top activity sequences
    shap_score_top_sampled = gradient_shap(top_sampled_seqs, deepstarr)
    attribution_map_top_sampled = process_attribution_map(shap_score_top_sampled, k=6)
    mask_top_sampled = unit_mask(top_sampled_seqs)

    # Entropic information for top sampled sequences
    phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad([attribution_map_top_sampled], 
                                                                 top_sampled_seqs, 
                                                                 mask_top_sampled, 
                                                                 radius_count_cutoff=0.04)
    
    LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
    entropic_information_top_sampled = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3)
    
    # Consistency across generated and observed sequence
    concatenated_seqs = torch.cat((X_test, sample_seqs), dim=0)
    shap_score_concatenated = gradient_shap(concatenated_seqs, deepstarr)
    attribution_map_concatenated = process_attribution_map(shap_score_concatenated, k=6)
    mask_concatenated = unit_mask(concatenated_seqs)

    phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad([attribution_map_concatenated], 
                                                                 concatenated_seqs, 
                                                                 mask_concatenated, 
                                                                 radius_count_cutoff=0.04)
    
    LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
    entropic_information_concatenated = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3)
    
    # Create results dictionary
    results = {
        'entropic information of top 2000 activity sampled sequences': entropic_information_top_sampled,
        'entropic information of concatenated sequences': entropic_information_concatenated
    }
    
    # Save results
    filename = f'{output_dir}/attribution_consistency_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Attribution consistency results saved to '{filename}'")
    return results