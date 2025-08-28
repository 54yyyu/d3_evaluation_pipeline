import numpy as np
import torch
from datetime import datetime
import pickle
from tqdm import tqdm
from captum.attr import GradientShap
import matplotlib.pyplot as plt
import logomaker

# Global variables for spherical coordinate calculations
N_EXP = None
volume_border_correction = None
Empirical_box_pdf_s = []
Empirical_box_count_s = []
Empirical_box_count_plain_s = []

def gradient_shap(x_seq, model, class_index=0, trim_end=None):
    """Compute gradient SHAP scores for sequence attribution."""
    # Detect device of the model
    device = next(model.parameters()).device
    
    # Convert tensor to numpy if needed
    if torch.is_tensor(x_seq):
        x_seq = x_seq.detach().cpu().numpy()
    
    x_seq = np.swapaxes(x_seq,1,2)
    N,A,L = x_seq.shape
    score_cache = []
    for i,x in tqdm(enumerate(x_seq), desc="Computing SHAP scores", total=len(x_seq)):
        # process sequences so that they are right shape (based on insertions)
        x = np.expand_dims(x, axis=0)
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32).to(device)
        #x_tensor = model._pad_end(x_tensor)
        x = x_tensor.cpu().detach().numpy()
        # random background
        num_background = 1000
        null_index = np.random.randint(0,3, size=(num_background,L))
        x_null = np.zeros((num_background,A,L))
        for n in range(num_background):
            for l in range(L):
                x_null[n,null_index[n,l],l] = 1.0
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32).to(device)
        #x_null_tensor = model._pad_end(x_null_tensor)
        # calculate gradient shap
        gradient_shap = GradientShap(model)
        grad = gradient_shap.attribute(x_tensor,
                                      n_samples=100,
                                      stdevs=0.1,
                                      baselines=x_null_tensor,
                                      target=class_index)
        grad = grad.data.cpu().numpy()
        # process gradients with gradient correction (Majdandzic et al. 2022)
        grad -= np.mean(grad, axis=1, keepdims=True)
        score_cache.append(np.squeeze(grad))
    score_cache = np.array(score_cache)
    if len(score_cache.shape)<3:
        score_cache=np.expand_dims(score_cache,axis=0)
    if trim_end:
        score_cache = score_cache[:,:,:-trim_end]
    return np.swapaxes(score_cache,1,2)

def plot_attribution_map(x_seq, shap_score, alphabet='ACGT', figsize=(20,1)):
    """Plot attribution map for visualization."""
    num_plot = len(x_seq)
    fig = plt.figure(figsize=(20,2*num_plot))
    i = 0
    i_subplot = 0
    for (x,grad) in zip(x_seq,shap_score):
        
        x_index = np.argmax(np.squeeze(x), axis=1)
        grad = np.squeeze(grad)
        L, A = grad.shape

        seq = ''
        saliency = np.zeros((L))
        for i in range(L):
            seq += alphabet[x_index[i]]
            saliency[i] = grad[i,x_index[i]]
        # create saliency matrix
        saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)

        ax = plt.subplot(num_plot,1,i_subplot+1)
        i_subplot+=1
        logomaker.Logo(saliency_df, figsize=figsize, ax=ax)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])

def process_attribution_map(saliency_map_raw, k=6):
    """Process attribution map with k-mer aggregation and orthonormal coordinates."""
    saliency_map_raw = saliency_map_raw - np.mean(saliency_map_raw, axis=-1, keepdims=True) # gradient correction
    saliency_map_raw = saliency_map_raw / np.sum(np.sqrt(np.sum(np.square(saliency_map_raw), axis=-1, keepdims=True)), axis=-2, keepdims=True) #normaliz
    saliency_special_map_k = saliency_map_raw.copy()
    for i in range(k-1): 
        saliency_map_raw_rolled_i = np.roll(saliency_map_raw, -i-1, axis=-2)
        saliency_special_map_k += saliency_map_raw_rolled_i
    saliency_special = orthonormal_coordinates(saliency_special_map_k)
    return saliency_special

def orthonormal_coordinates(attr_map):
    """Reduce 4d array to 3d using orthonormal coordinates."""
    attr_map_on = np.zeros((attr_map.shape[0], attr_map.shape[1], 3))

    x = attr_map[:, :, 0]
    y = attr_map[:, :, 1]
    z = attr_map[:, :, 2]
    w = attr_map[:, :, 3]

    # Now convert to new coordinates
    e1 = 1 / np.sqrt(2) * (-x + y)
    e2 = np.sqrt(2 / 3) * (-1/2*x -1/2*y)
    e3 = np.sqrt(3 / 4) * (-1/3*x -1/3*y -1/3*z + w)
    attr_map_on[:, :, 0] = e1
    attr_map_on[:, :, 1] = e2
    attr_map_on[:, :, 2] = e3

    return attr_map_on

def unit_mask(x_seq):
    """Create unit mask for sequences."""
    # Convert tensor to numpy if needed
    if torch.is_tensor(x_seq):
        x_seq = x_seq.detach().cpu().numpy()
    return np.sum(np.ones(x_seq.shape),axis=-1) / 4

def spherical_coordinates_process_2_trad(saliency_map_raw_s, X, mask, radius_count_cutoff=0.04):
    """Process spherical coordinates with traditional method."""
    # Convert tensor to numpy if needed
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    
    global N_EXP
    N_EXP = len(saliency_map_raw_s)
    radius_count=int(radius_count_cutoff * np.prod(X.shape)/4)
    cutoff=[]
    x_s, y_s, z_s, r_s, phi_1_s, phi_2_s = [], [], [], [], [], []
    PI = 3.1416
    for s in range (0, N_EXP):
        saliency_map_raw = saliency_map_raw_s[s]
        xxx_motif=saliency_map_raw[:,:,0]
        yyy_motif=(saliency_map_raw[:,:,1])
        zzz_motif=(saliency_map_raw[:,:,2])
        xxx_motif_pattern=saliency_map_raw[:,:,0]*mask
        yyy_motif_pattern=(saliency_map_raw[:,:,1])*mask
        zzz_motif_pattern=(saliency_map_raw[:,:,2])*mask
        r=np.sqrt(xxx_motif*xxx_motif+yyy_motif*yyy_motif+zzz_motif*zzz_motif)
        resh = X.shape[0] * X.shape[1]
        x=np.array(xxx_motif_pattern.reshape(resh,))
        y=np.array(yyy_motif_pattern.reshape(resh,))
        z=np.array(zzz_motif_pattern.reshape(resh,))
        r=np.array(r.reshape(resh,))
        #Take care of any NANs.
        x=np.nan_to_num(x)
        y=np.nan_to_num(y)
        z=np.nan_to_num(z)
        r=np.nan_to_num(r)
        cutoff.append( np.sort(r)[-radius_count] )
        R_cuttof_index = np.sqrt(x*x+y*y+z*z) > cutoff[s]
        #Cut off
        x=x[R_cuttof_index]
        y=y[R_cuttof_index]
        z=z[R_cuttof_index]
        r=np.array(r[R_cuttof_index])
        x_s.append(x)
        y_s.append(y)
        z_s.append(z)
        r_s.append(r)
        #rotate axis
        x__ = np.array(y)
        y__ = np.array(z)
        z__ = np.array(x)
        x = x__
        y = y__
        z = z__
        #"phi"
        phi_1 = np.arctan(y/x) #default
        phi_1 = np.where((x<0) & (y>=0), np.arctan(y/x) + PI, phi_1)   #overwrite
        phi_1 = np.where((x<0) & (y<0), np.arctan(y/x) - PI, phi_1)   #overwrite
        phi_1 = np.where (x==0, PI/2, phi_1) #overwrite
        #Renormalize temorarily to have both angles in [0,PI]:
        phi_1 = phi_1/2 + PI/2
        #"theta"
        phi_2=np.arccos(z/r)
        #back to list
        phi_1 = list(phi_1)
        phi_2 = list(phi_2)
        phi_1_s.append(phi_1)
        phi_2_s.append(phi_2)
    #print(cutoff)
    return phi_1_s, phi_2_s, r_s

def initialize_integration_2(box_length):
    """Initialize integration parameters."""
    LIM = 3.1416
    global volume_border_correction
    box_volume = box_length*box_length
    n_bins = int(LIM/box_length)
    volume_border_correction =  (LIM/box_length/n_bins)*(LIM/box_length/n_bins)
    #print('volume_border_correction = ', volume_border_correction)
    n_bins_half = int(n_bins/2)
    return LIM, box_length, box_volume, n_bins, n_bins_half

def calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, box_length, box_volume, prior_range):
    """Calculate entropy using KL divergence."""
    global Empirical_box_pdf_s
    global Empirical_box_count_s
    global Empirical_box_count_plain_s
    Empirical_box_pdf_s=[]
    Empirical_box_count_s = []
    Empirical_box_count_plain_s = []
    prior_correction_s = []
    Spherical_box_prior_pdf_s=[]
    for s in range (0,N_EXP):
        #print(s)
        Empirical_box_pdf_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[0])
        Empirical_box_count_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[1])
        Empirical_box_count_plain_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[2])
    Entropic_information = []
    for s in range (0,N_EXP):
        Entropic_information.append ( KL_divergence_2 (Empirical_box_pdf_s[s], Empirical_box_count_s[s], Empirical_box_count_plain_s[s], n_bins, box_volume, prior_range)  )
    return list(Entropic_information)

def KL_divergence_2(Empirical_box_pdf, Empirical_box_count, Empirical_box_count_plain, n_bins, box_volume, prior_range):
    """Calculate KL divergence with spherical prior."""
    # p= empirical distribution, q=prior spherical distribution
    # Notice that the prior distribution is never 0! So it is safe to divide by q.
    # L'Hospital rule provides that p*log(p) --> 0 when p->0. When we encounter p=0, we would just set the contribution of that term to 0, i.e. ignore it in the sum.
    Relative_entropy = 0
    PI = 3.1416
    for i in range (1, n_bins-1):
        for j in range(1,n_bins-1):
            if (Empirical_box_pdf[i,j] > 0  ):
                phi_1 = i/n_bins*PI
                phi_2 = j/n_bins*PI
                correction3 = 0
                prior_counter = 0
                prior=0
                for ii in range(-prior_range,prior_range):
                    for jj in range(-prior_range,prior_range):
                        if(i+ii>0 and i+ii<n_bins and j+jj>0 and j+jj<n_bins):
                            prior+=Empirical_box_pdf[i+ii,j+jj]
                            prior_counter+=1
                prior=prior/prior_counter
                if(prior>0) : KL_divergence_contribution = Empirical_box_pdf[i,j] * np.log (Empirical_box_pdf[i,j]  /  prior )
                if(np.sin(phi_1)>0 and prior>0 ): Relative_entropy+=KL_divergence_contribution  #and Empirical_box_count_plain[i,j]>1
    Relative_entropy = Relative_entropy * box_volume #(volume differential in the "integral")
    return np.round(Relative_entropy,3)

def Empiciral_box_pdf_func_2(phi_1, phi_2, r_s, n_bins, box_length, box_volume):
    """Compute empirical box PDF function."""
    N_points = len(phi_1) #Number of points
    Empirical_box_count = np.zeros((n_bins, n_bins))
    Empirical_box_count_plain = np.zeros((n_bins, n_bins))
    #Now populate the box. Go over every single point.
    for i in range (0, N_points):
        # k, l are box numbers of the (phi_1, phi_2) point
        k=np.minimum(int(phi_1[i]/box_length), n_bins-1)
        l=np.minimum(int(phi_2[i]/box_length), n_bins-1)
        #Increment count in (k,l,m) box:
        Empirical_box_count[k,l]+=1*r_s[i]*r_s[i]
        Empirical_box_count_plain[k,l]+=1
    #To get the probability distribution, divide the Empirical_box_count by the total number of points.
    Empirical_box_pdf = Empirical_box_count / N_points / box_volume
    #Check that it integrates to around 1:
    #print('Integral of the empirical_box_pdf (before first renormalization) = ' , np.sum(Empirical_box_pdf*box_volume), '(should be 1.0 if OK) \n')
    correction = 1 / np.sum(Empirical_box_pdf*box_volume)
    #Another, optional correction 
    count_empty_boxes = 0
    count_single_points = 0
    for k in range (1, n_bins-1):
        for l in range(1,n_bins-1):
            if(Empirical_box_count[k,l] ==1):
                count_empty_boxes+=1
                count_single_points+=1
    return Empirical_box_pdf * correction * 1 , Empirical_box_count *correction , Empirical_box_count_plain #, correction2

def run_attribution_consistency_analysis(deepstarr, sample_seqs, X_test, output_dir=".", sample_name=None):
    """
    Run attribution consistency analysis on sample sequences and test data.
    
    Args:
        deepstarr: The DeepSTARR model
        sample_seqs: Sample sequences tensor (N, L, A)
        X_test: Test sequences tensor (N, L, A)  
        output_dir: Directory to save results
        sample_name: Name of sample for batch processing (optional)
        
    Returns:
        dict: Results dictionary with entropic information
    """
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
    print("Computing SHAP scores for top 2000 sequences...")
    shap_score_top_sampled = gradient_shap(top_sampled_seqs, deepstarr)
    
    print("Processing attribution maps...")
    attribution_map_top_sampled = process_attribution_map(shap_score_top_sampled, k=6)
    mask_top_sampled = unit_mask(top_sampled_seqs)

    # Entropic information for top sampled sequences
    print("Computing spherical coordinates for top sequences...")
    phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad([attribution_map_top_sampled], 
                                                                 top_sampled_seqs, 
                                                                 mask_top_sampled, 
                                                                 radius_count_cutoff=0.04)
    
    LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
    entropic_information_top_sampled = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3)
    
    # Consistency across generated and observed sequence
    concatenated_seqs = torch.cat((X_test, sample_seqs), dim=0)
    print(f"Computing SHAP scores for {len(concatenated_seqs)} concatenated sequences...")
    shap_score_concatenated = gradient_shap(concatenated_seqs, deepstarr)
    
    print("Processing attribution maps for concatenated sequences...")
    attribution_map_concatenated = process_attribution_map(shap_score_concatenated, k=6)
    mask_concatenated = unit_mask(concatenated_seqs)

    print("Computing spherical coordinates for concatenated sequences...")
    phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad([attribution_map_concatenated], 
                                                                 concatenated_seqs, 
                                                                 mask_concatenated, 
                                                                 radius_count_cutoff=0.04)
    
    LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
    entropic_information_concatenated = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3)
    
    # Create results dictionary with concise metric names
    results = {
        'entropic information of top 2000 activity sampled sequences': entropic_information_top_sampled,
        'entropic information of concatenated sequences': entropic_information_concatenated,
        'KLD': entropic_information_top_sampled[0] if entropic_information_top_sampled else 0,  # First value for concise metrics
        'KLD_concat': entropic_information_concatenated[0] if entropic_information_concatenated else 0  # First value for concise metrics
    }
    
    # Handle batch vs single mode
    if sample_name is not None:
        # Batch mode - use new format
        from utils.batch_helpers import write_concise_csv, write_full_h5, get_concise_metrics
        
        # Write concise metrics
        concise_metrics = get_concise_metrics('attribution_consistency', results)
        write_concise_csv(output_dir, 'attribution_consistency', sample_name, concise_metrics)
        
        # Write full results
        write_full_h5(output_dir, 'attribution_consistency', sample_name, results)
        
        print(f"Attribution consistency results saved for sample '{sample_name}'")
    else:
        # Single mode - keep original format
        filename = f'{output_dir}/attribution_consistency_{current_date}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Attribution consistency results saved to '{filename}'")
    
    return results