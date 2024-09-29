import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.special import gammainc, erf

# Set the working directory
os.chdir('/Users/patricianing/Desktop/new')

np.random.seed(42)

def generalized_normal_pdf(x, mu, alpha, beta):
    """Multivariate Generalized Normal Distribution with parameters mu, alpha, and beta."""
    coef = beta / (2 * alpha * gamma(1 / beta))
    exponent = -np.sum((np.abs((x - mu) / alpha)) ** beta)
    return coef * np.exp(exponent)

def target_density(x):
    """Target density for the multivariate generalized normal distribution with mu=0, alpha=1, beta=3."""
    mu = 0
    alpha = 1
    beta = 3
    return generalized_normal_pdf(x, mu, alpha, beta)

def random_walk_mh(n_samples, burn_in, initial_value, proposal_scale):
    """Random Walk Metropolis-Hastings algorithm with acceptance counting after burn-in."""
    n_dim = len(initial_value)
    samples = np.zeros((n_samples, n_dim))
    samples[0] = initial_value
    accepted_after_burnin = 0
    esjd_sum = 0
    
    for i in range(1, n_samples):
        current_sample = samples[i - 1]
        proposal = current_sample + proposal_scale * np.random.normal(size=n_dim)
        acceptance_prob = min(1, target_density(proposal) / target_density(current_sample))
        
        if np.random.rand() < acceptance_prob:
            samples[i] = proposal
            if i >= burn_in:
                accepted_after_burnin += 1
        else:
            samples[i] = current_sample

        if i >= burn_in:
            esjd_sum += np.sum((samples[i] - samples[i - 1])**2)

    # Calculate acceptance rate after burn-in
    acceptance_rate = accepted_after_burnin / (n_samples - burn_in)
    esjd = esjd_sum / (n_samples - burn_in)
    
    return samples[burn_in:], acceptance_rate, esjd, samples

#########################################################################################################

def plot_traces(chains, n_samples, burn_in):
    """Plot the trace plots for Markov chains."""
    plt.figure(figsize=(12, 8))
    
    for chain_id, chain in enumerate(chains):
        plt.subplot(2, 2, chain_id + 1)
        plt.plot(range(burn_in, n_samples), chain[:, 0], label=f'Chain {chain_id + 1}')
        plt.title(f'Trace Plot for Chain {chain_id + 1}', fontsize=18)
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel('Value', fontsize=16)
        plt.grid(True)
    
    plt.tight_layout()


#########################################################################################################

def generalized_normal_cdf(x, mu, alpha, beta):
    """Compute the CDF of the Generalized Normal Distribution."""
    if beta == 2:
        # Standard normal CDF for beta=2 (Gaussian case)
        return 0.5 * (1 + erf((x - mu) / (alpha * np.sqrt(2))))
    else:
        # Generalized normal CDF for beta != 2 using the incomplete gamma function
        z = np.abs(x - mu) / alpha
        gamma_term = gammainc(1 / beta, z ** beta)
        return 0.5 * (1 + np.sign(x - mu) * gamma_term)

def empirical_cdf(samples):
    """Compute the empirical CDF of a set of samples."""
    sorted_samples = np.sort(samples)
    cdf_values = np.arange(1, len(samples) + 1) / len(samples)
    return sorted_samples, cdf_values

def plot_cdf(chains, mu=0, alpha=1, beta=3, dim=0):
    """
    Plot the empirical CDF of multiple chains for a specific dimension (dim) and compare it with the theoretical CDF.
    
    Parameters:
    chains -- List of 2D arrays where each array corresponds to samples from a chain (shape: num_samples x num_dimensions).
    mu -- Location parameter for the generalized normal distribution.
    alpha -- Scale parameter for the generalized normal distribution.
    beta -- Shape parameter for the generalized normal distribution.
    dim -- The dimension of the chain to be used for plotting the CDF (default is 0, the first dimension).
    """
    plt.figure(figsize=(10, 6))
    
    # Create a range of values for the theoretical CDF
    x_vals = np.linspace(-10, 10, 1000)
    theoretical_cdf_vals = generalized_normal_cdf(x_vals, mu, alpha, beta)
    
    # Plot theoretical CDF
    plt.plot(x_vals, theoretical_cdf_vals, label='Theoretical CDF', color='black', linestyle='--')
    
    # Plot empirical CDFs for each chain (specified dimension)
    for chain_id, chain in enumerate(chains):
        # Extract samples from the specified dimension
        chain_dim_samples = chain[:, dim]
        sorted_samples, emp_cdf_vals = empirical_cdf(chain_dim_samples)
        plt.step(sorted_samples, emp_cdf_vals, label=f'Chain {chain_id + 1} Empirical CDF', where='post')
    
    # Enlarge labels and title
    plt.xlabel('x', fontsize=16)
    plt.ylabel('CDF', fontsize=16)
    plt.title(f'Empirical CDF vs Theoretical CDF', fontsize=18)
    
    plt.legend()
    plt.grid(True)


#########################################################################################################

def plot_acceptance_probabilities(proposal_scales, n_samples, burn_in, initial_value, n_chains):
    """Plot acceptance rates for 4 chains with different proposal scales."""
    all_acceptance_rates = []
    
    # Calculate acceptance rates for each chain at different proposal scales
    for chain_id in range(n_chains):
        acceptance_rates = []
        for scale in proposal_scales:
            _, acc_rate, _, _ = random_walk_mh(n_samples, burn_in, initial_value, scale/np.sqrt(dim))
            acceptance_rates.append(acc_rate)
        all_acceptance_rates.append(acceptance_rates)
    
    # Plot acceptance rates vs. Proposal Scale for each chain
    plt.figure(figsize=(10, 6))
    
    for chain_id in range(n_chains):
        plt.plot(proposal_scales, all_acceptance_rates[chain_id], label=f'Chain {chain_id + 1}', marker='o')
    
    # Mark the line for acceptance rate of 0.234
    plt.axhline(y=0.234, color='r', linestyle='--', label='Acceptance Rate = 0.234')
    
    plt.xlabel('Proposal Scale', fontsize=16)
    plt.ylabel('Acceptance Rate', fontsize=16)
    plt.title('Acceptance Rate vs. Proposal Scale for 4 Chains', fontsize=18)
    plt.legend()
    plt.grid(True)


#########################################################################################################

def plot_esjd(proposal_scales, n_samples, burn_in, initial_value, n_chains):
    """Plot ESJD for 4 chains with different proposal scales."""
    all_esjd = []
    
    # For each chain, calculate the ESJD for different proposal scales
    for chain_id in range(n_chains):
        esjd_values = []
        for scale in proposal_scales:
            _, _, esjd, _ = random_walk_mh(n_samples, burn_in, initial_value, scale/np.sqrt(dim))
            esjd_values.append(esjd)
        all_esjd.append(esjd_values)
    
    # Plot ESJD vs. Proposal Scale for each chain
    plt.figure(figsize=(10, 6))
    
    for chain_id in range(n_chains):
        plt.plot(proposal_scales, all_esjd[chain_id], label=f'Chain {chain_id + 1}', marker='o')
    
    plt.xlabel('Proposal Scale', fontsize=16)
    plt.ylabel('ESJD (Expected Squared Jump Distance)', fontsize=16)
    plt.title('ESJD vs. Proposal Scale for 4 Chains', fontsize=18)
    plt.legend()
    plt.grid(True)


#########################################################################################################
    
# Parameters
n_samples = 20000
burn_in = 5000
initial_value = np.zeros(100)
dim=len(initial_value)
n_chains = 4
proposal_scales = [0.3666, 1.366, 2.366, 3.366, 4.366]
#optimal value 2.38/(3*np.sqrt(gamma(5/3)/gamma(1/3))) = 1.366

# Run MCMC with fixed proposal scale (1.366) for trace and CDF plots
proposal_scale = 1.366
chains = []
for _ in range(n_chains):
    samples, _, _, full_samples = random_walk_mh(n_samples, burn_in, initial_value, proposal_scale/np.sqrt(dim))
    chains.append(samples)

# Plot trace plots for the first dimension of 4 chains
plot_traces(chains, n_samples, burn_in)
# Save the plot as an image file (e.g., PNG, JPG, PDF)
plt.savefig('trace.png', dpi=300)  # You can specify the format and DPI (resolution)
plt.show()

# Plot the empirical CDFs for the first dimension of 4 chains
plot_cdf(chains)
# Save the plot as an image file (e.g., PNG, JPG, PDF)
plt.savefig('cdf.png', dpi=300)  # You can specify the format and DPI (resolution)
plt.show()

# Plot the acceptance probabilities for the four chains across varying proposal scales.
plot_acceptance_probabilities(proposal_scales, n_samples, burn_in, initial_value, n_chains)
# Save the plot as an image file (e.g., PNG, JPG, PDF)
plt.savefig('accept_prob.png', dpi=300)  # You can specify the format and DPI (resolution)
plt.show()

# Plot the ESJD for the four chains across varying proposal scales.
plot_esjd(proposal_scales, n_samples, burn_in, initial_value, n_chains)
# Save the plot as an image file (e.g., PNG, JPG, PDF)
plt.savefig('ESJD.png', dpi=300)  # You can specify the format and DPI (resolution)
plt.show()


