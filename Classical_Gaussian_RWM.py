import numpy as np
import matplotlib.pyplot as plt

def target_density(x):
    """Multivariate normal target density in 100 dimensions."""
    return np.exp(-0.5 * np.sum(x ** 2))

def random_walk_mh(n_samples, burn_in, initial_value, proposal_scale):
    """Random Walk Metropolis-Hastings algorithm."""
    n_dim = len(initial_value)
    samples = np.zeros((n_samples, n_dim))
    samples[0] = initial_value
    accepted = 0
    esjd_sum = 0
    
    for i in range(1, n_samples):
        current_sample = samples[i-1]
        # Propose a new sample using Gaussian random walk
        proposal = current_sample + proposal_scale * np.random.normal(size=n_dim)
        
        # Calculate the acceptance probability
        acceptance_prob = min(1, target_density(proposal) / target_density(current_sample))
        
        # Accept or reject the proposed sample
        if np.random.rand() < acceptance_prob:
            samples[i] = proposal
            accepted += 1
            # Only calculate ESJD after burn-in
            if i > burn_in:
                esjd_sum += np.sum((proposal - current_sample) ** 2)
        else:
            samples[i] = current_sample
    
    # Compute acceptance rate and ESJD
    acceptance_rate = accepted / n_samples
    esjd = esjd_sum / (accepted - burn_in) if accepted > burn_in else 0
    return samples, acceptance_rate, esjd

# Parameters
n_samples = 30000  # Increase the number of samples
burn_in = 5000     # Discard the first 5000 samples for burn-in
initial_value = np.zeros(300)  # Starting point in 300 dimensions
dim = len(initial_value)

# Test a larger range of proposal scales
#2.38 is the optimal value
proposal_scales = [0.38, 1.38,2.38,3.38,4.38,5.38]
acceptance_rates = []
esjds = []

for proposal_scale in proposal_scales:
    _, acceptance_rate, esjd = random_walk_mh(n_samples, burn_in, initial_value, proposal_scale/np.sqrt(300))
    acceptance_rates.append(acceptance_rate)
    esjds.append(esjd)
    print(f"Proposal Scale: {proposal_scale:.3f}, Acceptance Rate: {acceptance_rate:.3f}, ESJD: {esjd:.3f}")

# -----------------------------------------------------------
# Plot Acceptance Rate vs. Proposal Scale
# -----------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(proposal_scales, acceptance_rates, marker='o', label="Acceptance Rate", color='blue')
plt.axhline(y=0.234, color='r', linestyle='--', label="Target Acceptance Rate (0.234)")
plt.xlabel("Proposal Scale")
plt.ylabel("Acceptance Rate")
plt.title("Acceptance Rate vs. Proposal Scale")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# Plot ESJD vs. Proposal Scale
# -----------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(proposal_scales, esjds, marker='o', label="ESJD", color='green')
plt.xlabel("Proposal Scale")
plt.ylabel("Expected Squared Jumping Distance (ESJD)")
plt.title("ESJD vs. Proposal Scale")
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# Plot ESJD vs. Acceptance Rate
# -----------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(acceptance_rates, esjds, marker='o', label="ESJD", color='purple')
plt.axvline(x=0.234, color='r', linestyle='--', label="Target Acceptance Rate (0.234)")
plt.xlabel("Acceptance Rate")
plt.ylabel("Expected Squared Jumping Distance (ESJD)")
plt.title("ESJD vs. Acceptance Rate")
plt.legend()
plt.grid(True)
plt.show()
