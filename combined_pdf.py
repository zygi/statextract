import numpy as np
from scipy.signal import fftconvolve

def compute_combined_pdf(vals: list[float], pts=100):
    """
    Computes the probability density function (pdf) of the mean of uniform distributions
    defined by upper bounds in vals.

    Parameters:
    - vals: List of upper bounds (floats) for the uniform distributions.
    - pts: Number of points for discretization (int). Default is 100.

    Returns:
    - x_mean: Numpy array of x-values (mean values).
    - pdf_mean: Numpy array of corresponding pdf values.
    """
    k = 2  # As given in your problem statement
    n = len(vals)
    
    # Define the intervals for each uniform distribution
    intervals = [(v / k, v) for v in vals]
    
    # Determine the range for the sum of variables
    sum_min = sum(interval[0] for interval in intervals)
    sum_max = sum(interval[1] for interval in intervals)
    
    # Define a common grid for discretization
    num_points = pts
    x = np.linspace(sum_min, sum_max, num_points)
    dx = x[1] - x[0]
    
    # Initialize the pdf for the first uniform distribution
    r1, v1 = intervals[0]
    pdf = np.zeros_like(x)
    idx_start = np.searchsorted(x, r1)
    idx_end = np.searchsorted(x, v1)
    pdf[idx_start:idx_end] = 1 / (v1 - r1)
    
    # Perform convolution iteratively for remaining distributions
    for r_i, v_i in intervals[1:]:
        pdf_i = np.zeros_like(x)
        idx_start_i = np.searchsorted(x, r_i)
        idx_end_i = np.searchsorted(x, v_i)
        pdf_i[idx_start_i:idx_end_i] = 1 / (v_i - r_i)
        
        # Convolve using FFT for efficiency
        pdf = fftconvolve(pdf, pdf_i, mode='full') * dx
        # Update x-axis after convolution
        x = np.linspace(x[0] + x[0], x[-1] + x[-1], len(pdf))
    
    # Adjust for the mean
    x_mean = x / n
    pdf_mean = pdf * n
    
    # Normalize the pdf
    area = np.trapz(pdf_mean, x_mean)
    if area != 0:
        pdf_mean /= area
    else:
        raise ValueError("The area under the PDF is zero. Check the input values and 'pts' parameter.")
    
    return x_mean, pdf_mean

if __name__ == "__main__":
    vals = [0.05, 0.01, 0.05, 0.05, 0.0010]
    x, pdf = compute_combined_pdf(vals)
    import matplotlib.pyplot as plt

    plt.plot(x, pdf)
    plt.show()
