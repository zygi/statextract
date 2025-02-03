using Random
using Distributions



# class pvalue_distribution(rv_continuous):
#     def __init__(self, d, n, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.d = d  # Effect size
#         self.n = n  # Degrees of freedom
#         self.delta = d * np.sqrt(n)  # Non-centrality parameter

#     def _pdf(self, p):
#         """
#         Probability density function of the p-value distribution.
#         """
#         # Critical t-value under the null hypothesis for a two-sided test
#         t_c = stats.t.ppf(1 - p / 2, self.n)

#         # PDFs of the non-central t-distribution at t_c and -t_c
#         nct_pdf_pos = stats.nct.pdf(t_c, self.n, self.delta)
#         nct_pdf_neg = stats.nct.pdf(-t_c, self.n, self.delta)

#         # PDF of the central t-distribution at t_c
#         t_pdf = stats.t.pdf(t_c, self.n)

#         # Compute the derivative dt_c/dp
#         dt_c_dp = -1 / (2 * t_pdf)

#         # Compute the PDF of the p-value distribution
#         pdf = (nct_pdf_pos - nct_pdf_neg) * np.abs(dt_c_dp)
#         return pdf

#     def _cdf(self, p):
#         """
#         Cumulative distribution function of the p-value distribution.
#         """
#         # Critical t-value under the null hypothesis for a two-sided test
#         t_c = stats.t.ppf(1 - p / 2, self.n)

#         # CDFs of the non-central t-distribution
#         nct_cdf_pos = stats.nct.sf(t_c, self.n, self.delta)  # Survival function for T >= t_c
#         nct_cdf_neg = stats.nct.cdf(-t_c, self.n, self.delta)  # CDF for T <= -t_c

#         # Compute the CDF of the p-value distribution
#         cdf = nct_cdf_pos + nct_cdf_neg
#         return cdf

#     def _rvs(self, size=None, random_state=None):
#         """
#         Random variates of the p-value distribution.
#         """
#         # Generate random t-values from the non-central t-distribution
#         t_values = stats.nct.rvs(df=self.n, nc=self.delta, size=size, random_state=random_state)

#         # Compute the corresponding p-values
#         p_values = 2 * stats.t.sf(np.abs(t_values), df=self.n)
#         return p_values

# We want to implement this in Julia

struct PValueDistribution{T<:Real} <: ContinuousUnivariateDistribution
    d::T
    n::Int
    delta::T
end

function PValueDistribution(d::T, n::Int) where T<:Real
    return PValueDistribution(d, n, d * sqrt(n))
end


Base.rand(rng::AbstractRNG, d::PValueDistribution) = begin
    t_values = rand(rng, NoncentralT(d.n, d.d))
    return 2 * cdf(Normal(), abs(t_values))
end
Distributions.pdf(d::PValueDistribution, p::Real) = begin
    if !(0 <= p <= 1)
        return 0.0
    end
    
    # Critical t-value under the null hypothesis for a two-sided test
    t_c = quantile(TDist(d.n), 1 - p/2)
    
    # PDFs of the non-central t-distribution at t_c and -t_c
    nct_pdf_pos = pdf(NoncentralT(d.n, d.delta), t_c)
    nct_pdf_neg = pdf(NoncentralT(d.n, d.delta), -t_c)
    
    # PDF of the central t-distribution at t_c 
    t_pdf = pdf(TDist(d.n), t_c)
    
    # Compute the derivative dt_c/dp
    dt_c_dp = -1 / (2 * t_pdf)
    
    # Compute the PDF of the p-value distribution
    pdf_val = (nct_pdf_pos - nct_pdf_neg) * abs(dt_c_dp)
    
    return pdf_val
end

Distributions.cdf(d::PValueDistribution, p::Real) = begin
    t_c = quantile(TDist(d.n), 1 - p/2)
    nct_cdf_pos = ccdf(NoncentralT(d.n, d.delta), t_c)
    nct_cdf_neg = cdf(NoncentralT(d.n, d.delta), -t_c)
    return nct_cdf_pos + nct_cdf_neg
end

##

# function beta_inc_inv(a::T, b::T, y::T) where T<:Real
#     return quantile(Beta(a, b), y)
# end

using QuadGK


function test_pdf_from_cdf(dist)
    errors = []
    for p in 0.0:0.01:1.0
        # pdf_val = pdf(dist, p)
        cdf_val = cdf(dist, p)
        estimated_cdf = quadgk(x -> pdf(dist, x), 0, p; rtol=eps())[1]
        # println(p, " ", cdf_val, " ", estimated_cdf)
    end
end

quadgk(x -> pdf(PValueDistribution(0.001, 50), x), 0, 1.0; rtol=eps())
# typeof(0.001)