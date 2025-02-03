# using Random, Distributions

# function gen_groups_with_cohen_d(rng::AbstractRNG, d, n)
#     # generate a dataset of two groups, each of size n, whose difference has cohen d 'd'
#     # Generate control group from standard normal
#     control = rand(rng, Normal(0, 1), n)
    
#     # Generate treatment group with mean shifted by d*σ to achieve desired Cohen's d
#     treatment = rand(rng, Normal(d, 1), n)

#     actual_d = (mean(treatment) - mean(control)) / std(control)
#     return control, treatment, actual_d
# end


# function gen_2way_anova_data(rng::AbstractRNG, n)
#     # generate a 1x2 design dataset where 


using Polyhedra
using GLPK; solver = GLPK.Optimizer

using LinearAlgebra
using SparseArrays


# ENV["PYCALL_JL_RUNTIME_PYTHON"] = "/Users/zygi/python/statextract/.venv/bin/python"
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = readchomp(`realpath -s ./.venv/bin/python`)
ENV["VIRTUAL_ENV"] = readchomp(`realpath -s ./.venv`)

using PythonCall


# @pya `import sys; ans=sys.executable`
# pyimport("sys").executable

torch = pyimport("torch")
torch.random.manual_seed(0)

transformers = pyimport("transformers")
model = transformers.Phi3ForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-128k-instruct",  
    device_map="cpu",  
    torch_dtype="auto",  
    trust_remote_code=true,  
) 


##

py_wt = pyconvert(Matrix, model.lm_head.weight.type(torch.float32).detach().numpy())


# py"""
# import torch 
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Phi3ForCausalLM

# torch.random.manual_seed(0) 
# model = Phi3ForCausalLM.from_pretrained( 
#     "microsoft/Phi-3-mini-128k-instruct",  
#     device_map="cpu",  
#     torch_dtype="auto",  
#     trust_remote_code=True,  
# ) 


# def mm():
#     return model
# """

# torch = pyimport("torch")

# model = py"mm()"

# py_wt = model.lm_head.weight.type(torch.float32).detach().numpy()


# # dump py_wt using serialization
# using Serialization
# Serialization.write(py_wt, "test_wt.bin")



# # read py_wt
# py_wt = load("phi3_mini_weights.jld2")



##

N_DIM, M_DIM = size(py_wt) 

A = [-1.0*ones(Float32, N_DIM-1, 1) sparse(I(N_DIM-1))]
b = spzeros(N_DIM-1)

A_p = A * py_wt

ph2 = polyhedron(hrep(A_p, zeros(size(A_p, 1))), DefaultLibrary{Float64}(solver)); 

# hss = halfspaces(ph2)


##

# new_hrep = detecthlinearity(hrep(ph2), solver);

# fulldim(new_hrep)
# fulldim(hrep(ph2))

# isredundant(hrep(ph2), first(hss))

A_red = removehredundancy(hrep(ph2), solver)




# nhalfspaces(ph2)

# ph = polyhedron(hrep(A, b), DefaultLibrary{Float32}(solver))

# ph2 = dense(ph)# / (py_wt')

# vr = vrep(ph)


# randmap = randn(32, N_DIM)'


# A_p = A * randmap

# ph2 = polyhedron(hrep(sparse(A_p), spzeros(size(A_p, 1))), DefaultLibrary{Float64}(solver))



##

# A_red2 = ph / sparse(randmap')


A_red = removehredundancy(hrep(ph2), solver)

# hrep(A_red2)

# all(halfspaces(A_red2) .== halfspaces(ph2))
# ph2

# hrep(ph2)

# nrays(vr)