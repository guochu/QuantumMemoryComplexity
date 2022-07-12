include("tomography_entangled.jl")

using JSON


function main_fig3()
	# unitary tomography
	d = 2
	D = 5
	k = 51
	U = random_unitary(d, D, 0.1)


	init_state = randn(ComplexF64, d*D)
	init_state ./= norm(init_state)

	mps = ppt_tomography(U, init_state, k)

	bonds1 = bond_dimensions(mps)
	ents1 = [ _entropy((mps.s[i]).^2) for i in 2:length(mps)] 


	init_state = random_dm(d*D)
	pt = ProcessTensor(U, init_state)
	mps = ppt_tomography(pt, k)

	initial_entropy = _entropy(init_state)
	bonds2 = bond_dimensions(mps)
	ents2 = [ _entropy((mps.s[i]).^2) for i in 2:length(mps)] 

	results = Dict("pure_entropy"=>ents1, "mixed_entropy"=>ents2, "pure_Ds"=>bonds1, "mixed_Ds"=>bonds2, "mixed_initial_entropy"=>initial_entropy, "D"=>D)

	file_name = "fig3data/d$(d)D$(D)k$(k).json"

	open(file_name, "w") do f
		write(f, JSON.json(results))
	end
end


