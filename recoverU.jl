
include("tomography.jl")

push!(LOAD_PATH, "/Users/guochu/Documents/QuantumSimulator/Meteor.jl/src")

using Zygote
using Zygote: @adjoint
using Optim
using Meteor.Diff
using JSON


function hermitian_matrix_util(n::Int, v::AbstractVector{<:Real})
	h = Zygote.Buffer(v, ComplexF64, n, n)
	_count = 1
	for i in 1:n
		for j in i+1:n
			tmp = complex(v[_count], v[_count+1])
			h[i, j] = tmp
			h[j, i] = tmp'
			_count += 2
		end
	end
	# println(_count)
	for i in 1:n
		h[i, i] = v[_count]
		_count += 1
	end
	# println(_count)
	(_count == n * n + 1) || error("something wrong.")
	# println(_count)
	# mh = copy(h)
	# return mh + mh'
	return copy(h)
end

hermitian_matrix(n, v) = hermitian_matrix_util(n, v)
@adjoint hermitian_matrix(n, v) = begin
	r, back = Zygote.pullback(hermitian_matrix_util, n, v) 
	return r, z -> begin
		out = back(z)
		return (nothing, real(out[2]))
	end
end
unitary_mat(n::Int, v::AbstractVector{<:Real}) = exp(im .* hermitian_matrix(n, v))

function update_left(hold::AbstractMatrix, Aj::AbstractArray{<:Number, 3}, Bj::AbstractArray{<:Number, 3})
	a1, a2, a3 = size(Aj)
	b1, b2, b3 = size(Bj)
	# println("**********************")
	# println(size(hold))
	# println(size(Aj))
	# println(size(Bj))
	# (size(hold, 1) == a1) || error("error1")
	# (size(hold, 2) == b1) || error("error2")
	tmp = hold * reshape(Bj, (b1, b2 * b3))
	# println(size(tmp))
	return reshape(Aj, (a1 * a2, a3))' * reshape(tmp, (size(tmp, 1)*b2, b3))
end

# function _overlap_util(hold, mps, Ua, Ub)
# 	for i in 1:length(mps)
# 		hold = update_left(hold, mps[i], Ua)
# 	end
# 	return tr(hold * Ub)	
# end

# function _overlap(mps, Ua, Ub)
# 	hold = ones(ComplexF64, 1, 1)
# 	return _overlap_util(hold, mps, Ua, Ub)
# end

# @adjoint _overlap(mps, Ua, Ub) = _overlap(mps, Ua, Ub), z -> begin
# 	a, b, c, d = Zygote.pullback(_overlap_util)
# end

function recover_hm_util(U::AbstractArray{ComplexF64, 4}, env_state::Vector{ComplexF64}, N::Int)
	d, D = size(U, 1), size(U, 2)
	d2 = d * d
	mps = ppt_tomography(U, env_state, N)

	println("norm of initial state $(norm(mps))")

	L1 = d * D * d * D
	L2 = D * D
	# v1 = randn(L1)
	# v2 = randn(L2)
	# U1 = ParametericUnitaryMatrix(v1)
	# U2 = ParametericUnitaryMatrix(v2)

	target_mps = mps[1:N]

	env_state_ansatz = reshape(QuantumSpins.onehot(ComplexF64, D, 0), 1, D)

	heff = ones(ComplexF64, 1, 1)
	scale = (1 / sqrt(d))

	# function _overlap(mps, Ua, Ub)
	# 	hold = heff
	# 	for i in 1:length(mps)
	# 		hold = update_left(hold, mps[i], Ua)
	# 	end
	# 	return tr(hold * Ub)
	# end

	loss(v) = begin
		Ua = scale .* unitary_mat(d*D, v[1:L1])
		Ub = unitary_mat(D, v[L1+1:L1+L2])

		Tmat = reshape(permutedims(reshape(Ua, (d, D, d, D)), (2,1,3,4)), (D, d2, D))
		L = length(target_mps)
		Tmat1 = reshape(env_state_ansatz * reshape(Tmat, (D, d2*D)), (1, d2, D))
		# r = _overlap(target_mps, Tmat1, Ub)
		hold = update_left(heff, target_mps[1], Tmat1)
		for i in 2:L
			hold = update_left(hold, target_mps[i], Tmat)
		end
		r = tr(hold * Ub)
		return abs(r - 1.)
	end

	# x0 = 0.05 .* randn(L1 + L2)

	# loss_1(v) = sum(abs.(hermitian_matrix(D, v)))
	# println(check_gradient(loss, x0, verbose=3))

	g!(storage, x) = begin
		grad = gradient(loss, x)
		storage .= grad[1]
	end

	x0 = 0.05 .* randn(L1 + L2)
    println("initial loss is $(loss(x0))")
    results = Optim.optimize(loss, g!, x0, BFGS(), Optim.Options(g_tol=1e-8, iterations=2000,
        store_trace=true, show_trace=false))

    x_opt, ls = Optim.minimizer(results), Optim.minimum(results)

    f_values = Optim.f_trace(results)

    println("Optim used $(Optim.iterations(results)) to converge.")
    println("final loss is $(ls).")

    return reshape(scale .* unitary_mat(d*D, x_opt[1:L1]), (d, D, d, D) ), ls
end

function recover_hm(U::AbstractArray{ComplexF64, 4}, env_state::Vector{ComplexF64}, N::Int, ntrials::Int=3)
	losses = Float64[]
	rs = []
	for i in 1:ntrials
		a, b = recover_hm_util(U, env_state, N)
		push!(rs, a)
		push!(losses, b)
	end

	pos = argmin(losses)
	return rs[pos]
end

function check_unitary_mat(n)
	mat = unitary_mat(n, randn(n * n))
	println( maximum(abs.(mat * mat' - I) ))
	println( maximum(abs.(mat' * mat - I) ))
end

function _fidelity(x, y)
	L = round(Int, sqrt(length(x))) 
	x = reshape(x, L, L)
	y = reshape(y, L, L)
	return real(tr(sqrt(x) * sqrt(y)))
end

function check_recover_hm()
	d = 2
	d2 = d^2
	D = 10
	U = random_unitary(d, D, 0.1)
	env_state = randn(ComplexF64, D)
	env_state ./= norm(env_state)

	Ntrain = 3
	U2 = recover_hm(U, env_state, Ntrain)
	env_state_2 = QuantumSpins.onehot(ComplexF64, D, 0)

	pt = ProcessTensor(U, env_state)
	pt2 = ProcessTensor(U2, env_state_2)

	println("check the recover precision.")
	kappa = 4
	LL = d2^3
	N = 50
	for i in 1:(N - kappa +1)
		dm_a = partial_dm(pt,i,i+kappa-1)
		dm_b = partial_dm(pt2,i,i+kappa-1)
		# println("trace a = $( tr(reshape(dm_a, LL, LL)) ), trace b = $( tr(reshape(dm_b, LL, LL)) )")
		# println("distance for the $i-th dm is $(QuantumSpins._distance(dm_a, dm_b))")
		println("distance for the $i-th dm is $(1-_fidelity(dm_a, dm_b))")
	end	

end

function recover_util(U, env_state, Ntrain)
	d, D = size(U, 1), size(U, 2)
	d2 = d^2
	# U = random_unitary(d, D, 0.1)
	# env_state = randn(ComplexF64, D)
	# env_state ./= norm(env_state)

	# Ntrain = 3
	U2 = recover_hm(U, env_state, Ntrain)
	env_state_2 = QuantumSpins.onehot(ComplexF64, D, 0)

	pt = ProcessTensor(U, env_state)
	pt2 = ProcessTensor(U2, env_state_2)

	println("check the recover precision.")
	kappa = 4
	LL = d2^3
	N = 50

	diffs = Float64[]
	for i in 1:(N - kappa +1)
		dm_a = partial_dm(pt,i,i+kappa-1)
		dm_b = partial_dm(pt2,i,i+kappa-1)
		# println("trace a = $( tr(reshape(dm_a, LL, LL)) ), trace b = $( tr(reshape(dm_b, LL, LL)) )")
		# println("distance for the $i-th dm is $(QuantumSpins._distance(dm_a, dm_b))")
		# println("distance for the $i-th dm is $(1-_fidelity(dm_a, dm_b))")
		push!(diffs, 1-_fidelity(dm_a, dm_b))
	end	
	return diffs
end

function main_recover()
	d = 2
	D = 10
	U = random_unitary(d, D, 0.1)
	env_state = randn(ComplexF64, D)
	env_state ./= norm(env_state)
	
	results = Dict{Int, Any}()
	for Ntrain in 2:4
		diffs = recover_util(U, env_state, Ntrain)
		results[Ntrain] = diffs
	end

	file_name = "fig3data/results.json"

	open(file_name, "w") do f
		write(f, JSON.json(results))
	end
end













