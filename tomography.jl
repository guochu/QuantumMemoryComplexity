# push!(LOAD_PATH, "/Users/guochu/Documents/QuantumSimulator/QuantumSpins")

using LinearAlgebra
using TensorOperations
using QuantumSpins

# function updateright(hold::AbstractMatrix, mpsAj::AbstractArray, mpsBj::AbstractArray) 
# 	@tensor m2[-1 -2;-3] := conj(mpsAj[-1, -2, 1]) * hold[1, -3]
# 	@tensor hnew[-1;-2] := m2[-1,1,2] * mpsBj[-2,1,2]
# 	return hnew
# end

# function updateleft(hold::AbstractMatrix, mpsAj::AbstractArray, mpsBj::AbstractArray) 
# 	@tensor m2[-1 -2; -3] := conj(mpsAj[1, -2, -3]) * hold[1, -1]
# 	@tensor hnew[-1; -2] := m2[1,2,-1] * mpsBj[1,2,-2]
# 	return hnew
# end

function left_qr(m::AbstractArray{<:Number, 3})
	s1, s2, s3 = size(m)
	m2 = reshape(m, s1*s2, s3)
	q, r = qr(m2)
	sizem = size(q, 2)
	return reshape(q, s1, s2, sizem), r
end

function random_unitary(d, D, η)
	m = d * D
	h = randn(ComplexF64, m, m)
	h =  h + h'
	h = η * h + I
	h = exp(im * h)
	return (1 / sqrt(d)) * reshape(h, d, D, d, D)
end

function trivial_unitary(d, D)
	h = randn(ComplexF64, d, d)
	h = exp(im * (h + h'))
	iden = one(zeros(ComplexF64, D, D))
	@tensor mh[1,3,2,4] := h[1, 2] * iden[3, 4]
	return (1 / sqrt(d)) * mh
end

struct ProcessTensor
	U::Array{ComplexF64, 4}
	T::Array{ComplexF64, 3}
	env_state::Vector{ComplexF64}

function ProcessTensor(U::AbstractArray{<:Number, 4}, env_state::Vector{ComplexF64})
	d, D = size(U, 1), size(U, 2)
	(length(env_state) == D) || throw(DimensionMismatch("size mismatch."))
	Tmat = reshape(permutedims(U, (2,1,3,4)), D, d*d, D)
	new(U, Tmat, env_state)
end

end

function ProcessTensor(U::AbstractArray{<:Number, 4})
	D = size(U, 2)
	env_state = randn(ComplexF64, D)
	return ProcessTensor(U, env_state ./ norm(env_state))
end

get_d(x::ProcessTensor) = size(x.U, 1)
get_D(x::ProcessTensor) = size(x.U, 2)

function Base.getindex(x::ProcessTensor, i::Int)
	(i < 1) && throw(BoundsError("error."))
	if i == 1
		@tensor tmp[2,3] := x.env_state[1] * x.T[1,2,3]
		return reshape(tmp, 1, size(tmp, 1), size(tmp, 2))
	else
		return x.T
	end
end


function partial_dm(x::ProcessTensor, pos_start::Int, pos_end::Int)
	tmp = x[1]
	for i in 1:pos_start-1
		q, r = left_qr(tmp)
		@tensor tmp[1,3,4] := r[1,2] * x[i+1][2,3,4]
	end
	tmp_c = conj(tmp)
	@tensor dm[2,4,3,5] := tmp[1,2,3] * tmp_c[1,4,5]
	for i in pos_start+1:pos_end-1
		xi = x[i]
		@tensor tmp[1,5,2,7,6,8] := dm[1,2,3,4] * xi[3,5,6] * conj(xi[4,7,8])
		dm = reshape(tmp, size(tmp,1)*size(tmp,2), size(tmp,3)*size(tmp,4), size(tmp, 5), size(tmp, 6))
	end
	xi = x[pos_end]
	@tensor tmp[1,5,2,7] := dm[1,2,3,4] * xi[3,5,6] * conj(xi[4,7,6])
	d2 = get_d(x)^2
	shapes = [d2 for i in pos_start:pos_end]
	# return reshape(tmp, size(tmp,1)*size(tmp,2), size(tmp,3)*size(tmp,4))
	return reshape(tmp, Tuple(repeat(shapes, 2)) )
end

function partial_dm(x::MPS, pos_start::Int, pos_end::Int)
	sm = Matrix(Diagonal(x.s[pos_start]))
	@tensor tmp[1,3,4] := sm[1, 2] * x[pos_start][2,3,4]
	@tensor dm[2,4,3,5] := tmp[1,2,3] * conj(tmp[1,4,5])

	for i in pos_start+1:pos_end-1
		xi = x[i]
		@tensor tmp[1,5,2,7,6,8] := dm[1,2,3,4] * xi[3,5,6] * conj(xi[4,7,8])
		dm = reshape(tmp, size(tmp,1)*size(tmp,2), size(tmp,3)*size(tmp,4), size(tmp, 5), size(tmp, 6))
	end
	xi = x[pos_end]
	@tensor tmp[1,5,2,7] := dm[1,2,3,4] * xi[3,5,6] * conj(xi[4,7,6])

	shapes = [size(x[i], 2) for i in pos_start:pos_end]
	return reshape(tmp, Tuple(repeat(shapes, 2)) )
end

function disentangler(dm::AbstractArray{<:Number, N}, D::Int) where N
	d2 = size(dm, 1)
	L = d2^(div(N, 2))
	dm2 = reshape(dm, L, L)
	eigvalues, eigvectors = eigen(Hermitian(dm2))
	threshold = 1.0e-12
	pos = findfirst(x -> x >= 1.0e-12, eigvalues) - 1
	(length(eigvalues) - pos > D) && error("results imprecise.")

	# nontrivial eigenvectors
	sizem = div(L, d2)

	m = zeros(eltype(eigvectors), d2, sizem, L)
	for j in 1:sizem
		for i in 1:d2
			# pos = (j-1) * d + i
			posb = (i-1) * sizem + j
			@. m[d2+1-i, j, :] = conj(eigvectors[:, posb])
		end
	end
	return reshape(m, size(dm))
end

function partial_dot(a::MPS, b::MPS)
	L = length(a)
	hold = ones(1, 1)
	for i in 1:L
		hold = QuantumSpins.updateleft(hold, a[i], b[i])
	end
	return hold
end

get_kappa(d2::Int, D::Int) = ceil(Int, log(d2, D)) + 1

function ppt_tomography(U::AbstractArray{ComplexF64, 4}, env_state::Vector{ComplexF64}, N::Int)
	d, D = size(U, 1), size(U, 2)
	pt = ProcessTensor(U, env_state)
	init_state = MPS([copy(pt[i]) for i in 1:N])
	canonicalize!(init_state)
	# println("initial state norm is $(norm(init_state))")
	# println("initial state entropy is $([entropy(init_state.s[i]) for i in 2:length(init_state)])")
	d2 = d * d
	kappa = get_kappa(d2, D) 
	# println("kappa is $kappa")
	circuit = []
	state = copy(init_state)
	for i in 1:(N-kappa+1)
		dm = partial_dm(state, 1, kappa)
		Uj = disentangler(dm, D)
		# println(state.s[2])
		gate = QuantumGate(collect(1:kappa), Uj)
		push!(circuit, shift(gate, i-1))
		apply!(gate, state)
		# println(state[1])
		# println(state.s[2])
		state = MPS(state[2:end], state.s[2:end])
	end
	(length(state) == kappa-1) || error("something wrong!")
	# the last block Uf
	dm = partial_dm(state, 1, kappa-1)
	dm2 = reshape(dm, d2^(kappa-1), d2^(kappa-1))

	# println("trace of reduce dm is $(tr(dm2))")

	eigvalues, eigvectors = eigen(Hermitian(dm2))
	threshold = 1.0e-12
	pos = findfirst(x -> x >= 1.0e-12, eigvalues) - 1
	(length(eigvalues) - pos > D) && error("results imprecise.")	
	state_f = zeros(eltype(eigvectors), d2^(kappa-1), D)
	LL = length(eigvalues)
	for i in 1:D
		m_pos = LL - D + i
		# println("eigenvalue is $(eigvalues[m_pos])")
		(eigvalues[m_pos] >= -1.0e-12) || error("eigenvalue is negative, something wrong.")
		vl = (eigvalues[m_pos] >= 0) ? eigvalues[m_pos] : 0.
		@. state_f[:, i] = sqrt(vl) .* conj(eigvectors[LL - D + i, :])
	end
	state_f = reshape(state_f, 1, size(state_f, 1), size(state_f, 2))

	# now reconstruct the state as MPS
	mps_tensors = Vector{Array{ComplexF64, 3}}(undef, N)
	for i in 1:N-kappa+1
		mps_tensors[i] = reshape(QuantumSpins.onehot(ComplexF64, d2, 0), 1, d2, 1)
	end
	for i in 1:kappa-2
		size_r = div(size(state_f, 2), d2)
		tmp = reshape(state_f, size(state_f, 1), d2, size_r, size(state_f, 3))
		q, r = tqr!(tmp, (1,2), (3,4))
		mps_tensors[N-kappa+1+i] = q
		state_f = r
	end
	mps_tensors[N] = state_f

	out_mps = MPS(mps_tensors)
	canonicalize!(out_mps)
	for gate in reverse(circuit)
		apply!(gate', out_mps)
	end

	# println("out mps norm is $(norm(out_mps))")
	canonicalize!(out_mps)
	# println(bond_dimensions(out_mps))

	# check if the result is correct
	
	# op = partial_dot(init_state, out_mps)
	# println(maximum(abs.(op * op' - I)))

	# for i in 1:(N-kappa+1)
	# 	dm_a = partial_dm(init_state,i,i+kappa-1)
	# 	dm_b = partial_dm(out_mps,i,i+kappa-1)
	# 	println("distance for the $i-th dm is $(QuantumSpins._distance(dm_a, dm_b))")
	# end
	
	
	return out_mps
end

function check_ppt_tomography(U::AbstractArray{ComplexF64, 4}, env_state::Vector{ComplexF64}, N::Int)
	d, D = size(U, 1), size(U, 2)
	pt = ProcessTensor(U, env_state)
	state = MPS([copy(pt[i]) for i in 1:N])
	canonicalize!(state)
	state_2 = ppt_tomography(U, env_state, N)

	kappa = get_kappa(d*d, D)
	for i in 1:(N - kappa +1)
		dm_a = partial_dm(state,i,i+kappa-1)
		dm_b = partial_dm(state_2,i,i+kappa-1)
		println("distance for the $i-th dm is $(QuantumSpins._distance(dm_a, dm_b))")
	end	
end

function check_ppt_tensor(m)
	D, d2 = size(m, 1), size(m, 2)
	d = round(Int, sqrt(d2))
	m2 = reshape(permutedims(reshape(m, D, d, d, D), (2,1,3,4)), d*D, d*D) * sqrt(d)
	return maximum(abs.(m2' * m2 - I)), maximum(abs.(m2 * m2' - I))
end


function main()
	d = 2
	D = 9
	U = random_unitary(d, D, 0.1)
	env_state = randn(ComplexF64, D)
	env_state ./= norm(env_state)

	check_ppt_tomography(U, env_state, 50)
end



