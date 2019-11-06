VER = v"0.7.0"
if VERSION >= VER
    error("Julia version < "*string(VER)*" required!")
end

# Navier Stokes package

include("linear_solve.jl")

# Edges are either inlet, outlet, walls, or non-slip (object)
type EdgeClass
    inlet::BitMatrix
    outlet::BitMatrix
    walls::BitMatrix
    object::BitMatrix
end

# Given a bounding box, we can automatically assign edges as inlet, outlet, walls, or object
function EdgeClass(m::QuadMesh, bbox::Vector{Float64})
    qnum = size(m.t, 1);
    ec = EdgeClass(falses(qnum, 4), falses(qnum, 4), falses(qnum, 4), falses(qnum, 4))
    nodes = [1 2; 2 3; 3 4; 4 1];
    for i = 1:qnum
        for j = 1:4
            if m.bcs[i, j]
                ec.inlet[i, j] = all(m.p[m.t[i, nodes[j, :]], 1] .<= bbox[1]);
                ec.outlet[i, j] = all(m.p[m.t[i, nodes[j, :]], 1] .>= bbox[2]);
                ec.walls[i, j] = all(m.p[m.t[i, nodes[j, :]], 2] .<= bbox[3]) || all(m.p[m.t[i, nodes[j, :]], 2] .>= bbox[4]);
                ec.object[i, j] = !(ec.inlet[i, j] || ec.outlet[i, j] || ec.walls[i, j]);
            end
        end
    end
    ec
end

# Navier Stokes precomputed data
type nsdata
    m::QuadMesh
    dt::Float64
    damping_width::Float64
    uxpack::PartialMatrix
    uypack::PartialMatrix
    ppack::PartialMatrix
    ec::EdgeClass
end

# Precomputes the PDEs used in the Navier-Stokes equations
function navier_stokes_matrices( m::QuadMesh, n::Int64, dt::Float64 )
    bbox = zeros(4);
    bbox[1] = findmin(m.p[:, 1])[1]+1e-6;
    bbox[2] = findmax(m.p[:, 1])[1]-1e-6;
    bbox[3] = findmin(m.p[:, 2])[1]+1e-6;
    bbox[4] = findmax(m.p[:, 2])[1]-1e-6;
    ec = EdgeClass(m, bbox);
    damping_width = 1.0;
    # fconst = (x::Float64, y::Float64) -> -2.0/dt;
    # fconst1 = (x::Float64, y::Float64) -> 1.0
    # fzero = (x::Float64, y::Float64) -> 0.0;
    # fdamping = (x::Float64, y::Float64) -> damping_factor(x, damping_width, 5.0);
    # qnum = size(m.t, 1);
    # coeffs = Matrix{Matrix{Float64}}(qnum, 6);
    # coeffs[:, 1] = function_to_coeffs(m, n, fconst1);
    # coeffs[:, 2] = function_to_coeffs(m, n, fzero);
    # coeffs[:, 3] = function_to_coeffs(m, n, fdamping);
    # coeffs[:, 4] = function_to_coeffs(m, n, fzero);
    # coeffs[:, 5] = function_to_coeffs(m, n, fzero);
    # coeffs[:, 6] = function_to_coeffs(m, n, fconst);
    uxpack = fastorder2mesh_iter1( m, n, [1.0, 0.0, 1.0, 0.0, 0.0, -2.0/dt], ec.outlet|ec.walls)
    uypack = fastorder2mesh_iter1( m, n, [1.0, 0.0, 1.0, 0.0, 0.0, -2.0/dt], ec.outlet)
    ppack  = fastorder2mesh_iter1( m, n, [1.0, 0.0, 1.0, 0.0, 0.0, 0.0], ec.inlet|ec.walls|ec.object, ec.object, ec.outlet)
    return nsdata(m, dt, damping_width, uxpack, uypack, ppack, ec)
end

# Precomputed variables/matrices that can be passed as a single argument to functions (passed by reference)
type matrix_pack
    S1::SparseMatrixCSC{Float64, Int64}
    S0::SparseMatrixCSC{Float64, Int64}
    S::SparseMatrixCSC{Float64, Int64}
    M2::SparseMatrixCSC{Float64, Int64}
    M1::SparseMatrixCSC{Float64, Int64}
    iS0::Matrix{Float64}
    D2::SparseMatrixCSC{Float64, Int64}
    D1::SparseMatrixCSC{Float64, Int64}
    I::SparseMatrixCSC{Float64, Int64}
    d_dxrhs::Array{SparseMatrixCSC{Float64, Int64}, 1}
    d_dxlhs::Array{SparseMatrixCSC{Float64, Int64}, 1}
    d_dyrhs::Array{SparseMatrixCSC{Float64, Int64}, 1}
    d_dylhs::Array{SparseMatrixCSC{Float64, Int64}, 1}
    dd_dxrhs::Array{SparseMatrixCSC{Float64, Int64}, 1}
    dd_dxlhs::Array{SparseMatrixCSC{Float64, Int64}, 1}
    dd_dyrhs::Array{SparseMatrixCSC{Float64, Int64}, 1}
    dd_dylhs::Array{SparseMatrixCSC{Float64, Int64}, 1}
    quads::Vector{Matrix{Float64}}
    params::Vector{Matrix{Float64}}
    dc::Vector{Vector{Float64}}
    dvals::Vector{Matrix{Float64}}
    idvals::Vector{Matrix{Float64}}
    vand::Matrix{Float64}
    ivand::Matrix{Float64}
    plan::Vector{chplan}
    qnum::Int64
    n::Int64
    nn::Int64
end

# Actually creates matrix_pack
function matrix_pack(mesh::QuadMesh, n::Int64)
    qnum = size(mesh.t, 1);
    _, _, quads, params = cheb_mesh_grid(mesh, n)
    S1 = convertmat1(n);
    S0 = convertmat0(n);
    S = S1*S0;
    M2 = multmat2(n);
    M1 = multmat1(n);
    iS0 = invconvertmat0(n);
    D2 = diffmat2(n);
    D1 = diffmat1(n);
    I = speye(n);
    xr = Vector{SparseMatrixCSC{Float64, Int64}}(qnum);
    xl = Vector{SparseMatrixCSC{Float64, Int64}}(qnum);

    xxr = Vector{SparseMatrixCSC{Float64, Int64}}(qnum);
    xxl = Vector{SparseMatrixCSC{Float64, Int64}}(qnum);
    for i = 1:qnum
        xr[i] = (S1*D1).'*(params[i][3, 2]*I + params[i][4, 2]*M2).';
        xl[i] = (params[i][2, 2]*I + params[i][4, 2]*M2)*(S1*D1);

        xxr[i] = (D1).'*(params[i][3, 2]*I + params[i][4, 2]*M1).';
        xxl[i] = (params[i][2, 2]*I + params[i][4, 2]*M1)*(D1);
    end

    yr = Vector{SparseMatrixCSC{Float64, Int64}}(qnum);
    yl = Vector{SparseMatrixCSC{Float64, Int64}}(qnum);

    yyr = Vector{SparseMatrixCSC{Float64, Int64}}(qnum);
    yyl = Vector{SparseMatrixCSC{Float64, Int64}}(qnum);

    for i = 1:qnum
        yr[i] = (S1*D1).'*(params[i][3, 1]*I + params[i][4, 1]*M2).';
        yl[i] = (params[i][2, 1]*I + params[i][4, 1]*M2)*(S1*D1);

        yyr[i] = (D1).'*(params[i][3, 1]*I + params[i][4, 1]*M1).';
        yyl[i] = (params[i][2, 1]*I + params[i][4, 1]*M1)*(D1);
    end

    nn = nextpow2(div(3*n, 2))+1;
    xx, yy = meshgrid(chebpts(nn));
    vand = cheb_vander(nn);
    plan = Array{chplan}(Threads.nthreads());
    for i = 1:Threads.nthreads()
        plan[i] = cheb_plan(nn);
    end
    ivand = inv_cheb_vander(nn);
    dc = Vector{Vector{Float64}}(qnum)
    dvals = Vector{Matrix{Float64}}(qnum)
    idvals = Vector{Matrix{Float64}}(qnum)

    for i = 1:qnum
        dc[i] = det_coeffs(quads[i]);
        dvals[i] = dc[i][1]*xx + dc[i][2]*yy + dc[i][3];
        idvals[i] = vals2coeffs(1./dvals[i], plan[1])
    end
    return matrix_pack(S1, S0, S, M2, M1, iS0, D2, D1, I, xr, xl, yr, yl, xxr, xxl, yyr, yyl, quads, params, dc, dvals, idvals, vand, ivand, plan, qnum, n, nn)
end

# Creates constant boundary conditions
function const_bcs( m::QuadMesh, n::Int64, constant::Float64, nzbcs::BitMatrix)
    enum = size(m.e, 1);
    benum = enum - m.ienum;
    bcs = zeros(benum, n-1);
    boundaryPointer = 0;
    for i = 1:enum
        if m.e[i, 5] == 0
            boundaryPointer += 1;
            if nzbcs[m.e[i, 3], m.e[i, 4]]
                bcs[boundaryPointer, :] = constant;
            end
        end
    end
    return bcs;
end

# Creates boundary conditions that follow a certain function
function func_bcs( m::QuadMesh, n::Int64, func::Function )
    enum = size(m.e, 1)
    benum = enum - m.ienum
    bcs = zeros(benum, n-1);
    chpts = chebpts(n);
    boundaryPointer = 0;
    for i = 1:enum
        if m.e[i, 5] == 0
            boundaryPointer += 1;
            coords = m.p[m.t[m.e[i, 3], mod(m.e[i, 4]-1: m.e[i, 4], 4)], :];
            midpoint = [coords[1, 1]+coords[2, 1], coords[1, 2]+coords[2, 2]];
            difference = [coords[2, 1]-coords[1, 1], coords[2, 2]-coords[1, 2]];
            for j = 1:n-1
                pt = midpoint + chpts[j].*difference
                bcs[boundaryPointer, j] = func[pt[1], pt[2]];
            end
        end
    end
    bcs;
end

# function pressure_bcs( m::QuadMesh, n::Int64, ux::Vector{Matrix{Float64}}, uy::Vector{Matrix{Float64}}, nzbcs::BitMatrix, mp::matrix_pack)
#     enum = size(m.e, 1);
#     benum = enum - m.ienum;
#     bcs = zeros(benum, n-1);
#     boundaryPointer = 0;
#     vand = cheb_vander(n);
#     for i = 1:enum
#         if m.e[i, 5] == 0
#             boundaryPointer += 1;
#             ii = m.e[i, 3];
#             jj = m.e[i, 4];
#             if nzbcs[ii, jj]
#                 temp = dd_dx(uy[ii], mp, ii) - dd_dy(ux[ii], mp, ii);
#                 a = m.p[m.t[ii, mod(jj, 4)+1], :] - m.p[m.t[ii, jj], :];
#                 a /= sqrt(a[1]^2+a[2]^2);
#                 a = [a[2], -a[1]];
#                 temp = a[1]*dd_dy(temp, mp, ii) - a[2]*dd_dx(temp, mp, ii);
#                 temp = vand*temp*vand.';
#                 if jj == 1
#                     bcs[boundaryPointer, :] = temp[end, end:-1:2];
#                 elseif jj == 2
#                     bcs[boundaryPointer, :] = temp[end:-1:2, 1];
#                 elseif jj == 3
#                     bcs[boundaryPointer, :] = temp[1, 1:end-1];
#                 elseif jj == 4
#                     bcs[boundaryPointer, :] = temp[1:end-1, end];
#                 end
#             end
#         end
#     end
#     return bcs;
# end

# x derivative, returned multiplied by det(r, s)^3 and in ultraspherical coefficients of parameter 2
function d_dx(u::Matrix{Float64}, mp::matrix_pack, ii::Int64)
    d = mp.S*u*mp.d_dxrhs[ii];
    d -= mp.d_dxlhs[ii]*u*mp.S.';
    d = dets2(d, mp, ii);
    d
end

function d_dy(u::Matrix{Float64}, mp::matrix_pack, ii::Int64)
    d = -mp.S*u*mp.d_dyrhs[ii];
    d += mp.d_dylhs[ii]*u*mp.S.';
    d = dets2(d, mp, ii);
    d
end

# x derivative, returned in Chebyshev coefficients
function dd_dx(u::Matrix{Float64}, mp::matrix_pack, ii::Int64)
    d = (u*mp.dd_dxrhs[ii]) / mp.S0.';
    d -= mp.S0 \ (mp.dd_dxlhs[ii]*u);
    d = invdets(d, mp, ii);
    d
end

function dd_dy(u::Matrix{Float64}, mp::matrix_pack, ii::Int64)
    d = -(u*mp.dd_dyrhs[ii]) / mp.S0.';
    d += mp.S0 \ (mp.dd_dylhs[ii]*u);
    d = invdets(d, mp, ii);
    d
end

# product of Chebyshev coefficients and ultraspherical coefficients of parameter 2
function product(cheb::Matrix{Float64}, ultra::Matrix{Float64}, mp::matrix_pack)
    p = coeffs_multiply(cheb, mp.S \ ultra / mp.S.', mp.plan[Threads.threadid()]);
    p = p[1:mp.n, 1:mp.n];
    p = mp.S*p*mp.S.';
    p
end

# multiplies by the determinant of the jacobian of the transformation on quadrilateral i
function dets(u::Matrix{Float64}, mp::matrix_pack, i::Int64)
    return u*(mp.dc[i][1]*mp.M2.' + 0.5*mp.dc[i][3]*mp.I) + (mp.dc[i][2]*mp.M2 + 0.5*mp.dc[i][3]*mp.I)*u;
end

# Computes the lagrangian of u
function lagrangian(u::Matrix{Float64}, pack::PartialMatrix, i::Int64)
    n = pack.pack[i].n;
    buf = zeros(n, n);
    buf[1:n-2, 1:n-2] = pack.pre[i] \ (pack.pack[i].aa * u[:]);
    return buf;
end

# function weighted_y_lagrangian(u::Matrix{Float64}, mp::matrix_pack, i::Int64, coeffs::Matrix{Float64})
#     return dd_dx(dd_dx(u, mp, i), mp, i) + coeffs_multiply(coeffs, dd_dy(dd_dy(u, mp, i), mp, i), mp.plan[Threads.threadid()])[1:mp.n, 1:mp.n];
# end

# det(r, s)^2 and det(r, s)^3
dets2(u::Matrix{Float64}, mp::matrix_pack, ii::Int64) = dets(dets(u, mp, ii), mp, ii)
dets3(u::Matrix{Float64}, mp::matrix_pack, ii::Int64) = dets(dets(dets(u, mp, ii), mp, ii), mp, ii)

# det(r, s)^-1
function invdets(u::Matrix{Float64}, mp::matrix_pack, ii::Int64)
    return coeffs_multiply(u, mp.idvals[ii], mp.plan[Threads.threadid()])[1:mp.n, 1:mp.n]
end

# Starts a Navier-Stokes simulation from scratch
function navier_stokes_solve( dat::nsdata, steps::Int64, speed::Float64)
    laststate = Vector{Vector{Matrix{Float64}}}(4);
    n = dat.uxpack.n;
    qnum = size(dat.m.t, 1);
    for i = 1:4
        laststate[i] = Vector{Matrix{Float64}}(qnum)
        for j = 1:qnum
        laststate[i][j] = zeros(n, n);
        end
    end
    return navier_stokes_solve(dat, steps, speed, laststate);
end

# function damping_factor(x::Float64, width::Float64 = 0.1, finish::Float64 = 5.0, endval::Float64 = 2.0)
#     if x <= finish - width
#         return 1.0;
#     elseif x >= finish
#         return endval;
#     else
#         x = (finish - x)/width
#         return endval + (1-endval)*(10.0*x^3 - 15.0*x^4 + 6.0*x^5);
#     end
# end
# 
# function damping_factor(x::Array{Float64}, width::Float64 = 0.1, finish::Float64 = 5.0, endval::Float64 = 2.0)
#     y = Array{Float64}(size(x))
#     for i = 1:length(x)
#         y[i] = damping_factor(x[i], width, finish, endval);
#     end
#     return y
# end

function navier_stokes_solve( dat::nsdata, steps::Int64, speed::Float64, laststate::Vector{Vector{Matrix{Float64}}} )
#     bounding_box = [xmin, xmax, ymin, ymax]
    n = dat.uxpack.n;
    m = dat.m;
    qnum = size(m.t, 1);
    fff = falses(qnum, 4);
    dt = dat.dt;
    damping_width = dat.damping_width;

    # [ Upack.pack, Upack.pre, Upack.aig, Upack.agi, Upack.inv, quads ] = fastorder2mesh_iter1( vertex_list, coords_list, n, [1 0 1 0 0 -1./dt], outlet_edges );
    # [ Vpack.pack, Vpack.pre, Vpack.aig, Vpack.agi, Vpack.inv] = fastorder2mesh_iter1( vertex_list, coords_list, n, [1 0 1 0 0 -1./dt], outlet_edges|walls );
    #
    # Poisson solve for pressure
    # [ Ppack.pack, Ppack.pre, Ppack.aig, Ppack.agi, Ppack.inv] = fastorder2mesh_iter1( vertex_list, coords_list, n, [1 0 1 0 0 0], inlet_edges|walls|no_slip );
    println("Generating MPack...")
    MPack = matrix_pack(m, n);


    p = Vector{Matrix{Float64}}(qnum);

    for i = 1:qnum
        p[i] = zeros(n, n)
    end

    # damping_region = falses(qnum);
    # damping_coeffs = copy(p);

    x, y = cheb_mesh_grid(m, n);
    vand = cheb_vander(n);
    ivand = inv_cheb_vander(n);

    # for i = 1:qnum
    #   if any(m.p[m.t[i, :], 1] .> 5.0 - damping_width)
    #     damping_region[i] = true;
    #   end
    #   damping_coeffs[i] = damping_factor(x[i], damping_width, 5.0);
    #   damping_coeffs[i] = ivand*damping_coeffs[i]*ivand.'
    # end

#     Set up blank matrices
    nonlinear_curr_x = copy(p);
    nonlinear_last_x = copy(p);

    nonlinear_curr_y = copy(p);
    nonlinear_last_y = copy(p);

    rhsx = copy(p);
    rhsy = copy(p);
    rhs = copy(p);
    ux = copy(p);
    uy = copy(p);
    w = copy(p);

    ux = copy(laststate[1]);
    uy = copy(laststate[2]);
    p = copy(laststate[3]);
    w = copy(laststate[4]);

    res = 10;
    U = Matrix{Vector{Matrix{Float64}}}(div(steps, res), 4);
    len = 0;

    println("Beginning Loop...")
    # repeat this every step
    for k = 1:steps
        tic();
        # Screened Poisson:  lap(u*) - u*/dt = (u_n * del)u_n - u_n/dt
        # Dirichlet bcs (0 for no slip)
        # Poisson solve:  dt*lap(p) = div(u*), dp/dn = 0 for no-slip
        # u_n+1 = u* - dt*grad(p)

        # velocity is split into ux and uy, pressure is p
        # derivatives are ux_x, ux_y, uy_x, uy_y, p_x, p_y
        # (u * del)u = [ux*ux_x + uy*ux_y; ux*uy_x + uy*uy_y]
        # div(u) = ux_x + uy_y
        # grad(p) = [p_x, p_y]


        # Screened Poisson Solve
        # println("Creating rhsx and rhsy")
        Threads.@threads for i = 1:qnum
        # if damping_region[i]
        #   rhsx[i] = 3*nonlinear_curr_x[i] - nonlinear_last_x[i] - 2*dets3(MPack.S*ux[i]*MPack.S.', MPack, i)./dt - weighted_lagrangian(ux[i], MPack, i, damping_coeffs[i])
        #   rhsy[i] = 3*nonlinear_curr_y[i] - nonlinear_last_y[i] - 2*dets3(MPack.S*uy[i]*MPack.S.', MPack, i)./dt - weighted_lagrangian(uy[i], MPack, i, damping_coeffs[i])
        # else
            rhsx[i] = (3*nonlinear_curr_x[i] - nonlinear_last_x[i]) - lagrangian(ux[i], dat.ppack, i) - 2*dets3(MPack.S*ux[i]*MPack.S.', MPack, i)./dt;
            rhsy[i] = (3*nonlinear_curr_y[i] - nonlinear_last_y[i]) - lagrangian(uy[i], dat.ppack, i) - 2*dets3(MPack.S*uy[i]*MPack.S.', MPack, i)./dt;
        # end
        end

        val = 30;
        if k > val
            sp = 1;
            spd = 0;
        else
            sp = 3*(k./val).^2 - 2*(k./val).^3;
            spd = 1./val.*(6.*(k./val) - 6.*(k./val).^2);
        end

        bcsxtemp = const_bcs(m, n, speed.*sp, dat.ec.inlet);
        bcsytemp = const_bcs(m, n, 0.0, m.bcs);
        # println("Linear solves uxtemp and uytemp")
        uxtemp = fastorder2solve(dat.uxpack, bcsxtemp, rhsx);
        uytemp = fastorder2solve(dat.uypack, bcsytemp, rhsy);

        # println("Creating rhs")
        Threads.@threads for i = 1:qnum
            rhs[i] = ((d_dx(uxtemp[i], MPack, i) + d_dy(uytemp[i], MPack, i))./dt);
        end

        pbcs = const_bcs(m, n, spd*speed/dt, dat.ec.inlet);
        # pbcs -= pressure_bcs(m, n, uxtemp, uytemp, dat.ec.object, MPack);

        #    Pressure Solve
        #    NEED TO ADD NEUMANN OUTSIDE BCS AND ZERO PRESSURE ON ONE SIDE
        # println("Linear solve p")
        p = fastorder2solve(dat.ppack, pbcs, rhs);
        # println("Creating ux, uy, and w")
        Threads.@threads for i = 1:qnum
            ux[i] = (uxtemp[i] - dt.*dd_dx(p[i], MPack, i));
            uy[i] = (uytemp[i] - dt.*dd_dy(p[i], MPack, i));
            w[i] = (dd_dx(uy[i], MPack, i) - dd_dy(ux[i], MPack, i));

            nonlinear_last_x[i] = nonlinear_curr_x[i];
            nonlinear_last_y[i] = nonlinear_curr_y[i];

            nonlinear_curr_x[i] = product(ux[i], d_dx(ux[i], MPack, i), MPack) + product(uy[i], d_dy(ux[i], MPack, i), MPack);
            nonlinear_curr_y[i] = product(ux[i], d_dx(uy[i], MPack, i), MPack) + product(uy[i], d_dy(uy[i], MPack, i), MPack);
        end
        if mod(k, res) == 0
            if ~all(isfinite(ux[1]))
                return U
            end
            ind = div(k, res);
            U[ind, 1] = copy(ux);
            U[ind, 2] = copy(uy);
            U[ind, 3] = copy(p);
            U[ind, 4] = copy(w);
        end

        str = string("Step ", k, " / ", steps, " (", div(100*k, steps), "%), Time = ", toq());
        print("\b"^len, str);
        len = length(str);
    end
    U
end
