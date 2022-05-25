# -------------------------------------------
# FEniCS code  Variational Fracture Mechanics
################################################################################
#                                                                              #
# A Taylor-Hood finite element method for gradient damage models of            #
# fracture in incompressible hyperelastic materials                            #
# author: Bin Li                                                               #
# Email: bl736@cornell.edu                                                     #
# date: 10/01/2018                                                             #
#                                                                              #
################################################################################
# e.g. python3 traction-neo-Hookean.py --meshsize 100						   #
################################################################################

# ----------------------------------------------------------------------------
from __future__ import division
from dolfin import *
from mshr import *
from scipy import optimize
from ufl import eq

import argparse
import math
import os
import shutil
import sympy
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters for DOLFIN and SOLVER
# ----------------------------------------------------------------------------
set_log_level(LogLevel.WARNING)  # 20, // information of general interest
# set some dolfin specific parameters
# ----------------------------------------------------------------------------
parameters["form_compiler"]["representation"]="uflacs"
parameters["form_compiler"]["optimize"]=True
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["quadrature_degree"]=2
info(parameters,True)

# Parameters of the nonlinear SNES solver used for the displacement u-problem
solver_u_parameters  = {"nonlinear_solver": "snes",
                         "symmetric": True,
                         "snes_solver": {"linear_solver": "mumps",
                                         "method" : "newtontr",
                                         "line_search": "cp",
                                         "preconditioner" : "hypre_amg",
                                         "maximum_iterations": 200,
                                         "absolute_tolerance": 1e-10,
                                         "relative_tolerance": 1e-10,
                                         "solution_tolerance": 1e-10,
                                         "report": True,
                                         "error_on_nonconvergence": False}}

# parameters of the PETSc/Tao solver used for the alpha-problem
tao_solver_parameters = {"maximum_iterations": 100,
                         "report": False,
                         "line_search": "more-thuente",
                         "linear_solver": "cg",
                         "preconditioner" : "hypre_amg",
                         "method": "tron",
                         "gradient_absolute_tol": 1e-8,
                         "gradient_relative_tol": 1e-8,
                         "error_on_nonconvergence": True}

# Set up the solvers
solver_alpha  = PETScTAOSolver()
solver_alpha.parameters.update(tao_solver_parameters)
# info(solver_alpha.parameters,True) # uncomment to see available parameters

# Implement the box constraints for damage field
# --------------------------------------------------------------------
# Variational problem for the damage (non-linear to use variational inequality solvers of petsc)
# Define the minimisation problem by using OptimisationProblem class
class DamageProblem(OptimisationProblem):
    def __init__(self):
        OptimisationProblem.__init__(self)
        self.total_energy = damage_functional
        self.Dalpha_total_energy = E_alpha
        self.J_alpha = E_alpha_alpha
        self.alpha = alpha
        self.bc_alpha = bc_alpha
    def f(self, x):
        self.alpha.vector()[:] = x
        return assemble(self.total_energy)
    def F(self, b, x):
        self.alpha.vector()[:] = x
        assemble(self.Dalpha_total_energy, b)
        for bc in self.bc_alpha:
            bc.apply(b)
    def J(self, A, x):
        self.alpha.vector()[:] = x
        assemble(self.J_alpha, A)
        for bc in self.bc_alpha:
            bc.apply(A)

# Element-wise projection using LocalSolver
def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

# Define boundary sets for boundary conditions
# ----------------------------------------------------------------------------
class bot_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], -H/2, hsize) #and between(x[0], (0.0, 2.5))

class top_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H/2, hsize)

class pin_point(SubDomain):
    def inside(self, x, on_boundary):
        hsize2 = 0.02
        if near(x[0], 0.0, hsize2) and near(x[1], -H/2, hsize2) and near(x[2], T, 0.03):
            return True
        # if near(x[0], 0.0, 2*hsize) and near(x[1], H/2, 2*hsize) and near(x[2], T/2, hsize):
        #     return True

# Convert all boundary classes for visualization
bot_boundary = bot_boundary()
top_boundary = top_boundary()
pin_point = pin_point()

# User Parameters
# ----------------------------------------------------------------------------
parameters.parse()
userpar = Parameters("user")
userpar.add("mu", 1)          # Shear modulus
userpar.add("kappa", 1000)       # Bulk Modulus
userpar.add("Gc", 1)            # fracture toughness
userpar.add("k_ell", 5.e-5)     # residual stiffness
userpar.add("load_min", 0.0)
userpar.add("load_max", 1.3)
userpar.add("load_steps", 260)
userpar.add("hsize", 0.01)
userpar.add("ell_multi", 5)
userpar.add("exp_load", 0)
userpar.add("a_exp", 1.064)
userpar.add("b_exp", 0.00076979)
userpar.add("c_exp", -1.064)
userpar.add("d_exp", -0.1654)
# Parse command-line options
userpar.parse()

# Geometry parameters
W, H, T = 1.0, 1.5, 0.08
hsize = userpar["hsize"]
# Zero body force
body_force = Constant((0., 0., 0.))

# Material constants
mu    = userpar["mu"]
kappa = userpar["kappa"]
Gc    = userpar["Gc"]           # Fracture toughness
k_ell = userpar["k_ell"]        # Residual stiffness
# Damage regularization parameter - internal length scale used for tuning Gc
ell_multi = userpar["ell_multi"]
ell = Constant(ell_multi*hsize)

# Number of steps
load_min = userpar["load_min"]
load_max = userpar["load_max"]
load_steps = userpar["load_steps"]

# Exponential function loading
exp_load = userpar["exp_load"]
a_exp = userpar["a_exp"]
b_exp = userpar["b_exp"]
c_exp = userpar["c_exp"]
d_exp = userpar["d_exp"]

# Numerical parameters of the alternate minimization
maxiteration = 2500
AM_tolerance = 1e-4

modelname = "3D-TaylorHood"
meshname  = modelname + "-mesh.xdmf"
simulation_params = "R1_%.1f_R2_%.0f_h_%.3f_S_%.0f_dt_%.1f" \
                    % (Gc/mu, kappa/mu, hsize, load_steps, load_max)
savedir   = "output/" + modelname + "/" + simulation_params + "/"

if MPI.rank(MPI.comm_world) == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

# Mesh generation
# mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(L, H, W), Nx, Ny, Nz)
mesh = Mesh("../Geo/3DEdgeCrack.xml")
geo_mesh  = XDMFFile(MPI.comm_world, savedir+meshname)
geo_mesh.write(mesh)
# obtain number of space dimensions
mesh.init()
ndim = mesh.geometry().dim()
if MPI.rank(MPI.comm_world) == 0:
    print ("the dimension of mesh: {0:2d}".format(ndim))

if MPI.rank(MPI.comm_world) == 0:
  print("The kappa/mu: {0:4e}".format(kappa/mu))
  print("The mu/Gc: {0:4e}".format(mu/Gc))

# Define lines and points
lines = MeshFunction("size_t", mesh, mesh.topology().dim() - 2)
points = MeshFunction("size_t", mesh, mesh.topology().dim() - 3)

# show lines of interest
lines.set_all(0)
bot_boundary.mark(lines, 1)
top_boundary.mark(lines, 1)
file_results = XDMFFile(savedir + "/" + "lines.xdmf")
file_results.write(lines)

# Show points of interest
points.set_all(0)
pin_point.mark(points, 1)
file_results = XDMFFile(savedir + "/" + "points.xdmf")
file_results.write(points)

# Variational formulation
# ----------------------------------------------------------------------------
# Tensor space for projection of stress
T_DG0 = TensorFunctionSpace(mesh,'DG',0)
DG0 = FunctionSpace(mesh,'DG',0)
# Create Taylor-Hood function space for elasticity + Damage
V_CG2 = VectorFunctionSpace(mesh, "Lagrange", 2)
# CG1 also defines the function space for damage
CG1 = FunctionSpace(mesh, "Lagrange", 1)
P2elem = V_CG2.ufl_element()
P1elem = CG1.ufl_element()
TH = MixedElement([P2elem,P1elem])
# Define function spaces for displacement and pressure in V_u
V  = FunctionSpace(mesh, TH)

# Define the function, test and trial fields for elasticity problem
w_p    = Function(V)
u_p    = TrialFunction(V)
v_q    = TestFunction(V)
(u, p) = split(w_p)
(v, q) = split(v_q)         # Test functions for (u,p)
# define the function, test, and trial for damage problem
alpha  = Function(CG1)
dalpha = TrialFunction(CG1)
beta   = TestFunction(CG1)

# Define functions to save
PTensor = Function(T_DG0, name="Nominal Stress")
FTensor = Function(T_DG0, name="Deformation Gradient")
JScalar = Function(CG1, name="Volume Ratio")

# Dirichlet boundary condition
# --------------------------------------------------------------------
u00 = Expression("0.0", degree=0)
u0 = Expression(["0.0","0.0","0.0"], degree=0)
u1 = Expression("t",  t=0.0, degree=0)
u2 = Expression("-t", t=0.0, degree=0)
# Displacement on top and bottom boundaries
bc_u0 = DirichletBC(V.sub(0).sub(2), u00, pin_point, method='pointwise')
# Displacement fixed in x and y on bottom
bc_u1 = DirichletBC(V.sub(0).sub(0), u00, bot_boundary)
bc_u2 = DirichletBC(V.sub(0).sub(1), u2, bot_boundary)
bc_u3 = DirichletBC(V.sub(0).sub(0), u00, top_boundary)
bc_u4 = DirichletBC(V.sub(0).sub(1), u1, top_boundary)
bc_u = [bc_u0, bc_u1, bc_u2, bc_u3, bc_u4]

# bc - alpha (zero damage)
bc_alpha0 = DirichletBC(CG1, 0.0, top_boundary)
bc_alpha1 = DirichletBC(CG1, 0.0, bot_boundary)
bc_alpha = [bc_alpha0, bc_alpha1]

# Define the energy functional of damage problem
# --------------------------------------------------------------------
# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
I2 = ((tr(C))**2-tr(C*C))/2.    # Invariant 2 of the right CG
J  = det(F)                     # Invariant 3 of the deformation gradient, F

# Define some parameters for the eigenvalues
d_par = tr(C)/3.
e_par = sqrt(tr((C-d_par*I)*(C-d_par*I))/6.)
# conditional(condition, true_value, false_value) where eq means ==
f_par = conditional(eq(tr((C-d_par*I)*(C-d_par*I)),0.), 0.*I, (1./e_par)*(C-d_par*I))

# IMPORTANT: Bound the argument of 'acos' both from above and from below
g_par0 = det(f_par)/2.
# ge(a,b) is a >= b and le(a,b) is a <= b
g_par1 = conditional(ge(g_par0,  1.-DOLFIN_EPS),  1.-DOLFIN_EPS, g_par0)
g_par  = conditional(le(g_par0, -1.+DOLFIN_EPS), -1.+DOLFIN_EPS, g_par1)
h_par = acos(g_par)/3.

# Define the eigenvalues of C (principal stretches) where lmbda1s <= lmbda2s<= lmbda3s
lmbda1s = d_par + 2.*e_par*cos(h_par + 2.*pi/3.)
lmbda2s = d_par + 2.*e_par*cos(h_par + 4.*pi/3.)
lmbda3s = d_par + 2.*e_par*cos(h_par + 6.*pi/3.)

# Define the energy functional of elastic problem
# --------------------------------------------------------------------
def w(alpha):           # Specific energy dissipation per unit volume
    return alpha

def a(alpha):           # Modulation function
    return (1.0-alpha)**2

def b_sq(alpha):        # b(alpha) = (1-alpha)^6 therefore we define b squared
    return (1.0-alpha)**3

def P(u, alpha):        # Nominal stress tensor
    return a(alpha)*mu*(F - inv(F.T)) - b_sq(alpha)*p*J*inv(F.T)

def Heaviside(x):       # Heaviside step function: H(x) = 1, x>= 0; H(x) = 0, x< 0;
    return conditional(ge(x, 0.), 1., 0.)

# Elastic energy, additional terms enforce material incompressibility and regularizes the Lagrange Multiplier
elastic_energy  = (a(alpha)+k_ell)*(mu/2.0)*(Ic-3.0-2.*ln(J))*dx \
                - b_sq(alpha)*p*(J-1)*dx - 1./(2.*kappa)*p**2*dx
elastic_energy2 = (a(alpha)+k_ell)*(mu/2.0)*(lmbda1s+lmbda2s+lmbda3s-3.-2.*ln(lmbda1s*lmbda2s*lmbda3s))*dx \
                - b_sq(alpha)*p*(J-1)*dx - 1./(2.*kappa)*p**2*dx

external_work  = dot(body_force, u)*dx
elastic_potential = elastic_energy - external_work

# Elastic energy decomposition into active
W_act = (a(alpha)+k_ell)*Heaviside(sqrt(lmbda1s)-1.)*(mu/2.)*(lmbda1s-1.-2*ln(sqrt(lmbda1s)))*dx \
      + (a(alpha)+k_ell)*Heaviside(sqrt(lmbda2s)-1.)*(mu/2.)*(lmbda2s-1.-2*ln(sqrt(lmbda2s)))*dx \
      + (a(alpha)+k_ell)*Heaviside(sqrt(lmbda3s)-1.)*(mu/2.)*(lmbda3s-1.-2*ln(sqrt(lmbda3s)))*dx \
      - Heaviside(J-1.)*(b_sq(alpha)*p*(J-1.) + p**2/(2.*kappa))*dx

# Elastic energy decomposition into passive
W_pas = (mu/2.)*Heaviside(1.-sqrt(lmbda1s))*(lmbda1s-1.-2*ln(sqrt(lmbda1s)))*dx \
      + (mu/2.)*Heaviside(1.-sqrt(lmbda2s))*(lmbda2s-1.-2*ln(sqrt(lmbda2s)))*dx \
      + (mu/2.)*Heaviside(1.-sqrt(lmbda3s))*(lmbda3s-1.-2*ln(sqrt(lmbda3s)))*dx \
      - Heaviside(1.-J)*(p*(J-1.) + p**2/(2.*kappa))*dx

# Compute directional derivative about w_p in the direction of v (Gradient)
F_u = derivative(elastic_potential, w_p, v_q)
# Compute directional derivative about alpha in the direction of dalpha (Hessian)
J_u = derivative(F_u, w_p, u_p)

# Variational problem for the displacement
problem_u = NonlinearVariationalProblem(F_u, w_p, bc_u, J=J_u)
# Set up the solvers
solver_u  = NonlinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
# info(solver_u.parameters, True)

# Define the energy functional of damage problem
# --------------------------------------------------------------------
alpha_0 = interpolate(Expression("0.", degree=0), CG1)  # initial (known) alpha
z = sympy.Symbol("z", positive=True)
c_w = float(4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
# The elastic potential now consists of W_act + W_pas
damage_functional = W_act + W_pas + dissipated_energy

# Compute directional derivative about alpha in the direction of beta (Gradient)
E_alpha       = derivative(damage_functional, alpha, beta)
# Compute directional derivative about alpha in the direction of dalpha (Hessian)
E_alpha_alpha = derivative(E_alpha, alpha, dalpha)

# Set the lower and upper bound of the damage variable (0-1)
# alpha_lb = interpolate(Expression("x[0]>=0 & x[0]<=L/2 & near(x[1], H/2, 0.1 * hsize) ? 1.0 : 0.0", \
#                        hsize = hsize, L=L, H=H, degree=0), CG1)
alpha_lb = interpolate(Expression("0.", degree=0), CG1)
alpha_ub = interpolate(Expression("1.", degree=0), CG1)  # upper bound, set to 1

# Loading vector modeled after an exponential function
if exp_load == 1:
    fcn_load = np.linspace(load_min, load_steps, load_steps)
    load_multipliers = []
    for steps in fcn_load:
        exp1 = a_exp*exp(b_exp*steps) + c_exp*exp(d_exp*steps)
        load_multipliers.append(exp1)
else :
    load_multipliers = np.linspace(load_min, load_max, load_steps)

# initialization of vectors to store data of interest
energies   = np.zeros((len(load_multipliers), 5))
iterations = np.zeros((len(load_multipliers), 2))

# Data file name
file_tot = XDMFFile(MPI.comm_world, savedir + "/results.xdmf")
# Saves the file in case of interruption
file_tot.parameters["rewrite_function_mesh"] = False
file_tot.parameters["functions_share_mesh"]  = True
file_tot.parameters["flush_output"]          = True
# write the parameters to file
File(savedir+"/parameters.xml") << userpar

timer0 = time.process_time()        # Timer start

# Solving at each timestep
# ----------------------------------------------------------------------------
for (i_t, t) in enumerate(load_multipliers):
    if MPI.rank(MPI.comm_world) == 0:
        print("\033[1;32m--- Starting of Time step {0:2d}: t = {1:4f} ---\033[1;m".format(i_t, t))

    # Alternate Mininimization scheme
    # -------------------------------------------------------------------------
    # Solve for u holding alpha constant then solve for alpha holding u constant
    iteration = 1           # Initialization of iteration loop
    err_alpha = 1.0         # Initialization for condition for iteration

    # Conditions for iterations
    while err_alpha > AM_tolerance and iteration < maxiteration:
        # solve elastic problem
        solver_u.solve()
        # solve damage problem with box constraint
        solver_alpha.solve(DamageProblem(), alpha.vector(), alpha_lb.vector(), alpha_ub.vector())
        # test error
        alpha_error = alpha.vector() - alpha_0.vector()
        err_alpha = alpha_error.norm('linf')
        # monitor the results
        if MPI.rank(MPI.comm_world) == 0:
          print ("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))
        # update iteration
        alpha_0.assign(alpha)
        iteration = iteration + 1
    # updating the lower bound to account for the irreversibility
    alpha_lb.vector()[:] = alpha.vector()

    # Split into displacement and pressure
    (u, p)   = w_p.split()

    # Project nominal stress to tensor function space
    local_project(P(u, alpha), T_DG0, PTensor)
    local_project(F, T_DG0, FTensor)
    local_project(J, CG1, JScalar)

    # Rename for paraview
    alpha.rename("Damage", "alpha")
    u.rename("Displacement", "u")
    p.rename("Pressure", "p")

    # Write solution to file
    file_tot.write(alpha, t)
    file_tot.write(u, t)
    file_tot.write(p, t)
    file_tot.write(PTensor,t)
    file_tot.write(FTensor,t)
    file_tot.write(JScalar,t)

    # Update the displacement with each iteration
    u1.t = t
    u2.t = t

    # Some post-processing
    # ----------------------------------------
    # Save number of iterations for the time step
    iterations[i_t] = np.array([t, iteration])

    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    volume_ratio         = assemble(J/(T*H*W)*dx)

    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value+\
    	                      surface_energy_value, volume_ratio])

    if MPI.rank(MPI.comm_world) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{1:6f}]".format(elastic_energy_value, surface_energy_value))
        # print("\nElastic and Surface Energies: [{},{}]".format(elastic_energy_value, surface_energy_value))
        print("\nVolume Ratio: [{}]".format(volume_ratio))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir + '/Taylor-Hood-energies.txt', energies)
        np.savetxt(savedir + '/Taylor-Hood-iterations.txt', iterations)

print("elapsed CPU time: ", (time.process_time() - timer0))

# Plot energy and stresses
if MPI.rank(MPI.comm_world) == 0:
    p1, = plt.plot(energies[slice(None), 0], energies[slice(None), 1])
    p2, = plt.plot(energies[slice(None), 0], energies[slice(None), 2])
    p3, = plt.plot(energies[slice(None), 0], energies[slice(None), 3])
    plt.legend([p1, p2, p3], ["Elastic", "Dissipated", "Total"], loc="best", frameon=False)
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    plt.title('Taylor-Hood FEM')
    plt.savefig(savedir + '/Taylor-Hood-energies.pdf', transparent=True)
    plt.close()

    p4, = plt.plot(energies[slice(None), 0], energies[slice(None), 4])
    plt.xlabel('Displacement')
    plt.ylabel('Volume ratio')
    plt.title('Taylor-Hood FEM')
    plt.savefig(savedir + '/Taylor-Hood-volume-ratio.pdf', transparent=True)
    plt.close()
