################################################################################
# FEniCS Code
################################################################################
#
# A Taylor-Hood mixed finite element method for gradient damage models of
# fracture in incompressible hyperelastic materials for cases of plane stress
#
# Authors: Ida Ang and Bin Li
# Email: ia267@cornell.edu (primary contact) and bin.l@gtiit.edu.cn
# Date Last Edited: May 25th 2022
################################################################################

from dolfin import *
from mshr import *
from ufl import rank
# Import python script
import src.LogLoading as LogLoad
import math
import os
import shutil
import sympy
import time
import numpy as np
import matplotlib.pyplot as plt

# Parameters for DOLFIN and SOLVER
# ----------------------------------------------------------------------------
set_log_level(LogLevel.WARNING)

# Set some dolfin specific parameters
parameters["form_compiler"]["representation"]="uflacs"
parameters["form_compiler"]["optimize"]=True
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["quadrature_degree"]=2
info(parameters,True)

# Parameters of the solvers for displacement and damage (alpha-problem)
# -----------------------------------------------------------------------------
# Parameters of the nonlinear SNES solver used for the displacement u-problem
solver_u_parameters = {"nonlinear_solver": "snes",
                       "symmetric": True,
                       "snes_solver": {"linear_solver": "mumps",
                                       "method" : "newtontr",
                                       "line_search": "cp",
                                       "preconditioner" : "hypre_amg",
                                       "maximum_iterations": 100,
                                       "absolute_tolerance": 1e-10,
                                       "relative_tolerance": 1e-10,
                                       "solution_tolerance": 1e-10,
                                       "report": True,
                                       "error_on_nonconvergence": False}}


# Parameters of the PETSc/Tao solver used for the alpha-problem
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

# Initial condition (IC) class
class InitialConditions(UserExpression):
    def eval(self, values, x):
        # Displacement u0 = (values[0], values[1])
        values[0] = 0.0             # Displacement in x direction
        values[1] = 0.0             # Displacement in y direction
        values[2] = 0.0             # Pressure
        values[3] = 1.0             # Deformation gradient component: F_{33}
    def value_shape(self):
         return (4,)

# Define boundary sets for boundary conditions
# ----------------------------------------------------------------------------
class bot_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], -H/2, hsize)

class top_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H/2, hsize)

class pin_point(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L/2, hsize) and near(x[1], 0.0, hsize)

# Convert all boundary classes for visualization
bot_boundary = bot_boundary()
top_boundary = top_boundary()
pin_point = pin_point()

# Setting user parameters which can be defined through command line
parameters.parse()
userpar = Parameters("user")
userpar.add("mu",1)             # Shear modulus
userpar.add("kappa",1000)       # Bulk modulus
userpar.add("Gc",1)             # Fracture toughness
userpar.add("k_ell",5.e-5)      # Residual stiffness
userpar.add("load_max", 0.65)   # Maximum loading (fracture can occur earlier)
userpar.add("load_steps", 65)   # Steps in which loading from 0 to load_max occurs
userpar.add("hsize", 0.02)      # Element size in the center of the domain
userpar.add("ell_multi", 5)     # For definition of phase-field width
# 1 = on, 0 = off for loading defined by a log function
userpar.add("log_load", 0)
userpar.parse()

# Constants: some parsed from user parameters
# ----------------------------------------------------------------------------
# Geometry paramaters
L, H = 6.0, 1.0              # Length (x) and height (y-direction)
hsize = userpar["hsize"]     # Geometry based definition for regularization
# Zero body force
body_force = Constant((0., 0.))

# Material parameters
mu    = userpar["mu"]           # Shear Modulus
kappa = userpar["kappa"]        # Bulk Modulus
Gc    = userpar["Gc"]           # Fracture toughness
k_ell = userpar["k_ell"]        # Residual stiffness
# Damage regularization parameter - internal length scale used for tuning Gc
ell_multi = userpar["ell_multi"]
ell = Constant(ell_multi*hsize)

# Number of steps
load_min = 0.0
load_max = userpar["load_max"]
load_steps = userpar["load_steps"]

# Numerical parameters of the alternate minimization
maxiteration = 2500
AM_tolerance = 1e-4

if MPI.rank(MPI.comm_world) == 0:
  print("The kappa/mu: {0:4e}".format(kappa/mu))
  print("The mu/Gc: {0:4e}".format(mu/Gc))

# Naming parameters for saving output
modelname = "PlaneStressTaylorHood"
meshname  = modelname + "-mesh.xdmf"
simulation_params = "ShearTest_R1_%.1f_R2_%.0f_S_%.0f_dt_%.2f" % (Gc/mu, kappa/mu, load_steps, load_max)
savedir   = "output/" + modelname + "/" + simulation_params + "/"

# For parallel processing - write one directory
if MPI.rank(MPI.comm_world) == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

# Mesh generation of structured mesh using inbuilt functions
# mesh = RectangleMesh(Point(-L/2, -H/2), Point(L/2, H/2), Nx, Ny)
# Mesh generation using Gmsh for finer mesh resolution in certain sections
mesh = Mesh("../Gmsh/2DShearTest3Ref.xml")
# Obtain number of space dimensions
mesh.init()
ndim = mesh.geometry().dim()
# Structure used for one printout of the statement
if MPI.rank(MPI.comm_world) == 0:
    print ("the dimension of mesh: {0:2d}".format(ndim))
# Save mesh
geo_mesh  = XDMFFile(MPI.comm_world, savedir+meshname)
geo_mesh.write(mesh)

# Define lines and points this is for visualization to make sure boundary
# conditions are applied correctly
lines = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
points = MeshFunction("size_t", mesh, mesh.topology().dim() - 2)

# Show lines of interest
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

# Loading and initialization of vectors to store data of interest
log_load = userpar["log_load"]
if log_load == 1:   # If logistic loading is specified
    # Any of these parameters can be changed to modify the log function
    inc = 0.001
    int_point = 2
    growth_rate = 0.3
    load_multipliers = LogLoad.LogLoading(savedir, load_max, load_steps, inc, int_point, growth_rate)
else :
    load_multipliers = np.linspace(load_min, load_max, load_steps)

# Initialization of vectors to store data of interest
energies   = np.zeros((len(load_multipliers), 5))
iterations = np.zeros((len(load_multipliers), 2))

# Variational formulation
# -----------------------------------------------------------------------------
# The plane stress formulation requires the explicit definition of the F_{33}
# component of the deformation gradient

# Tensor space for projection of quantities of interest
T_DG0 = TensorFunctionSpace(mesh,'DG',0)
DG0   = FunctionSpace(mesh,'DG',0)
# Create mixed function space for elasticity
V_CG2 = VectorFunctionSpace(mesh, "Lagrange", 2)
# CG1 also defines the function space for damage
CG1 = FunctionSpace(mesh, "Lagrange", 1)
V_CG2elem = V_CG2.ufl_element()
CG1elem = CG1.ufl_element()
# Stabilized mixed method for incompressible elasticity
# Define in order that you unpack them in (displacement, pressure, and F_{33})
MixElem = MixedElement([V_CG2elem, CG1elem, CG1elem])
# Define function space for displacement, pressure, and F_{33} (u, p, F33) in V_u
V = FunctionSpace(mesh, MixElem)

# Define the function, test and trial fields for elastic sub-problem
w_p = Function(V)
u_p = TrialFunction(V)
v_q = TestFunction(V)
(u, p, F33) = split(w_p)     # Functions for (u, p, F33)
(v, q, v_F33) = split(v_q)   # Test functions for (u, p, F33)
# Define the function, test and trial fields for damage problem
alpha  = Function(CG1)
dalpha = TrialFunction(CG1)
beta   = TestFunction(CG1)

# Define functions to save
PTensor = Function(T_DG0, name="Nominal Stress")
FTensor = Function(T_DG0, name="Deformation Gradient")
JScalar = Function(CG1, name="Volume Ratio")

# Initial Conditions (IC)
#------------------------------------------------------------------------------
# Initial conditions are created by using the class defined and then
# interpolating into a finite element space
init = InitialConditions(degree=1)          # Expression requires degree def.
w_p.interpolate(init)                       # Interpolate current solution

# Dirichlet boundary condition
# -----------------------------------------------------------------------------
u00 = Constant((0.0))
u0 = Expression(["0.0", "0.0"], degree=0)
u1 = Expression(["0.0", "t"], t=0.0, degree=0)
u2 = Expression(["0.0", "-t"], t=0.0, degree=0)

# Pin Point
bc_u0 = DirichletBC(V.sub(0), u0, pin_point, method='pointwise')
# Top/bottom boundaries have displacement in the y direction
bc_u1 = DirichletBC(V.sub(0), u1, top_boundary)
bc_u2 = DirichletBC(V.sub(0), u2, bot_boundary)
bc_u = [bc_u1, bc_u2]

# No damage to the boundaries - damage does not initiate from constrained edges
bc_alpha0 = DirichletBC(CG1, 0.0, bot_boundary)
bc_alpha1 = DirichletBC(CG1, 0.0, top_boundary)
bc_alpha = [bc_alpha0, bc_alpha1]

# Define the energy functional of damage problem
# -----------------------------------------------------------------------------
# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Plane stress invariants
Ic = tr(C) + F33**2
J = det(F)*F33

# Define the energy functional of the elasticity problem
# ----------------------------------------------------------------------------
def w(alpha):           # Specific energy dissipation per unit volume
    return alpha

def a(alpha):           # Modulation function
    return (1.0-alpha)**2

def b_sq(alpha):        # b(alpha) = (1-alpha)^6 therefore we define b squared
    return (1.0-alpha)**3

def P(u, alpha):        # Nominal stress tensor
    return a(alpha)*mu*(F - inv(F.T)) - b_sq(alpha)*p*J*inv(F.T)

# Elastic energy, additional terms enforce material incompressibility and
# regularizes the Lagrange Multiplier
elastic_energy    = (a(alpha)+k_ell)*(mu/2.0)*(Ic-3.0-2.0*ln(J))*dx \
                  - b_sq(alpha)*p*(J-1.0)*dx - 1./(2.*kappa)*p**2*dx
external_work     = dot(body_force, u)*dx
elastic_potential = elastic_energy - external_work

# Compute directional derivative about w_p in the direction of v_q (Gradient)
# The additional terms in the second line are related plane stress
F_u = derivative(elastic_potential, w_p, v_q) \
    + (a(alpha)*mu*(F33 - 1/F33) - b_sq(alpha)*p*J/F33)*v_F33*dx
# Compute directional derivative about w_p in the direction of u_p
J_u = derivative(F_u, w_p, u_p)

# Variational problem for the displacement
problem_u = NonlinearVariationalProblem(F_u, w_p, bc_u, J=J_u)
# Set up the solvers
solver_u  = NonlinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
# info(solver_u.parameters, True)

# Define the energy functional of damage problem
# --------------------------------------------------------------------
# Initial (known) damage is an undamaged state
alpha_0 = interpolate(Expression("0.", degree=0), CG1)
# Define the specific energy dissipation per unit volume
z = sympy.Symbol("z", positive=True)
c_w = float(4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))
# Define the phase-field fracture term of the damage functional
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
damage_functional = elastic_potential + dissipated_energy

# Compute directional derivative about alpha in the direction of beta (Gradient)
E_alpha       = derivative(damage_functional, alpha, beta)
# Compute directional derivative about alpha in the direction of dalpha (Hessian)
E_alpha_alpha = derivative(E_alpha, alpha, dalpha)

# Lower and upper bound, set to 0 and 1 respectively
# Uncomment the below for a completely undamaged state
# alpha_lb = interpolate(Expression("0.", degree=0), CG1)
# Damage from the left hand side to the center point
alpha_lb = interpolate(Expression("x[0]>=-L/2 & x[0]<=0.0 & near(x[1], 0.0, 0.1 * hsize) ? 1.0 : 0.0", \
                       hsize = hsize, L=L, degree=0), CG1)
alpha_ub = interpolate(Expression("1.", degree=0), CG1)

# Split solutions
(u, p, F33) = w_p.split()

# Data file name
file_tot = XDMFFile(MPI.comm_world, savedir + "/results.xdmf")
# Saves the file in case of interruption
file_tot.parameters["rewrite_function_mesh"] = False
file_tot.parameters["functions_share_mesh"]  = True
file_tot.parameters["flush_output"]          = True
# Write the parameters to file
File(savedir+"/parameters.xml") << userpar

# Timing
timer0 = time.process_time()

# Solving at each timestep
# ----------------------------------------------------------------------------
for (i_t, t) in enumerate(load_multipliers):
    # Structure used for one printout of the statement
    if MPI.rank(MPI.comm_world) == 0:
        print("\033[1;32m--- Start of time step {0:2d}: t = {1:4f} ---\033[1;m".format(i_t, t))

    # Alternate Mininimization Scheme
    # -------------------------------------------------------------------------
    # Solve for u holding alpha constant then solve for alpha holding u constant
    iteration = 1           # Initialization of iteration loop
    err_alpha = 1.0         # Initialization for condition for iteration

    # Conditions for iteration
    while err_alpha > AM_tolerance and iteration < maxiteration:
        # Solve elastic problem
        solver_u.solve()
        # Solve damage problem with lower and upper bound constraint
        solver_alpha.solve(DamageProblem(), alpha.vector(), alpha_lb.vector(), alpha_ub.vector())
        # Test error
        alpha_error = alpha.vector() - alpha_0.vector()
        err_alpha = alpha_error.norm('linf')
        # Printouts to monitor the results and number of iterations
        if MPI.rank(MPI.comm_world) == 0:
            print ("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))
        # Update variable for next iteration
        alpha_0.assign(alpha)
        iteration = iteration + 1
    # Updating the lower bound to account for the irreversibility of damage
    alpha_lb.vector()[:] = alpha.vector()

    # Project to the correct function space
    local_project(P(u, alpha), T_DG0, PTensor)
    local_project(F, T_DG0, FTensor)
    local_project(J, CG1, JScalar)

    # Rename for visualization in paraview
    alpha.rename("Damage", "alpha")
    u.rename("Displacement", "u")
    p.rename("Pressure", "p")
    F33.rename("F33", "F33")

    # Write solution to file
    file_tot.write(alpha, t)
    file_tot.write(u, t)
    file_tot.write(p, t)
    file_tot.write(F33, t)
    file_tot.write(PTensor, t)
    file_tot.write(FTensor, t)
    file_tot.write(JScalar,t)

    # Update the displacement with each iteration
    u1.t = t
    u2.t = t

    # Post-processing
    # ----------------------------------------
    # Save number of iterations for the time step
    iterations[i_t] = np.array([t, iteration])

    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    volume_ratio = assemble(J/(L*H)*dx)
    # Save time, energies, and J = detF to data array
    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value+\
    	                      surface_energy_value, volume_ratio])

    if MPI.rank(MPI.comm_world) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{1:6f}]".format(elastic_energy_value, surface_energy_value))
        print("\nElastic and Surface Energies: [{},{}]".format(elastic_energy_value, surface_energy_value))
        print("\nVolume Ratio: [{}]".format(volume_ratio))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir + '/Taylor-Hood-energies.txt', energies)
        np.savetxt(savedir + '/Taylor-Hood-iterations.txt', iterations)

print("elapsed CPU time: ", (time.process_time() - timer0))

# Plot energy and stresses
# ----------------------------------------------------------------------------
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
