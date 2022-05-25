# FEniCS code  Variational Fracture Mechanics
################################################################################
#
# A stabilized mixed finite element method for gradient damage models of
# fracture in incompressible hyperelastic materials
# Author: Ida Ang and Bin Li                                                              #
# Email: ia267@cornell.edu                                                     #
# Modifed for plane stress cases
#
################################################################################
# e.g. python3 traction-stabilized.py --meshsize 100						   #
################################################################################

# ----------------------------------------------------------------------------
from __future__ import division
from dolfin import *
from mshr import *
from scipy import optimize
from ufl import rank

import argparse
import math
from math import hypot
import os
import shutil
import sympy
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters for DOLFIN and SOLVER
# ----------------------------------------------------------------------------
set_log_level(LogLevel.WARNING)  # 20 Information of general interest

# Set some dolfin specific parameters
# ----------------------------------------------------------------------------
parameters["form_compiler"]["representation"]="uflacs"
parameters["form_compiler"]["optimize"]=True
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["quadrature_degree"]=2
info(parameters,True)

# Parameters of the solvers for displacement and damage (alpha-problem)
# -----------------------------------------------------------------------------
# Parameters of the nonlinear SNES solver used for the displacement u-problem
solver_up_parameters  = {"nonlinear_solver": "snes",
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
                                         "error_on_nonconvergence": True}}

# Parameters of the PETSc/Tao solver used for the alpha-problem
tao_solver_parameters = {"maximum_iterations": 100,
                         "line_search": "more-thuente",
                         "linear_solver": "cg",
                         "preconditioner" : "hypre_amg",
                         "method": "tron",
                         "gradient_absolute_tol": 1e-8,
                         "gradient_relative_tol": 1e-8,
                         "report": False,
                         "error_on_nonconvergence": True}

# Variational problem for the damage problem (non-linear - use variational inequality solvers in PETSc)
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

# Initial condition (IC) class, necessary for 2D formulations
class InitialConditions(UserExpression):
    def eval(self, values, x):
        # Displacement u0 = (values[0], values[1])
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0        # Pressure
        values[3] = 1.0        # F_33
    def value_shape(self):
         return (4,)

# Define boundary sets for boundary conditions
# ----------------------------------------------------------------------------
class bot_boundary(SubDomain):
    def inside(self, x, on_boundary):
        if on_boundary and x[1] < -H/4:
            if near(x[0], -L/2, 0.01*hsize) and between(x[1], (-H/2 + hsize, H/2 - hsize)):
                return False
            elif near(x[0], L/2, 0.01*hsize) and between(x[1], (-H/2 + hsize, H/2 - hsize)):
                return False
            else:
                return True

class top_boundary(SubDomain):
    def inside(self, x, on_boundary):
        if on_boundary and x[1] > H/4:
            if near(x[0], -L/2, 0.01*hsize) and between(x[1], (-H/2 + hsize, H/2 - hsize)):
                return False
            elif near(x[0], L/2, 0.01*hsize) and between(x[1], (-H/2 + hsize, H/2 - hsize)):
                return False
            else:
                return True

class pin_point(SubDomain):
    def inside(self, x, on_boundary):
        if near(x[0], 0.0, 0.389+DOLFIN_EPS) and near(x[1], pin, 0.389+DOLFIN_EPS):
            return True

# Convert all boundary classes for visualization
bot_boundary = bot_boundary()
top_boundary = top_boundary()
pin_point = pin_point()

# Set the user parameters
parameters.parse()
userpar = Parameters("user")
userpar.add("mu", 1)            # Shear modulus
userpar.add("kappa", 1000)       # Bulk modulus
userpar.add("Gc", 1)            # Fracture toughness (2.4E3)
userpar.add("k_ell", 5.e-5)     # Residual stiffness
userpar.add("load_min", 0.0)
userpar.add("load_max", 1.42)
userpar.add("load_steps", 200)
userpar.add("hsize", 0.01)
userpar.add("ell_multi", 5)
exp_load = 0
userpar.add("a_exp", 1.071)
userpar.add("b_exp", 0.00141)
userpar.add("c_exp", -1.071)
userpar.add("d_exp", -0.1969)
# Parse command-line options
userpar.parse()

# Constants: some parsed from user parameters
# ----------------------------------------------------------------------------
# Geometry parameters
L, H = 8.0, 9.5           # Length (x) and height (y-direction)
r_d = 5
y_c = (r_d*r_d - L/2*L/2)**(1/2) + H/2
center = Point(0, y_c)
pin = H/2 - y_c

hsize = userpar["hsize"]

# Material model parameters
mu    = userpar["mu"]           # Shear modulus
kappa = userpar["kappa"]        # Bulk Modulus
Gc    = userpar["Gc"]           # Fracture toughness
k_ell = userpar["k_ell"]        # Residual stiffness
# Damage regularization parameter - internal length scale used for tuning Gc
ell_multi = userpar["ell_multi"]
ell = Constant(ell_multi*hsize)

# Exponential function loading
a_exp = userpar["a_exp"]
b_exp = userpar["b_exp"]
c_exp = userpar["c_exp"]
d_exp = userpar["d_exp"]

# Number of steps
load_min = userpar["load_min"]
load_max = userpar["load_max"]
load_steps = userpar["load_steps"]

# Numerical parameters of the alternate minimization scheme
maxiteration = 2500         # Sets a limit on number of iterations
AM_tolerance = 1e-4

# Naming parameters for saving output
modelname = "PlaneStressStabilized"
meshname  = modelname + "-mesh.xdmf"
simulation_params = "Trap_R1_%.1f_R2_%.0f_h_%.2f_S_%.0f_dt_%.1f" \
                    % (Gc/mu, kappa/mu, hsize, load_steps, load_max)
savedir   = "output/" + modelname + "/" + simulation_params + "/"

# For parallel processing - write one directory
if MPI.rank(MPI.comm_world) == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

# Mesh generation of structured mesh
# mesh = RectangleMesh(Point(-L/2, -H/2), Point(L/2, H/2), Nx, Ny)
# Mesh generation of structured and refined mesh
mesh = Mesh("../Geo/2DTrapezoidal.xml")
# Obtain number of space dimensions
mesh.init()
ndim = mesh.geometry().dim()
# Structure used for one printout of the statement
if MPI.rank(MPI.comm_world) == 0:
    print ("Mesh Dimension: {0:2d}".format(ndim))
# Mesh printout
geo_mesh = XDMFFile(MPI.comm_world, savedir + meshname)
geo_mesh.write(mesh)

# Characteristic element length - used for stabilization
h = CellDiameter(mesh)

#-----------------------------------------------------------------------------
p0 = -(3.0*8.0/3.0*mu*5.0*hsize+Gc)/(8.0/3.0*mu*5.0*hsize)
q0 = 2.0
tc = 2.*sqrt(-p0/3.0)*cos(1./3.*acos(3.0*q0/2.0/p0*sqrt(-3.0/p0)))-1.0

if MPI.rank(MPI.comm_world) == 0:
  print("The critical loading: [{}]".format(tc))
  print("The kappa/mu: {0:4e}".format(kappa/mu))
  print("The mu/Gc: {0:4e}".format(mu/Gc))

lines = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
points = MeshFunction("size_t", mesh, mesh.topology().dim() - 2)

# show lines of interest
lines.set_all(0)
bot_boundary.mark(lines, 1)
top_boundary.mark(lines, 2)
file_results = XDMFFile(savedir + "/" + "lines.xdmf")
file_results.write(lines)

# Show points of interest
points.set_all(0)
pin_point.mark(points, 1)
file_results = XDMFFile(savedir + "/" + "points.xdmf")
file_results.write(points)

# Constitutive functions of the damage model
def w(alpha):
    return alpha

def a(alpha):
    return (1.0-alpha)**2

def b(alpha):
    return (1.0-alpha)**6

def b_sq(alpha):
    return (1.0-alpha)**3

# Variational formulation
# ----------------------------------------------------------------------------
# Tensor space for projection of stress
TT = TensorFunctionSpace(mesh,'DG',0)
# Create equal order function space for elasticity + damage
P2 = VectorFunctionSpace(mesh, "Lagrange", 2)
P1 = FunctionSpace(mesh, "Lagrange", 1)
P2elem = P2.ufl_element()
P1elem = P1.ufl_element()
# Stabilized mixed FEM for incompressible elasticity
TH  = MixedElement([P2elem,P1elem,P1elem])
# Define function spaces for displacement, pressure, and F_{33} in V_u
V_u = FunctionSpace(mesh, TH)
# Define function space for damage in V_alpha
V_alpha = FunctionSpace(mesh, "Lagrange", 1)

# Define the function, test and trial fields for elasticity problem
w_p    = Function(V_u)
u_p    = TrialFunction(V_u)
v_q    = TestFunction(V_u)
(u, p, F33) = split(w_p)     # Displacement, pressure, (u, p, F_{33})
(v, q, v_F33) = split(v_q)   # Test functions for u, p and F33
# Define the function, test and trial fields for damage problem
alpha  = Function(V_alpha, name = "Damage")
dalpha = TrialFunction(V_alpha)
beta   = TestFunction(V_alpha)

# Dirichlet boundary condition
# --------------------------------------------------------------------
u00 = Constant((0.0))
u0 = Expression(["0.0", "0.0"], degree=0)
u1 = Expression("t", t= 0.0, degree=0)
u2 = Expression("-t", t= 0.0, degree=0)

# Pin Point
bc_u0 = DirichletBC(V_u.sub(0).sub(0), u00, pin_point)
# Top/bottom boundaries have displacement in the y direction
bc_u2 = DirichletBC(V_u.sub(0).sub(1), u1, top_boundary)
bc_u4 = DirichletBC(V_u.sub(0).sub(1), u2, bot_boundary)
# Combine
bc_u = [bc_u0, bc_u2, bc_u4]

# No damage to the boundaries - damage does not initiate from constrained edges
bc_alpha0 = DirichletBC(V_alpha, 0.0, bot_boundary)
bc_alpha1 = DirichletBC(V_alpha, 0.0, top_boundary)
bc_alpha = [bc_alpha0, bc_alpha1]

# Initial Conditions (IC)
#------------------------------------------------------------------------------
# Initial conditions are created by using the class defined and then
# interpolating into a finite element space
init = InitialConditions(degree=1)          # Expression requires degree def.
w_p.interpolate(init)                       # Interpolate current solution

# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants
Ic = tr(C) + (F33)**2
J = det(F)*(F33)

# Define the energy functional of the elasticity problem
# --------------------------------------------------------------------
# Nominal stress tensor
def P(u, alpha):
    return a(alpha)*mu*(F - inv(F.T)) - b_sq(alpha)*p*J*inv(F.T)

# Zero body force
body_force = Constant((0., 0.))
# Non-dimension non-negative stability parameter
varpi_ = 1.0
# Eq 19 in Klaas
varpi  = project(varpi_*h**2/(2.0*mu), FunctionSpace(mesh,'DG',0))
# Elastic energy, additional terms enforce material incompressibility and regularizes the Lagrange Multiplier
elastic_energy    = (a(alpha)+k_ell)*(mu/2.0)*(Ic-3.0-2.0*ln(J))*dx \
                    - b_sq(alpha)*p*(J-1.)*dx - 1/(2*kappa)*p**2*dx
external_work     = dot(body_force, u)*dx
elastic_potential = elastic_energy - external_work

# Define the stabilization term and the additional weak form eq.
# Compute directional derivative about w_p in the direction of v (Gradient)
F_u = derivative(elastic_potential, w_p, v_q) \
    + (a(alpha)*mu*(F33 - 1/F33) - b_sq(alpha)*p*J/F33)*v_F33*dx
    - varpi*b_sq(alpha)*J*inner(inv(C),outer(b_sq(alpha)*grad(p),grad(q)))*dx \
    - varpi*b_sq(alpha)*J*inner(inv(C),outer(grad(b_sq(alpha))*p,grad(q)))*dx \
    + varpi*b_sq(alpha)*inner(mu*(F-inv(F.T))*grad(a(alpha)),inv(F.T)*grad(q))*dx
    # - varpi*b_sq(alpha)*J*inner(inv(C), outer(grad(p),grad(q)))*dx

# Compute directional derivative about w_p in the direction of u_p (Hessian)
J_u = derivative(F_u, w_p, u_p)

# Variational problem to solve for displacement and pressure
problem_up = NonlinearVariationalProblem(F_u, w_p, bc_u, J=J_u)
# Set up the solver for displacement and pressure
solver_up  = NonlinearVariationalSolver(problem_up)
solver_up.parameters.update(solver_up_parameters)
# info(solver_up.parameters, True) # uncomment to see available parameters

# Define the energy functional of the damage problem
# --------------------------------------------------------------------
# Initializing known alpha
alpha_0 = interpolate(Expression("0.", degree=0), V_alpha)
# Define the specific energy dissipation per unit volume
z = sympy.Symbol("z", positive=True)
c_w = float(4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))
# Define the phase-field fracture term of the damage functional
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
damage_functional = elastic_potential + dissipated_energy

# Compute directional derivative about alpha in the direction of beta (Gradient)
E_alpha = derivative(damage_functional, alpha, beta)
# Compute directional derivative about alpha in the direction of dalpha (Hessian)
E_alpha_alpha = derivative(E_alpha, alpha, dalpha)

# Set the lower and upper bound of the damage variable (0-1)
alpha_lb = interpolate(Expression("0.", degree=0), V_alpha)
# alpha_lb = interpolate(Expression("x[0]>=-1.0 & x[0]<=1.0 & near(x[1], 0.0, 0.1*hsize) ? 1.0 : 0.0", \
#                        hsize = hsize, L=L, degree=0), V_alpha)
alpha_ub = interpolate(Expression("1.", degree=0), V_alpha)

# Set up the solvers
solver_alpha  = PETScTAOSolver()
solver_alpha.parameters.update(tao_solver_parameters)
# info(solver_alpha.parameters, True) # uncomment to see available parameters

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

# Split into displacement, pressure, and F33
(u, p, F33) = w_p.split()
# Data file name
file_tot = XDMFFile(MPI.comm_world, savedir + "/results.xdmf")
# Saves the file in case of interruption
file_tot.parameters["rewrite_function_mesh"] = False
file_tot.parameters["functions_share_mesh"]  = True
file_tot.parameters["flush_output"]          = True
# Write the parameters to file
File(savedir+"/parameters.xml") << userpar

timer0 = time.process_time()

# Solving at each timestep
# ----------------------------------------------------------------------------
for (i_t, t) in enumerate(load_multipliers):
    # Structure used for one printout of the statement
    if MPI.rank(MPI.comm_world) == 0:
        print("\033[1;32m--- Starting of Time step {0:2d}: t = {1:4f} ---\033[1;m".format(i_t, t))

    # Alternate Mininimization scheme
    # -------------------------------------------------------------------------
    # Solve for u holding alpha constant then solve for alpha holding u constant
    iteration = 1           # Initialization of iteration loop
    err_alpha = 1.0         # Initialization for condition for iteration

    # Conditions for iteration
    while err_alpha > AM_tolerance and iteration < maxiteration:
        # Solve elastic problem
        solver_up.solve()
        # Solve damage problem with box constraint
        solver_alpha.solve(DamageProblem(), alpha.vector(), alpha_lb.vector(), alpha_ub.vector())
        # Update the alpha condition for iteration by calculating the alpha error norm
        alpha_error = alpha.vector() - alpha_0.vector()
        err_alpha = alpha_error.norm('linf')    # Row-wise norm
        # Printouts to monitor the results and number of iterations
        volume_ratio = assemble(J/(L*H)*dx)
        if MPI.rank(MPI.comm_world) == 0:
            print ("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))
        # Update variables for next iteration
        alpha_0.assign(alpha)
        iteration = iteration + 1

    # Updating the lower bound to account for the irreversibility of damage
    alpha_lb.vector()[:] = alpha.vector()

    # Project nominal stress to tensor function space
    PTensor = project(P(u, alpha), TT)
    JScalar = project(J, P1)

    # Rename for paraview
    alpha.rename("Damage", "alpha")
    u.rename("Displacement", "u")
    p.rename("Pressure", "p")
    F33.rename("F33", "F33")
    PTensor.rename("Nominal Stress", "P")
    JScalar.rename("J", "J")

    # Write solution to file
    file_tot.write(alpha, t)
    file_tot.write(u, t)
    file_tot.write(p, t)
    file_tot.write(F33, t)
    file_tot.write(PTensor,t)
    file_tot.write(JScalar,t)

    # Update the displacement with each iteration
    u1.t = t
    u2.t = t

    # Post-processing
    # --------------------------------------------------------------------------
    # Save number of iterations for the time step
    iterations[i_t] = np.array([t, iteration])

    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)

    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, \
                              elastic_energy_value+surface_energy_value, volume_ratio])

    if MPI.rank(MPI.comm_world) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{1:6f}]".format(elastic_energy_value, surface_energy_value))
        print("\nElastic and Surface Energies: [{},{}]".format(elastic_energy_value, surface_energy_value))
        print("\nVolume Ratio: [{}]".format(volume_ratio))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir + '/stabilized-energies.txt', energies)
        np.savetxt(savedir + '/stabilized-iterations.txt', iterations)

# ----------------------------------------------------------------------------
print("elapsed CPU time: ", (time.process_time() - timer0))

# Plot energy and stresses
if MPI.rank(MPI.comm_world) == 0:
    p1, = plt.plot(energies[slice(None), 0], energies[slice(None), 1])
    p2, = plt.plot(energies[slice(None), 0], energies[slice(None), 2])
    p3, = plt.plot(energies[slice(None), 0], energies[slice(None), 3])
    plt.legend([p1, p2, p3], ["Elastic", "Dissipated", "Total"], loc="best", frameon=False)
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    plt.title('stabilized FEM')
    plt.savefig(savedir + '/stabilized-energies.pdf', transparent=True)
    plt.close()
    p4, = plt.plot(energies[slice(None), 0], energies[slice(None), 4])
    plt.xlabel('Displacement')
    plt.ylabel('Volume ratio')
    plt.title('stabilized FEM')
    plt.savefig(savedir + '/stabilized-volume-ratio.pdf', transparent=True)
    plt.close()
