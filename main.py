#!/usr/bin/python3

# ALGORITHM FILE
# Main file of PSO-tool, a personal academic project, for further information visit its GitHub page: github.com/rodosara/PSO-tool
# Rodolfo Saraceni

# Import section
import random, math, time
import numpy as np
from tqdm import tqdm
from operator import attrgetter
from scipy.spatial import distance
import sobol_seq
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import benchmark_function as bf # Import the file with all the benchmark functions to compare

# CLASS PARTICLE DEFINITION
class Particle:
    def __init__(self, number, position, pBest, pBest_position, pBest_viol, pBest_feasible, velocity, w, c1, c2):
        self.id = number
        self.pos = position
        self.pBest = pBest
        self.pBest_pos = pBest_position
        self.pBest_viol= pBest_viol
        self.pBest_feasible = pBest_feasible
        self.Vid = velocity
        self.w = w
        self.c1 = c1
        self.c2 = c2

# CREATE INITIAL POPULATION
def init_pop(pop_size): # Create population that verify all constraints
    pbar = tqdm(total=pop_size)
    pbar.set_description("Initialize population")
    swarm, pos_part = [], np.zeros(0)
    while len(swarm) < pop_size: # Create a population of particle of pop_size dimension appengind class in a list
        if len(eval("bf."+opt_func+".bounds")) != 0: # If there are bounds create a particle inside
             for k in range(len(eval("bf."+opt_func+".bounds"))):
                bound = (eval("bf."+opt_func+".bounds"))[k]
                pos = np.random.uniform(bound[0], bound[1], 1)
                pos_part = np.append(pos_part, pos) 
        else: # If there isn't bound create random position
            pos_part = np.random.uniform(-500, 500, eval("bf."+opt_func+".dimension"))

        if eval("bf."+opt_func+".verify_const(pos_part)") == True: # Verify if the particle respect the constraints
            swarm.append(Particle (len(swarm), pos_part, eval("bf."+opt_func+".function(pos_part)"), pos_part, 0, 0, 0, 0, 0, 0))
            pbar.update(1)

        pos_part = np.zeros(0)

    pbar.close()
    return swarm

def init_pop_nc(pop_size): # Create a population within the bounds
    pbar = tqdm(total=pop_size)
    pbar.set_description("Initialize population") 
    swarm, pos_part = [], np.zeros(0)
    while len(swarm) < pop_size: # Create a population of particle of pop_size dimension appengind class in a list
        if len(eval("bf."+opt_func+".bounds")) != 0: # If there are bounds create a particle inside
            for k in range(len(eval("bf."+opt_func+".bounds"))):
                bound = (eval("bf."+opt_func+".bounds"))[k]
                pos = np.random.uniform(bound[0], bound[1], 1)
                pos_part = np.append(pos_part, pos)
        else: # If there isn't bound create random position
            pos_part = np.random.uniform(-500, 500, eval("bf."+opt_func+".dimension")) 

        swarm.append(Particle (len(swarm), pos_part, eval("bf."+opt_func+".function(pos_part)"), pos_part, 0, 0, 0, 0, 0, 0)) # Append particle without verify constraints
        pbar.update(1)

        pos_part = np.zeros(0) 

    pbar.close()
    return swarm

def opposite_pop(swarm): # From a popolation create an opposite one
    opposite_swarm, index = [], 0 
    for obj in swarm:
        if len(eval("bf."+opt_func+".bounds")) == eval("bf."+opt_func+".dimension"): # Only if there are have bounds for each dimension
            opp_pos = np.zeros(0)
            for k in range(len(eval("bf."+opt_func+".bounds"))):
                bound = eval("bf."+opt_func+".bounds")[k] # Obtain the bounds from benchmark function file [a,b]
                opposite = (bound[0]+bound[1]) - obj.pos[k] # Compute the opposite particle with (a+b)-pos for each dimension
                opp_pos = np.append(opp_pos, opposite)
            opposite_swarm.append(Particle (index, opp_pos, eval("bf."+opt_func+".function(opp_pos)"), opp_pos, 0, 0, 0, 0, 0, 0)) # Append each opposite particle in a new list
            index += 1
    joined_swarm = swarm + opposite_swarm # Join the original and the opposite swarm
    joined_swarm.sort(key=lambda x: x.pBest, reverse=False) # Sort the all swarm base on the pBest values
    swarm = joined_swarm[0:len(swarm)] # Pick the first population size elements

    print("OPPOSING POPULATION...")
    return swarm

# FUNCTION FOR THE THREE PSO ALGORITHMS
def pso1(): # PSO with True/False penalty approach
    # PARAMETERS DEFINITION
    space_dim = eval("bf."+opt_func+".dimension")
    pop_size = 20
    # Iteration number
    if opt_func == "g1_7" or opt_func == "g1_10":
        iteration = 2000
    else:
        iteration = 200
    V_max = 10
    w = 0.5 + random.uniform(0, 1)/2.0
    c1 = 1.49445
    c2 = c1

    gBest_iter, swarm_pos_iter = np.zeros(0), []
    
    swarm = init_pop(pop_size) # Generate population
    if use_opposite: # Use opposite population
        swarm = opposite_pop(swarm)

    pbar = tqdm(total=iteration)
    pbar.set_description("PSO True/False penalty")
    
    for obj in swarm:
        obj.pBest = eval("bf."+opt_func+".function(obj.pos)")

    gBest = min(swarm, key=attrgetter('pBest')).pBest # Compute initial gBest
    gBest_pos = min(swarm, key=attrgetter('pBest')).pBest_pos

    for l in range(iteration): # Iteration loop
        swarm_pos_iter.append([bird.pos for bird in swarm]) # Append swarm list for animation graph

        for obj in swarm:
            fitting_val  = eval("bf."+opt_func+".function(obj.pos)") # Calc fitting value for each particle

            if fitting_val < obj.pBest and eval("bf."+opt_func+".verify_const(obj.pos)") == True: # Update pBest
                obj.pBest = fitting_val
                obj.pBest_pos = obj.pos

        if min(swarm, key=attrgetter('pBest')).pBest < gBest: # Update gBest
            gBest = min(swarm, key=attrgetter('pBest')).pBest
            gBest_pos = min(swarm, key=attrgetter('pBest')).pBest_pos
        
        w = 0.5 + random.uniform(0, 1)/2.0 # Compute random inertial weight

        for obj in swarm: # Update position
            obj.Vid = w*obj.Vid + c1*random.uniform(0,1)*(obj.pBest_pos-obj.pos) + c2*random.uniform(0,1)*(gBest_pos-obj.pos)
            obj.Vid[obj.Vid > V_max] = V_max # Control the max velocity
            obj.Vid[obj.Vid < -V_max] = -V_max
            obj.pos = obj. pos + obj.Vid
        
        gBest_iter = np.append(gBest_iter, gBest)
        pbar.update(1) 

    pbar.close()
    return gBest, gBest_pos, gBest_iter, swarm_pos_iter

def pso2(): # PSO penalty function approach
    def penalty_func(pos, it): # Penalty function 
        def gamma_func(q_val): # Gamma function 
            if q_val < 1:
                return 1
            else:
                 return 2

        def theta_func(q_val): # Theta function
            if q_val < 0.001:
                return 10
            elif 0.001 <= q_val <= 0.1:
                return 20
            elif 0.1 < q_val <= 1:
                return 100
            else:
                return 300

        #h_val = math.sqrt(it) # Used only per test case one
        h_val = it*math.sqrt(it)
        H_val, viol_sum = 0, 0
        tot_inequal_constraints = len(eval("bf."+opt_func+".inequal_constraints")) + len(eval("bf."+opt_func+".bounds")*2)

        for m in range(tot_inequal_constraints):
            viol_val = eval("bf."+opt_func+".pick_inequal_bounds_const(pos, m)") # Compute violation value of g(x)
            if viol_val > violation_tol: # Verify if is greater than violation tolerance
                q_val = max(0,viol_val)
                viol_sum += q_val
                H_val += theta_func(q_val)*q_val**gamma_func(q_val) # Calc H value
        return H_val*h_val, viol_sum

    # PARAMETERS DEFINITION
    space_dim = eval("bf."+opt_func+".dimension")
    Vid = 0
    pop_size = 100
    iteration = 1000
    Xi = 0.73
    V_max = 4
    w = 1.2 # Decrease each iteration [0.1, 1.2]
    c1 = 2
    c2 = c1
    violation_tol = 10**(-5)

    gBest_iter, swarm_pos_iter = np.zeros(0), []

    swarm = init_pop_nc(pop_size) # Generate an initial population without constraints
    if use_opposite: # Use opposite population
        swarm = opposite_pop(swarm)

    pbar = tqdm(total=iteration)
    pbar.set_description("PSO Penalty function")
    
    for obj in swarm: # Compute che value of penalty function per each inital pBest
        violation_val, violation_sum = penalty_func(obj.pos, 1) # Compute value of penalty for each particle
        obj.pBest = eval("bf."+opt_func+".function(obj.pos)") + violation_val
        obj.pBest_viol = violation_sum

    gBest = min(swarm, key=attrgetter('pBest')).pBest # Find initial gBest
    gBest_pos = min(swarm, key=attrgetter('pBest')).pBest_pos
    gBest_viol = min(swarm, key=attrgetter('pBest')).pBest_viol

    for l in range(1, iteration+1, 1): # Iteration loop
        swarm_pos_iter.append([bird.pos for bird in swarm]) # Append swarm list for animation graph

        for obj in swarm: # Update position of each particles
            obj.Vid = Xi*(w*obj.Vid + c1*np.dot(random.uniform(0,1),(obj.pBest_pos - obj.pos)) + c2*np.dot(random.uniform(0,1),(gBest_pos - obj.pos)))

            obj.Vid[obj.Vid > V_max] = V_max # Control the max velocity
            obj.Vid[obj.Vid < -V_max] = -V_max
            obj.pos = obj.pos + obj.Vid

            violation_val, violation_sum = penalty_func(obj.pos, l) # Compute value of penalty for each particle
            fitting_val = eval("bf."+opt_func+".function(obj.pos)") + violation_val # Compute value of penalty for each particle

            if fitting_val < obj.pBest: # Update pBest
                obj.pBest = fitting_val
                obj.pBest_pos = obj.pos
                obj.pBest_viol = violation_sum

        if min(swarm, key=attrgetter('pBest')).pBest < gBest: # Update gBest
            gBest = min( swarm, key=attrgetter('pBest')).pBest
            gBest_pos = min(swarm, key=attrgetter('pBest')).pBest_pos
            gBest_viol = min(swarm, key=attrgetter('pBest')).pBest_viol
        
        w = w - 0.0011 # Decrease inertial weight for each iteration
        
        gBest_iter = np.append(gBest_iter, eval("bf."+opt_func+".function(gBest_pos)")) # Real value of gBest are provided (without value's penalty function)
        pbar.update(1) 

    pbar.close()
    print("--> gBest_viol", gBest_viol, "gBest real:", eval("bf."+opt_func+".function(gBest_pos)")) # Print the sum of constraints violation (only for this algorithm)
    return eval("bf."+opt_func+".function(gBest_pos)"), gBest_pos, gBest_iter, swarm_pos_iter # Real value ogBest are provided (without value's penalty function)

def pso3(): # PSO SAPSO 2011
    def point_hypersphere(dimension, centre, radius): # Pick a random point inside hypersphere
        v = np.random.uniform(0,1,dimension) # Create a random point of function dimension
        inv_len = 1.0 / math.sqrt(sum(coord * coord for coord in v)) # Normalize it
        norm = v * inv_len
        return (centre - norm*radius) # Traslate the centre and correct the radius

    def violation_function(pos_part): # Violation function
        equal_viol_sum, inequal_viol_sum = 0, 0

        for m in range(len(eval("bf."+opt_func+".inequal_constraints"))): # Calc inequal constraints sum
            inequal_viol_sum += max(0,eval("bf."+opt_func+".pick_inequal_const(pos_part, m)"))
        for m in range(len(eval("bf."+opt_func+".equal_constraints"))): # Calc equal constraints sum
            equal_viol_sum += abs(eval("bf."+opt_func+".pick_equal_const(pos_part, m)"))

        return equal_viol_sum + inequal_viol_sum

    def saturation_strategy(pos_part): # Saturation strategy function
        for k in range(len(pos_part)):
            bound = np.array(eval("bf."+opt_func+".bounds")[k]) # Pick bounds from benchmark function
            if len(bound) != 0:
                if pos_part[k] < bound[0]:
                    pos_part[k] = bound[0] # Correct lower bound
                if pos_part[k] > bound[1]:
                    pos_part[k] = bound[1] # Correct upper bound
        return pos_part

    def update_parameters(pBest_pos, gBest_pos, iter_num):
        beta = distance.euclidean(gBest_pos, pBest_pos) # Calc the beta
        w = (w_s-w_f) * math.exp((-gamma_w*iter_num)/beta) + w_f # Update velocity
        c1 = (c1_s-c1_f) * math.exp((-gamma_c1*iter_num)/beta) + c1_f # Update cognitive factor 
        c2 = (c2_s-c2_f) * math.exp((gamma_c2*iter_num)/beta) + c2_f # Update social factor
        return [w, c1, c2]
 
    # PARAMETERS DEFINITION
    space_dim = eval("bf."+opt_func+".dimension")
    pop_size = 100
    iteration = 4000
    w_s = 0.9 # Initial inertial weight
    w_f = 0.1 # Final inertial weight
    c1_s = 2.5 # Initial cognitive factor
    c1_f = 0.1 # Final cognitive factor
    c2_s = 0.1 # Initial social factor
    c2_f = 2.5 # Final social factor

    # Compute the gamma factor for each parameters
    gamma_w = (w_s - w_f) / iteration
    gamma_c1 = (c1_s - c1_f) / iteration
    gamma_c2 = (c2_s - c2_f) / iteration

    gBest_iter, swarm_pos_iter = np.zeros(0), []
    
    swarm = init_pop_nc(pop_size) # Generate an initial population without constraints

    pbar = tqdm(total=iteration)
    pbar.set_description("PSO SAPSO 2011")
    
    # Compute initial relaxation value as median of each particles violation
    initial_violation = np.zeros(0)
    for obj in swarm:
        obj.w, obj.c1, obj.c2 = w_s, c1_s, c2_s # Set initial parameters
        obj.pBest_viol = violation_function(obj.pos)
        initial_violation = np.append(initial_violation, obj.pBest_viol)
    sigma = np.median(initial_violation)

    # Set violation of each pBest
    for obj in swarm:
        if obj.pBest_viol <= sigma:
            obj.pBest_feasible = True
        else:
            obj.pBest_feasible = False
    
    # Compute first gBest
    gBest = min(swarm, key=attrgetter('pBest')).pBest
    gBest_pos = min(swarm, key=attrgetter('pBest')).pBest_pos
    gBest_viol = min(swarm, key=attrgetter('pBest')).pBest_viol
    gBest_feasible = min(swarm, key=attrgetter('pBest')).pBest_feasible

    for l in range(1, iteration+1, 1):
        Ff = 0  # Set 0 the feasible solution
        swarm_pos_iter.append([bird.pos for bird in swarm]) # Append swarm list for animation graph

        for obj in swarm: # Update the particle position
            bar = obj.pos + ((obj.c1*(obj.pBest_pos-obj.pos) + obj.c2*(gBest_pos-obj.pos)) / 3) # Compute the baricenter
            if all(obj.pBest_pos == gBest_pos):
                bar = (obj.pos + obj.pos + obj.c2*(gBest_pos-obj.pos))/2

            radius = distance.euclidean(bar, obj.pos)
            X = point_hypersphere(space_dim, bar, random.uniform(0, radius)) # Pick random point
            
            obj.Vid = obj.w*obj.Vid + X - obj.pos # Update velocity and position
            obj.pos = obj.Vid + obj.pos

            obj.pos = saturation_strategy(obj.pos) # Applied the saturation strategy
            violation_sum = violation_function(obj.pos) # Compute the violation value
            
            if violation_sum <= sigma: # Update the number of feasible solution
                Ff += 1

            fitting_val = eval("bf."+opt_func+".function(obj.pos)") + violation_sum # Compute the value of the function
            
            # Feasibility-based rule for pBest updating
            if (obj.pBest_feasible == False and
            violation_sum <= sigma):  # Case where current solution is feasible and previous one not
                obj.pBest_viol = violation_sum
                obj.pBest = fitting_val
                obj.pBest_pos = obj.pos
                obj.pBest_feasible = True
            elif (obj.pBest_feasible == True and
            violation_sum <= sigma and
            fitting_val < obj.pBest): # Case where previous and current solution are both unfeasible
                obj.pBest_viol = violation_sum
                obj.pBest = fitting_val
                obj.pBest_pos = obj.pos
                obj.pBest_feasible = True
            elif (obj.pBest_feasible == False and
            violation_sum > sigma and
            violation_sum < obj.pBest_viol): # Case where previous and current solution are both unfeasible
                obj.pBest_viol = violation_sum
                obj.pBest = fitting_val
                obj.pBest_pos = obj.pos
                obj.pBest_feasible = False

        # Feasibility-based rule for gBest updating
        if (gBest_feasible == False and # Case where current solution is feasible and previous one not
        min(swarm, key=attrgetter('pBest')).pBest_feasible == True):
            gBest_viol = min(swarm, key=attrgetter('pBest')).pBest_viol
            gBest = min(swarm, key=attrgetter('pBest')).pBest
            gBest_pos = min(swarm, key=attrgetter('pBest')).pBest_pos
            gBest_feasible = True
        elif (gBest_feasible == True and  # Case where previous and current solution are both feasible
        min(swarm, key=attrgetter('pBest')).pBest_feasible == True and
        min(swarm, key=attrgetter('pBest')).pBest < gBest):
            gBest_viol = min(swarm, key=attrgetter('pBest')).pBest_viol
            gBest = min(swarm, key=attrgetter('pBest')).pBest
            gBest_pos = min(swarm, key=attrgetter('pBest')).pBest_pos
            gBest_feasible = True
        elif (gBest_feasible == False and # Case where previous and current solution are both unfeasible 
        min(swarm, key=attrgetter('pBest')).pBest_feasible == False and
        min(swarm, key=attrgetter('pBest')).pBest_viol < gBest_viol):
            gBest_viol = min(swarm, key=attrgetter('pBest')).pBest_viol
            gBest = min(swarm, key=attrgetter('pBest')).pBest
            gBest_pos = min(swarm, key=attrgetter('pBest')).pBest_pos
            gBest_feasible = False

        for obj in swarm: # Update the algorithms parameters
            if np.array_equal(gBest_pos,obj.pBest_pos) == False:
                obj.w, obj.c1, obj.c2 = update_parameters(gBest_pos, obj.pBest_pos, l)
        
        sigma = sigma*(1 - Ff/pop_size) # Decrease the relaxation value

        gBest_iter = np.append(gBest_iter, gBest)
        pbar.update(1)

    print("--> gBest_viol:", gBest_viol)

    pbar.close()
    return gBest, gBest_pos, gBest_iter, swarm_pos_iter

def pyswarm_library(): # PSO using Pysmwarm library
    from pyswarm import pso # link https://pythonhosted.org/pyswarm/
    # PARAMETERS DEFINITION
    pop_size = 500
    iteration = 1000
    w = 0.5
    c1 = 0.5
    c2 = 0.5
    minstep = 1e-3
    minfunc = 1e-3
    debug = False

    print("Use default parameters:\n" # Default parameters
        "\t\t\tpop_size =", pop_size, "\n",
        "\t\t\titeration =", iteration, "\n",
        "\t\t\tw =", w, "\n",
        "\t\t\tc1 =", c1, "\n",
        "\t\t\tc2 =", c2, "\n",
        "\t\t\tminstep =", minstep, "\n",
        "\t\t\tminfunc =", minfunc, "\n",
        "\t\t\tdebug =", debug, "\n\n",
        "or set personal ones? [D for default|P for personal]")

    param = input() # Set personal parameters
    if param == "P":
        print("Size of population = [int]")
        pop_size = eval(input())
        print("Number of interation = [int]")
        iteration = eval(input())
        print("Inertial weight = [float]")
        w = eval(input())
        print("Cognitive coefficient = [float]")
        c1 = eval(input())
        print("Social coefficient [float]")
        c2 = eval(input())
        print("The minimum stepsize of swarm’s best position before the search terminates = [int]")
        minstep = eval(input())
        print("Minimum change of swarm’s best objective value before the search terminates = [int]")
        minfunc = eval(input())
        print("Active debyg = [boolean]")
        debug = eval(input()) 

    def constraints(x): # Pick the constraints from benchmarck_function file
        for k in range(len(x)):
            locals()[f"x{k+1}"] = x[k]
        constraints = eval("bf."+opt_func+".inequal_constraints")
        positive_constraints = []
        for k in range(len(constraints)):
            positive_constraints.extend([eval(constraints[k])*-1]) # Library use >= notation
        return positive_constraints 

    lb, ub = [], []
    for k in range(len(eval("bf."+opt_func+".bounds"))): # Pick the bounds of the fitting function
        lb.extend([eval("bf."+opt_func+".bounds")[k][0]])
        ub.extend([eval("bf."+opt_func+".bounds")[k][1]]) 

    start_time = time.time() # Compute gBest with library function
    gBest_pos, gBest = pso(eval("bf."+opt_func+".function"), lb, ub, f_ieqcons=constraints, swarmsize=pop_size, omega=w, phip=c1, phig=c2, maxiter=iteration, minstep=minstep, minfunc=minfunc, debug=debug)
    end_time = time.time()
    print("ELAPSED TIME:", f"{end_time-start_time:.0f}", "sec for", runs, "runs")

    return gBest, gBest_pos

# GRAPH SECTION
def graph_gBest(data_graph, run): # Plot the gBest behavoir along the iteration
    fig = plt.figure("gBest iteration progress")
    plt.plot(np.linspace(1,len(data_graph),len(data_graph)), data_graph, label=f"Runs {k}")
    plt.title("gBest iteration progress")
    plt.xlabel("Iteration")
    plt.ylabel("gBest value")
    plt.legend()

def graph_iteration(data_graph, gBest_pos, opt_func):
    def update(frame, data, gBest_pos, bounds, minimum=[], ): # Function for update snapshot
        x_data, y_data = np.zeros(0), np.zeros(0)
        for k in range(len(data[frame])): # Plot particle in 2d space
            x_data = np.append(x_data, data[frame][k][0])
            y_data = np.append(y_data, data[frame][k][1])
        storm_plot = ax.scatter(x_data, y_data, facecolors="none", edgecolors="b")
        
        gBest_plot = ax.scatter(gBest_pos[0], gBest_pos[1], s=80, c='r', marker='x', label="gBest") # Plot gBest position with X marker

        xmin, xmax, ymin, ymax = bounds # Plot area of feasible space from bounds
        bounds_space = patch.Rectangle((xmin,ymin), width=(xmax-xmin), height=(ymax-ymin), fill=True, alpha=0.1, linestyle='--', color='g', label="Bounds space")
        bounds_plot = ax.add_patch(bounds_space)

        if eval("bf."+opt_func+".solution") != []:
            minimum_plot = ax.scatter(minimum[0], minimum[1], s=80, c='m', marker='*', label="Minimum") # Plot minimum position with * marker
            return storm_plot, gBest_plot, bounds_plot, minimum_plot, 
        else:
            return storm_plot, gBest_plot, bounds_plot,  

    fig, ax = plt.subplots(figsize=(12,9)) # Setting all graph parameters
    fig.canvas.set_window_title("Swarm iteration movements")
    ax.set_title("Swarm iteration movements")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    legend_patch = [Line2D([0], [0], marker='o', color='b', lw=0,  fillstyle="none", label="Particle"),
                    Line2D([0], [0], marker='x', color='r', lw=0, label="gBest"),
                    Line2D([0], [0], marker='*', color='m', lw=0, label="Minimum"),
                    patch.Patch(facecolor='g', alpha=0.1, linestyle='--', label="Bounds space")]
    fig.legend(handles=legend_patch) # Set the legend

    bounds = [] # Pick the bound from benchmark function file
    if len(eval("bf."+opt_func+".bounds")) != 0:
        for k in range(2):
            bounds.extend(eval("bf."+opt_func+".bounds")[k])

    if eval("bf."+opt_func+".solution") != []:
        minimum = [eval("bf."+opt_func+".solution")[0], eval("bf."+opt_func+".solution")[1]] # Pick the minimum from benchmark function file
        anim = animation.FuncAnimation(fig, update, frames=len(data_graph), fargs=(data_graph, gBest_pos, bounds, minimum, ), interval=200, blit=True, repeat=False) # Create animation of following snapshot
    else:
        anim = animation.FuncAnimation(fig, update, frames=len(data_graph), fargs=(data_graph, gBest_pos, bounds, ), interval=200, blit=True, repeat=False) # Create animation of following snapshot
    
# MAIN LOOP
print("\nWhich PSO algorithm do you want use?")
method = eval(input("[1] = True/False penalty approach\n[2] = Penalty function approach\n[3] = SAPSO 2011\n[4] = Py_swarm python library\n"))
use_opposite = bool(int(input("Do you want use an opposite population? [0|1]: ")))
runs = eval(input("How many runs? [int]: "))
opt_func = input("Which function do you want optimize: ")
print("")

gBest_runs, gBest_pos_runs = np.zeros(0), []

for k in range (1, runs+1): # Switch case to manage all algorithms
    print("RUN NUMBER:", k)
    if method == 1:
        gBest, gBest_pos, gBest_iter, swarm_pos_iter = pso1()
    elif method == 2:
        gBest, gBest_pos, gBest_iter, swarm_pos_iter = pso2()
    elif method == 3:
        gBest, gBest_pos, gBest_iter, swarm_pos_iter = pso3()
    elif method == 4:
        gBest, gBest_pos = pyswarm_library()

    if method != 4: # Call graph function
        graph_iteration(swarm_pos_iter, gBest_pos, opt_func)
        graph_gBest(gBest_iter, k) 

    gBest_runs = np.append(gBest_runs, gBest)
    gBest_pos_runs.append(gBest_pos)
    
    print()

# PRINT RESULTS
print()
print("----------------- RESULT -----------------")
print("--> TOTAL gBest", gBest_runs, "\n") # gBest in each runs
print("--> MEAN SOLUTION:", "\n\tgBest:", gBest_runs.mean(),"\n\twith STANDARD DEVIATION:", gBest_runs.std(), "\n\tand with ERROR:", gBest_runs.mean() - eval("bf."+opt_func+".minimum"), "\n") # Mean gBest for all runs
print("--> BEST SOLUTION:", "\n\tgBest:", np.min(gBest_runs), "\n\twith ERROR:", np.min(gBest_runs) - eval("bf."+opt_func+".minimum"),"\n\twith position:", gBest_pos_runs[gBest_runs.argmin()], "\n") # Best gBest from all runs
print("--> WORST SOLUTION:", "\n\tgBest:", np.max(gBest_runs), "\n\twith ERROR:", np.max(gBest_runs) - eval("bf."+opt_func+".minimum"),"\n\twith position:", gBest_pos_runs[gBest_runs.argmax()], "\n") # Worst gBest from all runs 
print()

plt.show() # Show all graphs
