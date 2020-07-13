#!/usr/bin/env  python3

"""
ALGORITHM FILE
Main file of PSO-tool, a personal academic project
For further information visit its GitHub page: github.com/rodosara/PSO-tool
Rodolfo Saraceni
"""


import random
import math
import time
import numpy as np
from pyswarm import pso # link https://pythonhosted.org/pyswarm/
from tqdm import tqdm
from operator import attrgetter
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import benchmark_function as bf

class Particle:
    """ Class Particle Definition"""

    def __init__(self, number, position, pbest, pbest_position,
                 pbest_viol, pbest_feasible, velocity, w, c1, c2):
        self.id = number
        self.pos = position
        self.pbest = pbest
        self.pbest_pos = pbest_position
        self.pbest_viol= pbest_viol
        self.pbest_feasible = pbest_feasible
        self.Vid = velocity
        self.w = w
        self.c1 = c1
        self.c2 = c2

def init_pop(pop_size):
    """Create population that verify all constraints"""

    pbar = tqdm(total=pop_size)
    pbar.set_description("Initialize population")
    swarm = []
    pos_part = np.zeros(0)

    # Create a population of particle of pop_size dimension appengind class in a list
    while len(swarm) < pop_size:
    # If there are bounds create a particle inside
        if len(eval("bf." + opt_func + ".bounds")) != 0:
             for k in range(len(eval("bf." + opt_func + ".bounds"))):
                bound = (eval("bf." + opt_func + ".bounds"))[k]
                pos = np.random.uniform(bound[0], bound[1], 1)
                pos_part = np.append(pos_part, pos) 
        # If there isn't bound create random position
        else:
            pos_part = np.random.uniform(-500, 500, eval("bf." + opt_func + ".dimension"))

        # Verify if the particle respect the constraints
        if eval("bf." + opt_func + ".verify_const(pos_part)") == True:
            swarm.append(Particle (len(swarm), pos_part, eval("bf." + opt_func + ".function(pos_part)"), pos_part, 0, 0, 0, 0, 0, 0))
            pbar.update(1)

        pos_part = np.zeros(0)

    pbar.close()

    return swarm

def init_pop_nc(pop_size):
    """Create a population within the bounds"""

    pbar = tqdm(total=pop_size)
    pbar.set_description("Initialize population")
    swarm = []
    pos_part = np.zeros(0)

    # Create a population of particle of pop_size dimension appengind class in a list
    while len(swarm) < pop_size:
        # If there are bounds create a particle inside
        if len(eval("bf." + opt_func + ".bounds")) != 0:
            for k in range(len(eval("bf." + opt_func + ".bounds"))):
                bound = (eval("bf." + opt_func + ".bounds"))[k]
                pos = np.random.uniform(bound[0], bound[1], 1)
                pos_part = np.append(pos_part, pos)
        # If there isn't bound create random position
        else:
            pos_part = np.random.uniform(-500, 500, eval("bf." + opt_func + ".dimension"))

        # Append particle without verify constraints
        swarm.append(Particle (len(swarm), pos_part, eval("bf." + opt_func + ".function(pos_part)"), pos_part, 0, 0, 0, 0, 0, 0))
        pbar.update(1)

        pos_part = np.zeros(0)

    pbar.close()

    return swarm

def opposite_pop(swarm):
    """From a popolation create an opposite one"""

    opposite_swarm = []
    index = 0
 
    for obj in swarm:
        # Only if there are have bounds for each dimension
        if len(eval("bf." + opt_func + ".bounds")) == eval("bf." + opt_func + ".dimension"):
            opp_pos = np.zeros(0)
            for k in range(len(eval("bf." + opt_func + ".bounds"))):
                # Obtain the bounds from benchmark function file [a,b]
                bound = eval("bf." + opt_func + ".bounds")[k]
                # Compute the opposite particle with (a+b)-pos for each dimension
                opposite = (bound[0]+bound[1]) - obj.pos[k]
                opp_pos = np.append(opp_pos, opposite)
            # Append each opposite particle in a new list
            opposite_swarm.append(Particle (index, opp_pos, eval("bf." + opt_func + ".function(opp_pos)"), opp_pos, 0, 0, 0, 0, 0, 0))
            index += 1

    # Join the original and the opposite swarm
    joined_swarm = swarm + opposite_swarm
    # Sort the all swarm base on the pbest values
    joined_swarm.sort(key=lambda x: x.pbest, reverse=False)
    # Pick the first population size elements
    swarm = joined_swarm[0:len(swarm)] 

    print("OPPOSING POPULATION...")

    return swarm

# FUNCTION FOR THE THREE PSO ALGORITHMS

def pso1():
    """PSO with True/False penalty approach"""

    # PARAMETERS DEFINITION
    space_dim = eval("bf." + opt_func + ".dimension")
    pop_size = 20
    # Iteration number
    if opt_func == "g1_7" or opt_func == "g1_10":
        iteration = 2000
    else:
        iteration = 200
    V_max = 10
    w = 0.5 + random.uniform(0, 1) / 2.0
    c1 = 1.49445
    c2 = c1

    gbest_iter = np.zeros(0)
    swarm_pos_iter = []

    # Generate population
    swarm = init_pop(pop_size)
    # Use opposite population
    if use_opposite:
        swarm = opposite_pop(swarm)

    pbar = tqdm(total=iteration)
    pbar.set_description("PSO True/False penalty")

    for obj in swarm:
        obj.pbest = eval("bf." + opt_func + ".function(obj.pos)")

    # Compute initial gbest
    gbest = min(swarm, key=attrgetter('pbest')).pbest
    gbest_pos = min(swarm, key=attrgetter('pbest')).pbest_pos

    for l in range(iteration):
        # Append swarm list for animation graph
        swarm_pos_iter.append([bird.pos for bird in swarm])

        for obj in swarm:
            # Calc fitting value for each particle
            fitting_val  = eval("bf." + opt_func + ".function(obj.pos)")

            # Update pbest
            if fitting_val < obj.pbest and eval("bf." + opt_func + ".verify_const(obj.pos)") == True:
                obj.pbest = fitting_val
                obj.pbest_pos = obj.pos

        # Update gbest
        if min(swarm, key=attrgetter('pbest')).pbest < gbest:
            gbest = min(swarm, key=attrgetter('pbest')).pbest
            gbest_pos = min(swarm, key=attrgetter('pbest')).pbest_pos
        
        # Compute random inertial weight
        w = 0.5 + random.uniform(0, 1) / 2.0

        for obj in swarm: # Update position
            obj.Vid = w*obj.Vid + \
                c1*random.uniform(0,1)*(obj.pbest_pos-obj.pos) + \
                c2*random.uniform(0,1)*(gbest_pos-obj.pos)

            # Control the max velocity
            obj.Vid[obj.Vid > V_max] = V_max
            obj.Vid[obj.Vid < -V_max] = -V_max
            obj.pos = obj. pos + obj.Vid

        gbest_iter = np.append(gbest_iter, gbest)
        pbar.update(1)

    pbar.close()

    return gbest, gbest_pos, gbest_iter, swarm_pos_iter

def pso2():
    """PSO penalty function approach"""

    def penalty_func(pos, it):
        """Penalty function"""

        def gamma_func(q_val):
            """Gamma function"""
            if q_val < 1:
                return 1
            else:
                return 2

        def theta_func(q_val):
            """Theta function"""

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
        tot_inequal_constraints = len(eval("bf." + opt_func + ".inequal_constraints")) + len(eval("bf." + opt_func + ".bounds")*2)

        for m in range(tot_inequal_constraints):
            # Compute violation value of g(x)
            viol_val = eval("bf." + opt_func + ".pick_inequal_bounds_const(pos, m)")
            # Verify if is greater than violation tolerance
            if viol_val > violation_tol:
                q_val = max(0,viol_val)
                viol_sum += q_val
                # Calc H value
                H_val += theta_func(q_val)*q_val**gamma_func(q_val)

        return H_val*h_val, viol_sum

    # PARAMETERS DEFINITION
    space_dim = eval("bf." + opt_func + ".dimension")
    Vid = 0
    pop_size = 100
    iteration = 1000
    Xi = 0.73
    V_max = 4
    w = 1.2  # Decrease each iteration [0.1, 1.2]
    c1 = 2
    c2 = c1
    violation_tol = 10**(-5)

    gbest_iter = np.zeros(0)
    swarm_pos_iter = []

    # Generate an initial population without constraints
    swarm = init_pop_nc(pop_size)
    # Use opposite population
    if use_opposite:
        swarm = opposite_pop(swarm)

    pbar = tqdm(total=iteration)
    pbar.set_description("PSO Penalty function")
    
    # Compute che value of penalty function per each inital pbest
    for obj in swarm:
        # Compute value of penalty for each particle
        violation_val, violation_sum = penalty_func(obj.pos, 1)
        obj.pbest = eval("bf." + opt_func + ".function(obj.pos)") + violation_val
        obj.pbest_viol = violation_sum

    # Find initial gbest
    gbest = min(swarm, key=attrgetter('pbest')).pbest
    gbest_pos = min(swarm, key=attrgetter('pbest')).pbest_pos
    gbest_viol = min(swarm, key=attrgetter('pbest')).pbest_viol

    for l in range(1, iteration+1, 1):
        # Append swarm list for animation graph
        swarm_pos_iter.append([bird.pos for bird in swarm])

        # Update position of each particles
        for obj in swarm:
            obj.Vid = Xi*(w*obj.Vid + \
                        c1*np.dot(random.uniform(0,1),(obj.pbest_pos - obj.pos)) + \
                        c2*np.dot(random.uniform(0,1),(gbest_pos - obj.pos)))

            # Control the max velocity
            obj.Vid[obj.Vid > V_max] = V_max
            obj.Vid[obj.Vid < -V_max] = -V_max
            obj.pos = obj.pos + obj.Vid

            # Compute value of penalty for each particle
            violation_val, violation_sum = penalty_func(obj.pos, l)
            # Compute value of penalty for each particle
            fitting_val = eval("bf." + opt_func + ".function(obj.pos)") + violation_val

            # Update pbest
            if fitting_val < obj.pbest:
                obj.pbest = fitting_val
                obj.pbest_pos = obj.pos
                obj.pbest_viol = violation_sum

        # Update gbest
        if min(swarm, key=attrgetter('pbest')).pbest < gbest:
            gbest = min( swarm, key=attrgetter('pbest')).pbest
            gbest_pos = min(swarm, key=attrgetter('pbest')).pbest_pos
            gbest_viol = min(swarm, key=attrgetter('pbest')).pbest_viol
        
        w = w - 0.0011 # Decrease inertial weight for each iteration

        # Real value of gbest are provided (without value's penalty function)
        gbest_iter = np.append(gbest_iter, eval("bf." + opt_func + ".function(gbest_pos)"))
        pbar.update(1) 

    pbar.close()
    # Print the sum of constraints violation (only for this algorithm)
    print("--> gbest_viol", gbest_viol, "gbest real:", eval("bf." + opt_func + ".function(gbest_pos)"))

    # Real value ogbest are provided (without value's penalty function)
    return eval("bf." + opt_func + ".function(gbest_pos)"), gbest_pos, gbest_iter, swarm_pos_iter

def pso3():
    """PSO SAPSO 2011"""

    def point_hypersphere(dimension, centre, radius):
        """Pick a random point inside hypersphere"""

        # Create a random point of function dimension
        v = np.random.uniform(0,1,dimension)
        # Normalize it
        inv_len = 1.0 / math.sqrt(sum(coord * coord for coord in v))
        norm = v * inv_len

        # Traslate the centre and correct the radius
        return (centre - norm*radius)

    def violation_function(pos_part):
        """Violation function"""
        equal_viol_sum = 0
        inequal_viol_sum = 0

        # Calc inequal constraints sum
        for m in range(len(eval("bf." + opt_func + ".inequal_constraints"))):
            inequal_viol_sum += max(0,eval("bf." + opt_func + ".pick_inequal_const(pos_part, m)"))
        # Calc equal constraints sum
        for m in range(len(eval("bf." + opt_func + ".equal_constraints"))):
            equal_viol_sum += abs(eval("bf." + opt_func + ".pick_equal_const(pos_part, m)"))

        return equal_viol_sum + inequal_viol_sum

    def saturation_strategy(pos_part):
        """Saturation strategy function"""

        for k in range(len(pos_part)):
            # Pick bounds from benchmark function
            bound = np.array(eval("bf." + opt_func + ".bounds")[k])
            if len(bound) != 0:
                if pos_part[k] < bound[0]:
                    pos_part[k] = bound[0]  # Correct lower bound
                if pos_part[k] > bound[1]:
                    pos_part[k] = bound[1]  # Correct upper bound

        return pos_part

    def update_parameters(pbest_pos, gbest_pos, iter_num):
        """Update the w, c1 and c2 factor depending by beta distance"""

        # Calc the beta
        beta = distance.euclidean(gbest_pos, pbest_pos)
        # Update velocity
        w = (w_s-w_f) * math.exp((-gamma_w*iter_num)/beta) + w_f
        # Update cognitive factor 
        c1 = (c1_s-c1_f) * math.exp((-gamma_c1*iter_num)/beta) + c1_f
        # Update social factor
        c2 = (c2_s-c2_f) * math.exp((gamma_c2*iter_num)/beta) + c2_f

        return [w, c1, c2]

    # Parameters definition
    space_dim = eval("bf." + opt_func + ".dimension")
    pop_size = 100
    iteration = 4000
    w_s = 0.9  # Initial inertial weight
    w_f = 0.1  # Final inertial weight
    c1_s = 2.5  # Initial cognitive factor
    c1_f = 0.1  # Final cognitive factor
    c2_s = 0.1  # Initial social factor
    c2_f = 2.5  # Final social factor

    # Compute the gamma factor for each parameters
    gamma_w = (w_s - w_f) / iteration
    gamma_c1 = (c1_s - c1_f) / iteration
    gamma_c2 = (c2_s - c2_f) / iteration

    gbest_iter = np.zeros(0)
    swarm_pos_iter = []

    # Generate an initial population without constraints
    swarm = init_pop_nc(pop_size)

    pbar = tqdm(total=iteration)
    pbar.set_description("PSO SAPSO 2011")
    
    # Compute initial relaxation value as median of each particles violation
    initial_violation = np.zeros(0)
    for obj in swarm:
        # Set initial parameters
        obj.w, obj.c1, obj.c2 = w_s, c1_s, c2_s
        obj.pbest_viol = violation_function(obj.pos)
        initial_violation = np.append(initial_violation, obj.pbest_viol)
    sigma = np.median(initial_violation)

    # Set violation of each pbest
    for obj in swarm:
        if obj.pbest_viol <= sigma:
            obj.pbest_feasible = True
        else:
            obj.pbest_feasible = False

    # Compute first gbest
    gbest = min(swarm, key=attrgetter('pbest')).pbest
    gbest_pos = min(swarm, key=attrgetter('pbest')).pbest_pos
    gbest_viol = min(swarm, key=attrgetter('pbest')).pbest_viol
    gbest_feasible = min(swarm, key=attrgetter('pbest')).pbest_feasible

    for l in range(1, iteration+1, 1):
        # Set 0 the feasible solution
        Ff = 0
        # Append swarm list for animation graph
        swarm_pos_iter.append([bird.pos for bird in swarm])

        for obj in swarm:
            # Compute the baricenter
            bar = obj.pos + ((obj.c1*(obj.pbest_pos-obj.pos) + obj.c2*(gbest_pos-obj.pos)) / 3)
            if all(obj.pbest_pos == gbest_pos):
                bar = (obj.pos + obj.pos + obj.c2*(gbest_pos-obj.pos))/2

            radius = distance.euclidean(bar, obj.pos)
            # Pick random point
            X = point_hypersphere(space_dim, bar, random.uniform(0, radius))

            # Update velocity and position
            obj.Vid = obj.w*obj.Vid + X - obj.pos
            obj.pos = obj.Vid + obj.pos

            # Applied the saturation strategy
            obj.pos = saturation_strategy(obj.pos)
            # Compute the violation value
            violation_sum = violation_function(obj.pos)

            # Update the number of feasible solution
            if violation_sum <= sigma:
                Ff += 1

            # Compute the value of the function
            fitting_val = eval("bf." + opt_func + ".function(obj.pos)") + violation_sum
            
            # Feasibility-based rule for pbest updating

            # Case where current solution is feasible and previous one not
            if (
                obj.pbest_feasible == False and
                violation_sum <= sigma
            ):
                obj.pbest_viol = violation_sum
                obj.pbest = fitting_val
                obj.pbest_pos = obj.pos
                obj.pbest_feasible = True


            # Case where previous and current solution are both unfeasible
            elif (
                obj.pbest_feasible == True and
                violation_sum <= sigma and
                fitting_val < obj.pbest
            ):
                obj.pbest_viol = violation_sum
                obj.pbest = fitting_val
                obj.pbest_pos = obj.pos
                obj.pbest_feasible = True


            # Case where previous and current solution are both unfeasible
            elif (
                obj.pbest_feasible == False and
                violation_sum > sigma and
                violation_sum < obj.pbest_viol
            ):
                obj.pbest_viol = violation_sum
                obj.pbest = fitting_val
                obj.pbest_pos = obj.pos
                obj.pbest_feasible = False

        # Feasibility-based rule for gbest updating


        # Case where current solution is feasible and previous one not
        if (
            gbest_feasible == False and
            min(swarm, key=attrgetter('pbest')).pbest_feasible == True
        ):
            gbest_viol = min(swarm, key=attrgetter('pbest')).pbest_viol
            gbest = min(swarm, key=attrgetter('pbest')).pbest
            gbest_pos = min(swarm, key=attrgetter('pbest')).pbest_pos
            gbest_feasible = True


        # Case where previous and current solution are both feasible
        elif (
            gbest_feasible == True and
            min(swarm, key=attrgetter('pbest')).pbest_feasible == True and
            min(swarm, key=attrgetter('pbest')).pbest < gbest
        ):
            gbest_viol = min(swarm, key=attrgetter('pbest')).pbest_viol
            gbest = min(swarm, key=attrgetter('pbest')).pbest
            gbest_pos = min(swarm, key=attrgetter('pbest')).pbest_pos
            gbest_feasible = True


        # Case where previous and current solution are both unfeasible
        elif (
            gbest_feasible == False and
            min(swarm, key=attrgetter('pbest')).pbest_feasible == False and
            min(swarm, key=attrgetter('pbest')).pbest_viol < gbest_viol
        ):
            gbest_viol = min(swarm, key=attrgetter('pbest')).pbest_viol
            gbest = min(swarm, key=attrgetter('pbest')).pbest
            gbest_pos = min(swarm, key=attrgetter('pbest')).pbest_pos
            gbest_feasible = False


# Update the algorithms parameters
        for obj in swarm:
            if np.array_equal(gbest_pos,obj.pbest_pos) == False:
                obj.w, obj.c1, obj.c2 = update_parameters(gbest_pos, obj.pbest_pos, l)
        
        # Decrease the relaxation value
        sigma = sigma*(1 - Ff/pop_size)

        gbest_iter = np.append(gbest_iter, gbest)
        pbar.update(1)

    print("--> gbest_viol:", gbest_viol)

    pbar.close()

    return gbest, gbest_pos, gbest_iter, swarm_pos_iter


def pyswarm_library():
    """PSO using Pysmwarm library"""

    # PARAMETERS DEFINITION
    pop_size = 500
    iteration = 1000
    w = 0.5
    c1 = 0.5
    c2 = 0.5
    minstep = 1e-3
    minfunc = 1e-3
    debug = False

    # Default parameters
    print("Use default parameters:\n"
          "\t\t\tpop_size =", pop_size, "\n",
          "\t\t\titeration =", iteration, "\n",
          "\t\t\tw =", w, "\n",
          "\t\t\tc1 =", c1, "\n",
          "\t\t\tc2 =", c2, "\n",
          "\t\t\tminstep =", minstep, "\n",
          "\t\t\tminfunc =", minfunc, "\n",
          "\t\t\tdebug =", debug, "\n\n",
          "or set personal ones? [D for default|P for personal]")

    # Set personal parameters
    param = input()
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
        print(
            "The minimum stepsize of swarm’s best position before the search terminates = [int]")
        minstep = eval(input())
        print(
            "Minimum change of swarm’s best objective value before the search terminates = [int]")
        minfunc = eval(input())
        print("Active debyg = [boolean]")
        debug = eval(input()) 
 
    def constraints(x):
        """Pick the constraints from benchmarck_function file"""
        for k in range(len(x)):
            locals()[f"x{k+1}"] = x[k]
        constraints = eval("bf." + opt_func + ".inequal_constraints")
        positive_constraints = []
        for k in range(len(constraints)):
            # Library use >= notation
            positive_constraints.extend([eval(constraints[k])*-1])
        return positive_constraints 

    lb = []
    ub = []
    # Pick the bounds of the fitting function
    for k in range(len(eval("bf." + opt_func + ".bounds"))):
        lb.extend([eval("bf." + opt_func + ".bounds")[k][0]])
        ub.extend([eval("bf." + opt_func + ".bounds")[k][1]])

    # Compute gbest with library function
    start_time = time.time()
    gbest_pos, gbest = pso(eval("bf." + opt_func + ".function"), lb, ub, f_ieqcons=constraints, swarmsize=pop_size,
        omega=w, phip=c1, phig=c2, maxiter=iteration, minstep=minstep, minfunc=minfunc, debug=debug)

    end_time = time.time()
    print("ELAPSED TIME:", f"{end_time-start_time:.0f}", "sec for", runs, "runs")

    return gbest, gbest_pos

# GRAPH SECTION
def graph_gbest(data_graph, run):
    """Plot the gbest behavoir along the iteration"""

    fig = plt.figure("gbest iteration progress")
    plt.plot(np.linspace(1,len(data_graph),len(data_graph)), data_graph, label=f"Runs {k}")
    plt.title("gbest iteration progress")
    plt.xlabel("Iteration")
    plt.ylabel("gbest value")
    plt.legend()

def graph_iteration(data_graph, gbest_pos, opt_func):
    """Animated graph to evaluate the particles movements"""

    def update(frame, data, gbest_pos, bounds, minimum=[], ):
        """Function for update snapshot"""

        x_data, y_data = np.zeros(0), np.zeros(0)
        # Plot particle in 2d space
        for k in range(len(data[frame])):
            x_data = np.append(x_data, data[frame][k][0])
            y_data = np.append(y_data, data[frame][k][1])
        storm_plot = ax.scatter(x_data, y_data, facecolors="none", edgecolors="b")
        
        gbest_plot = ax.scatter(gbest_pos[0], gbest_pos[1], s=80, c='r', marker='x', label="gbest") # Plot gbest position with X marker

        # Plot area of feasible space from bounds
        xmin, xmax, ymin, ymax = bounds
        bounds_space = patch.Rectangle((xmin,ymin), width=(xmax-xmin), height=(ymax-ymin),
            fill=True, alpha=0.1, linestyle='--', color='g', label="Bounds space")
        bounds_plot = ax.add_patch(bounds_space)

        if eval("bf." + opt_func + ".solution") != []:
            # Plot minimum position with * marker
            minimum_plot = ax.scatter(
                minimum[0], minimum[1], s=80, c='m', marker='*', label="Minimum")
            return storm_plot, gbest_plot, bounds_plot, minimum_plot,
        else:
            return storm_plot, gbest_plot, bounds_plot,  

    # Setting all graph parameters
    fig, ax = plt.subplots(figsize=(12,9))
    fig.canvas.set_window_title("Swarm iteration movements")
    ax.set_title("Swarm iteration movements")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    legend_patch = [Line2D([0], [0], marker='o', color='b', lw=0,  fillstyle="none", label="Particle"),
                    Line2D([0], [0], marker='x', color='r', lw=0, label="gbest"),
                    Line2D([0], [0], marker='*', color='m', lw=0, label="Minimum"),
                    patch.Patch(facecolor='g', alpha=0.1, linestyle='--', label="Bounds space")]
    # Set the legend
    fig.legend(handles=legend_patch)

    # Pick the bound from benchmark function file
    bounds = []
    if len(eval("bf." + opt_func + ".bounds")) != 0:
        for k in range(2):
            bounds.extend(eval("bf." + opt_func + ".bounds")[k])

    if eval("bf." + opt_func + ".solution") != []:
        # Pick the minimum from benchmark function file
        minimum = [eval("bf." + opt_func + ".solution")[0], eval("bf." + opt_func + ".solution")[1]]
        # Create animation of following snapshot
        anim = animation.FuncAnimation(fig, update, frames=len(data_graph), fargs=(data_graph, gbest_pos, bounds, minimum, ), interval=200, blit=True, repeat=False)
    else:
        # Create animation of following snapshot
        anim = animation.FuncAnimation(fig, update, frames=len(data_graph), fargs=(data_graph, gbest_pos, bounds, ), interval=200, blit=True, repeat=False)
    

if __name__ == '__main__':

    print("\nWhich PSO algorithm do you want use?")
    method = eval(input(
        "[1] = True/False penalty approach\n[2] = Penalty function approach\n[3] = SAPSO 2011\n[4] = Py_swarm python library\n"))
    use_opposite = bool(
        int(input("Do you want use an opposite population? [0|1]: ")))
    runs = eval(input("How many runs? [int]: "))
    opt_func = input("Which function do you want optimize: ")
    print()

    gbest_runs = np.zeros(0)
    gbest_pos_runs = []

    # Switch case to manage all algorithms
    for k in range (1, runs+1):
        print("RUN NUMBER:", k)
        if method == 1:
            gbest, gbest_pos, gbest_iter, swarm_pos_iter = pso1()
        elif method == 2:
            gbest, gbest_pos, gbest_iter, swarm_pos_iter = pso2()
        elif method == 3:
            gbest, gbest_pos, gbest_iter, swarm_pos_iter = pso3()
        elif method == 4:
            gbest, gbest_pos = pyswarm_library()

        # Call graph function
        if method != 4:
            graph_iteration(swarm_pos_iter, gbest_pos, opt_func)
            graph_gbest(gbest_iter, k) 

        gbest_runs = np.append(gbest_runs, gbest)
        gbest_pos_runs.append(gbest_pos)
    
        print()

    # Print results
    print()
    print("----------------- RESULT -----------------")
    # gbest in each runs
    print("--> TOTAL gbest", gbest_runs, "\n")
    # Mean gbest for all runs
    print("--> MEAN SOLUTION:", "\n\tgbest:", gbest_runs.mean(),"\n\twith STANDARD DEVIATION:", gbest_runs.std(), "\n\tand with ERROR:", gbest_runs.mean() - eval("bf." + opt_func + ".minimum"), "\n")
    # Best gbest from all runs
    print("--> BEST SOLUTION:", "\n\tgbest:", np.min(gbest_runs), "\n\twith ERROR:", np.min(gbest_runs) - eval("bf." + opt_func + ".minimum"),"\n\twith position:", gbest_pos_runs[gbest_runs.argmin()], "\n")
    # Worst gbest from all runs 
    print("--> WORST SOLUTION:", "\n\tgbest:", np.max(gbest_runs), "\n\twith ERROR:", np.max(gbest_runs) - eval("bf." + opt_func + ".minimum"),"\n\twith position:", gbest_pos_runs[gbest_runs.argmax()], "\n")
    print()

    # Show all graphs
    plt.show()
