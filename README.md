# Presentation
The slides used for the final pitch of the competition can be accessed via the following link:

https://docs.google.com/presentation/d/1ykRrBzCtyXQL5KJPWq4Q7lcP9V0t40od7l0uFsD5T3E/edit?usp=sharing

More information on the challenge and the pitch event can be found via the following link:

https://app.ekipa.de/challenges/deloitte-quantum/brief

# Quantum-Challenge
The Deloitte Quantum Climate Challenge 2022 aims to explore how the contribution of aviation to the anthropogenic climate change can be reduced by optimizing flight routes using hybrid
quantum-classical algorithms. The case-study includes a sample of multiple flights with different
flight paths and flight schedules. The flight routes are to be optimized in such a way that the
warming of the climate is minimal, taking into account all flights, while at the same time being
compatible with flight safety regulations. Different climate effects occur depending on fuel consumption, the geographic location, flight altitude, weather conditions, and flight times. To solve
this problem efficiently on a quantum computer, the problem was split into two separate combinatorial optimization problems: Finding a set of climate-optimized trajectories and disentangling the
resulting flight plan. The first problem is solved using Grover’s search algorithms, and the second
problem is solved using the filtering variational quantum eigensolver.

This repository is structured as follows: The utils folder contains code used in various calculations, which is outsourced for better readability. The data folder contains the data provided for the challenge and the results of the various experiment runs. The notebook folder contains the main body of work on this Challenge. It is divided into the following notebooks: 
 

- Preliminaries: Contains code to adapt the provided data for the appropriate problem formulations.
- Classical Optimization: Contains the two classical trajectory optimizations that will be used as benchmarks
- Trajectory Optimization: Contains the quantum trajectory optimization run on a 5-qubit simulator and a 5-qubit IBM processor
- Trajectory Visualization: Contains the visualization of the trajectories calculated above.
- Conflict Resolution: Contains the QUBO formulation of the conflict resolution problem from the computed trajectories, solved with Filtering-VQEs.
