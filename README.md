# Simulating Spin Decoherence in Realistic Environments

This repository contains the code and results used in an undergraduate thesis paper, which presents a comprehensive study of spin decoherence in Vandadyl Tetraphenylporphyrinate [VO(TPP)], a material with promising potential in quantum computing applications. This research focuses on investigating the impact of nuclear and electronic spins in the bath on the decoherence of the central spin qudit. Through simulations of this quantum system using the Cluster Correlation Expansion method in Python (PyCCE) and combining bath contributions via the analytical product rule, this study challenges conventional assumptions by demonstrating that electronic spin interactions in the bath play a more dominant role in driving decoherence than nuclear magnetic spins. These findings contribute to the field of quantum computing materials and offer an opportunity for development of further materials with greater coherence time.

## Getting Started

Results are found in /VOTPP folder/Results and code can be found it VOTPP folder/  
A lot of the code is built with mpi4py. You will need to run code (e.g. like ```mpiexec -np NUMBER_OF_CORES python "VOTPP folder/VOTPP_[n-e]-(n).py```) where the number of cores is an integer which must be equal to or greater than the value of nbstates.  
Requirements can be accessed in requirements.txt.

### Sample Results
![Coherence Plot](https://github.com/aoneillmark/Capstone/blob/master/VOTPP%20folder/Results/Plots%202/combined_coherence_plot.png?raw=true)  
Simulation results with product rule visualisation. The envelope of the overall coherence curve is determined by the coherence curve of the electronic spin bath.  

![T2 vs B Plot](https://github.com/aoneillmark/Capstone/blob/master/VOTPP%20folder/Results/T2_vs_B/Combined_T2_Product_Rules.png?raw=true)  
𝑇_2 vs 𝐵_0 for many nuclear transitions, simulated with PyCCE and analysed with Product Rule.


### Authors

Author: Mark O'Neill  
Supervisors: Prof Alessandro Lunghi, Valerio Briganti

### Acknowledgements 

My sincerest thanks to Prof. Alessandro Lunghi and PhD candidate Valerio Briganti for their
invaluable guidance and generosity throughout the development of this thesis. Their insights
and support have been instrumental to this research. Additionally, I would like to thank all
members of Prof. Lunghi’s research group for their kindness, warmth, and openness. The
optimised structure of the [VO(TPP)] cell and the hyperfine interaction tensor were provided
by Valerio Briganti.  
I would like to thank Dr. Nikita Onizhuk for his expert guidance on PyCCE simulations. His
advice on pulse sequence implementation and Monte Carlo bath sampling theory was
instrumental in advancing this research.  
This work was supported by RIT (Research IT, Trinity College Dublin). Various calculations
were performed on the Boyle cluster maintained by the Trinity Centre for High Performance
Computing. This cluster was funded through grants from the European Research Council and
Science Foundation Ireland.  
In the preparation of this research, I acknowledge the use of Large Language Models (GitHub
Copilot etc.) for assistance in generating code related to data processing and plotting.