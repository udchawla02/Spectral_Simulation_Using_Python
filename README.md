Spectral Simulation using Python Instruction on how to run the code
						
Name: Udit Chawla 
Matriculation Number: 5771359
						
Structure of the project						
The main branch is the main branch, it has the most updated codes and the most updated results. And you can run all milestones from this branch.
Setting Up the Conda Environment
To ensure the results presented in this report can be accurately reproduced, an environment.yml file is included with the project. This file specifies the Conda environment configuration, including all necessary packages and their versions. The environment is particularly focused on libraries crucial for the project, such as mufft and mugrid.


The mufft library is essential for performing fast Fourier transforms, a key component in the spectral simulations detailed in the report. The mugrid library, on the other hand, provides tools for grid-based operations, which are integral to handling and processing the data within the simulations. Both libraries are included in the environment.yml file to ensure that all functionalities are available for reproducing the results.


To set up the environment, use the following command in your terminal or command prompt:


conda env create -f environment.yml


This command will create a Conda environment named dl2023-ex01-tensors-dl2023-rangers with all specified packages, including mufft and mugrid, along with other dependencies. After creating the environment, activate it with:
conda activate dl2023-ex01-tensors-dl2023-rangers
Activating this environment ensures that all the required libraries are correctly installed and configured, allowing you to run the code and achieve results that are consistent with those reported.
Once the environment is set up and activated, you can directly run the program. The results for Milestones 1 and 2 will be displayed in a separate window, allowing you to view them immediately. For Milestones 3 and 4, the output will be saved in their respective directories, making it easy to review the results later. This setup ensures a smooth workflow for running the simulations and accessing the results as described in the report.
		

