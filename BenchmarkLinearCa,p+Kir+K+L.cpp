/*


Optimizing RIM neuron of the C. elegans using HillVallEA 
By Juanlu J. Laredo
Code available at:

github.com/juanluck/HillVallEA

HillVallEA forked from:

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

*/

#include "HillVallEA/hillvallea.hpp"
#include "HillVallEA/fitness.h"
#include <cmath>
#include <string> 

namespace hillvallea
{
class multimodal_RIM_t : public fitness_t
{
	public:

	multimodal_RIM_t()
	{
	  number_of_parameters = 15;
	  maximum_number_of_evaluations = 10000;
	}
	~multimodal_RIM_t() {}

	void get_param_bounds(vec_t & lower, vec_t & upper) const
	{
	  lower.resize(number_of_parameters, 0);
	  upper.resize(number_of_parameters, 0);
	  
	  lower[0]  = 0.1;
	  lower[1]  = 0.1;
	  lower[2]  = 0.1;
	  lower[3]  = 0.1;
	  lower[4]  = 20;
	  lower[5]  = -100;
	  lower[6]  = -90;
	  lower[7]  = -90;
	  lower[8]  = -90;
	  lower[9]  = -90;
	  lower[10] = -90;
	  lower[11] = 1;
	  lower[12] = -30;
	  lower[13] = 1;
	  lower[14] = -30;

	  upper[0]  = 50;
	  upper[1]  = 50;
	  upper[2]  = 50;
	  upper[3]  = 50;
	  upper[4]  = 150;
	  upper[5]  = -2;
	  upper[6]  = 30;
	  upper[7]  = -2;
	  upper[8]  = -2;
	  upper[9]  = -2;
	  upper[10] = -2;
	  upper[11] = 30;
	  upper[12] = -1;
	  upper[13] = 30;
	  upper[14] = -1;
	}

	double xinf(double V, double V12, double k)
	{
		double x;
		x=1/(1+exp((V12-V)/k));	
		return x;
	}

		double Iinf(double V)
	{

		gCa=0.3; gKir=0.12; gK=0.86; gL=0.44;
		ECa=35.3; EK=-77.1; EL=-64.23;
		V12mCa=-29.75; V12hKir=-74.6; V12mK=-48.9; V12hK=-79.6;
		kmCa=29.1; kKir=-15.1; kmK=28.71; khK=-21;

		double y;
		y = gCa*xinf(V,V12mCa,kmCa)*xinf(VH,V12hCa,khCa)*(V-ECa)
		+ gKir*xinf(V,V12hKir,kKir)*(V-EK)
		+ gK*xinf(V,V12mK,kmK)*xinf(V,V12hK,khK)*(V-EK)
		+ gL*(V-EL);
		return y;
	}

	void define_problem_evaluation(solution_t & sol)
	{
  	  double vecV[16] = {-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50};
	  double Inf[16] = {Iinf(-100), Iinf(-90), Iinf(-80), Iinf(-70), Iinf(-60), Iinf(-50), Iinf(-40), Iinf(-30), Iinf(-20), Iinf(-10), Iinf(0), Iinf(10), Iinf(20), Iinf(30), Iinf(40), Iinf(50)};


	  sol.f = 0;
	  double e = 0;	
	  for (int i=0; i<16; ++i) 
	  {
	    
		e += pow(
			Inf[i]-
			(
				sol.param[0]*xinf(vecV[i],sol.param[7],sol.param[11])
				* (vecV[i]-sol.param[4]) + sol.param[1]
				* xinf(vecV[i],sol.param[8],sol.param[12])
				* (vecV[i]-sol.param[5]) + sol.param[2]
				* xinf(vecV[i],sol.param[9],sol.param[13])
				* xinf(vecV[i],sol.param[10],sol.param[14])
				* (vecV[i]-sol.param[5]) + sol.param[3]
				* (vecV[i]-sol.param[6])
			)
			,2 // Raise to the square
		);

	  }
	  sol.f = e/16.0;
	  
	  sol.penalty = 0.0;
	}

	std::string name() const { return "MultimodlRIM"; }
};
}

	
// Main: Run the CEC2013 niching benchmark
//--------------------------------------------------------
int main(int argc, char **argv)
{
  
  // Problem definition
  // Note: define as minimization problem!
  //-----------------------------------------
  hillvallea::fitness_pt fitness_function = std::make_shared<hillvallea::multimodal_RIM_t>();
  hillvallea::vec_t lower_range_bounds, upper_range_bounds;
  fitness_function->get_param_bounds(lower_range_bounds, upper_range_bounds);
  
  // HillVallEA Settings
  //-----------------------------------------
  // Type of local optimizer to be used.
  // 0 = AMaLGaM, 1 = AMaLGaM-Univariate, 20 = iAMaLGaM, 21 = iAMaLGaM-Univariate
  size_t local_optimizer_index = 1; // AMaLGaM-Univariate (1) is suggested
  
  int maximum_number_of_evaluations = 1000000; // maximum number of evaluations
  int maximum_number_of_seconds = 120; // maximum runtime in seconds
  
  // if the optimum is known, you can terminate HillVallEA if it found a solution
  // with fitness below the value_to_reach (vtr)
  double value_to_reach = 0;
  bool use_vtr = false;
  
  // random seed initialization for reproducibility
  int random_seed = 356770;
  
  // Output to test files
  bool write_generational_solutions = false;
  bool write_generational_statistics = true;
  std::string write_directory = "./";
  std::string file_appendix = "RIM_seed_"+std::to_string(random_seed); // can be used when multiple runs are outputted in the same directory
  
  // Initialization of HillVallEA
  //-----------------------------------------
  hillvallea::hillvallea_t opt(
     fitness_function,
     (int) local_optimizer_index,
     (int) fitness_function->number_of_parameters,
     lower_range_bounds,
     upper_range_bounds,
     lower_range_bounds,
     upper_range_bounds,
     maximum_number_of_evaluations,
     maximum_number_of_seconds,
     value_to_reach,
     use_vtr,
     random_seed,
     write_generational_solutions,
     write_generational_statistics,
     write_directory,
     file_appendix
  );

  // Running HillVallEA
  std::cout << "Running HillVallEA on the Multimodal RIM neuron" << std::endl;
  
  
  opt.run();
  
  std::cout << "HillVallEA finished" << std::endl;
  std::cout << "Generation statistics written to " << write_directory << "statistics" << file_appendix << ".dat" << std::endl;
  std::cout << "Elitist archive written to       " << write_directory << "elites" << file_appendix << ".dat" << std::endl;
  
  std::cout << "HillVallEA Obtained " << opt.elitist_archive.size() << " elites: " << std::endl;
  
  std::cout << "    Fitness      Penalty   Params" << std::endl;
  for(size_t i = 0; i < opt.elitist_archive.size(); ++i)
  {
    std::cout << std::setw(11) << std::scientific << std::setprecision(3) << opt.elitist_archive[i]->f << "  ";
    std::cout << std::setw(11) << std::scientific << std::setprecision(3) << opt.elitist_archive[i]->penalty << "  ";
    std::cout << std::setw(11) << std::scientific << std::setprecision(3) << opt.elitist_archive[i]->param << std::endl;
  }
  

  return(0);
}
