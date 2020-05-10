/*


Optimizing ICa,t+IK,p+IL-model benchmark with monotonic steady-state current using HillVallEA 
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
class BenchmarkLinearCatKLt : public fitness_t
{
	public:

	BenchmarkLinearCatKLt()
	{
	  number_of_parameters = 12;
	  maximum_number_of_evaluations = 100000;
	}
	~BenchmarkLinearCatKLt() {}

	void get_param_bounds(vec_t & lower, vec_t & upper) const
	{
	  lower.resize(number_of_parameters, 0);
	  upper.resize(number_of_parameters, 0);
	  
	  lower[0]  = 0.1;
	  lower[1]  = 0.1;
	  lower[2]  = 0.1;
	  lower[3]  = 20;
	  lower[4]  = -100;
	  lower[5]  = -90;
	  lower[6]  = -90;
	  lower[7]  = -90;
	  lower[8]  = -90;
	  lower[9]  = 1;
	  lower[10] = -30;
	  lower[11] = 1;

	  upper[0]  = 50;
	  upper[1]  = 50;
	  upper[2]  = 50;
	  upper[3]  = 150;
	  upper[4]  = -2;
	  upper[5]  = 30;
	  upper[6]  = -2;
	  upper[7]  = -2;
	  upper[8]  = -2;
	  upper[9]  = 30;
	  upper[10] = -1;
	  upper[11] = 30;
	}

	double xinf(double V, double V12, double k)
	{
		double x;
		x = 1/(1+exp((V12-V)/k));	
		return x;
	}

	double Iinf(double V)
	{
		double gCa = 0.2; 
		double gK = 0.4; 
		double gL = 0.3;
		double ECa = 104; 
		double EK = -18.4;
		double EL = -63.1;
		double V12mCa = -23.8;
		double V12hCa = -37.2;
		double V12mK = -25.1;
		double kmCa = 18.4;
		double khCa = -27.8;
		double kmK = 16.3;

		double y;
		y = gCa*xinf(V,V12mCa,kmCa)*xinf(V,V12hCa,khCa)*(V-ECa)
		+ gK*xinf(V,V12mK,kmK)*(V-EK)
		+ gL*(V-EL);
		return y;
	}


	void define_problem_evaluation(solution_t & sol)
	{
  	  double* vecV = new double[18];
	  vecV[0]=-120; 
	  vecV[1]=-110; 
	  vecV[2]=-100;
	  vecV[3]=-90; 
	  vecV[4]=-80; 
	  vecV[5]=-70; 
 	  vecV[6]=-60; 
	  vecV[7]=-50; 
	  vecV[8]=-40; 
	  vecV[9]=-30; 
	  vecV[10]=-20; 
	  vecV[11]=-10; 
	  vecV[12]=0; 
	  vecV[13]=10; 
	  vecV[14]=20; 
	  vecV[15]=30; 
	  vecV[16]=40; 
	  vecV[17]=50; 

	  double* Inf = new double[18];
	  Inf[0]=Iinf(-120); 
  	  Inf[1]=Iinf(-110); 
	  Inf[2]=Iinf(-100); 
	  Inf[3]=Iinf(-90); 
	  Inf[4]=Iinf(-80); 
	  Inf[5]=Iinf(-70); 
	  Inf[6]=Iinf(-60); 
	  Inf[7]=Iinf(-50); 
	  Inf[8]=Iinf(-40); 
	  Inf[9]=Iinf(-30); 
	  Inf[10]=Iinf(-20); 
	  Inf[11]=Iinf(-10); 
	  Inf[12]=Iinf(0); 
	  Inf[13]=Iinf(10); 
	  Inf[14]=Iinf(20); 
	  Inf[15]=Iinf(30); 
	  Inf[16]=Iinf(40); 
	  Inf[17]=Iinf(50);
	  
//	  sol.f = (4.0-2.1*p0s + p0s*p0s/3.0) * p0s + sol.param[0]*sol.param[1] + (-4.0 + 4.0*p1s)*p1s;

	  sol.f = 0;	
	  for (int i=0; i<18; ++i) 
	  {
	    //sol.f = sol.f + (Inf[i] - (sol.param[0]*xinf(vecV[i],sol.param[6],sol.param[9])*xinf(vecV[i],sol.param[7],sol.param[10])*(vecV[i]-sol.param[3]) + sol.param[1]*xinf(vecV[i],sol.param[8],sol.param[11])*(vecV[i]-sol.param[4]) + sol.param[2]*(vecV[i]-sol.param[5])))^2;
		double power = pow(
			Inf[i] - 
			(
				sol.param[0]*xinf(vecV[i],sol.param[6],sol.param[9])
				* xinf(vecV[i],sol.param[7],sol.param[10])
				* (vecV[i]-sol.param[3]) 
				+ sol.param[1]
				* xinf(vecV[i],sol.param[8],sol.param[11])
				* (vecV[i]-sol.param[4]) 
				+ sol.param[2]
				* (vecV[i]-sol.param[5])
			)
			,2
		);	    

		sol.f += power;

	  }

	  sol.f /= 18.0;
	  
	  sol.penalty = 0.0;
	}

	std::string name() const { return "BenchmarkLinearCa,t+K+L"; }
};
}

	
// Main: Run the CEC2013 niching benchmark
//--------------------------------------------------------
int main(int argc, char **argv)
{
  
  // Problem definition
  // Note: define as minimization problem!
  //-----------------------------------------
  hillvallea::fitness_pt fitness_function = std::make_shared<hillvallea::BenchmarkLinearCatKLt>();
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
  std::string file_appendix = "benchmarklinearCatKLt_seed_"+std::to_string(random_seed); // can be used when multiple runs are outputted in the same directory
  
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
  std::cout << "Running HillVallEA on the Multimodal BenchmarkLinearCa,t+K+L" << std::endl;
  
  
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
