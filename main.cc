// C++ headers
#include <iostream>
#include <cassert>

// Deal.II headers
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

// Project headers
#include <Utils/Utilities.h>
#include <Parameter.h>
#include <Driver.h>


using namespace dealii;
using namespace StructuralOptimization;



int main(int argc, char *argv[]) {

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);
  const unsigned int this_mpi_process(
      Utilities::MPI::this_mpi_process(mpi_communicator));
  ConditionalOStream pcout(std::cout, this_mpi_process == 0);

#ifdef DEBUG
  pcout << "DEBUG MODE" << std::endl;
#else
  pcout << "RELEASE MODE" << std::endl;
#endif

#ifdef STOP_USE_TRILINOS
  pcout << "TRILINOS used for LA" << std::endl;
#else
  pcout << "PETSC used for LA" << std::endl;
#endif

  pcout << "Running with " << Utilities::MPI::n_mpi_processes(mpi_communicator) << " processors" << std::endl;

  std::string parameter_file;

  if (argc < 2) {
    parameter_file = "";
    Parameter par(parameter_file);
    par.DeclareParameters();
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      par.OutputDefaultParameters();
    return 0;
  } else if (argc == 2)
    parameter_file = argv[1];
  else
    throw std::runtime_error ("Wrong number of parameters");

  Parameter par(parameter_file);
  par.DeclareParameters();
  par.ParseParameters();

  pcout << "We are solving a "
        << par.data.dim << "D "
        << par.data.bvp_type << " "
        << par.data.problem_type << " problem."
        << std::endl;

  if (par.data.dim == 2) {
    StructuralOptimization::Driver<2> drive(par);
    drive.Run();
  }

  if (par.data.dim == 3) {
    StructuralOptimization::Driver<3> drive(par);
    drive.Run();
  }
  return 0;

} // End of main
