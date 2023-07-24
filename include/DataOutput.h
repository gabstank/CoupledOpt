#pragma once

// C++ headers
#include <iostream>
#include <memory> // smart pointers
#include <sys/stat.h>

// Deal.II headers
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/base/data_out_base.h>

#include <deal.II/hp/dof_handler.h>
// Project headers
#include <Utilities.h>
#include <Parameter.h>

namespace StructuralOptimization {


  //data_out.add_data_vector(*(_out_data_dof_handler[iter]),
  //    _out_data[iter],
  //    _out_data_names[iter],
  //    _out_data_interpretation[iter]);

  /**
   * @brief Common interface class to pass data to dealii data_out.add_data_vector. This function 
   * specifically designed to handle output with different DoFHandler<dim>.
   * @tparam dim 
   * @tparam DoFHandlerType 
   */
  template<int dim, typename DoFHandlerType>
  class OutputDataHandler {
  public:
    /**
     * @brief Construct a new OutputDataHandler object.
     * 
     * @param dof_handler 
     * @param data 
     * @param names 
     * @param data_interpretation 
     */
    OutputDataHandler(DoFHandlerType &dof_handler,
                      Vector<float> &data,
                      std::vector<std::string> &names,
                      std::vector<DataComponentInterpretation::DataComponentInterpretation> &data_interpretation)
        : _out_dof_handler(dof_handler),
          _out_data(data),
          _out_data_name(names),
          _out_data_interpretation(data_interpretation) {
    }

    ~OutputDataHandler() = default;

    /**
     * @brief Function to add data to dealii data_out.add_data_vector.
     * 
     * @param data_out 
     */
    void AddDataVector(DataOut <dim, DoFHandlerType> &data_out) {
      if (_out_data.size() == _out_dof_handler.n_dofs()) {
        data_out.add_data_vector(_out_dof_handler,
                                 _out_data,
                                 _out_data_name,
                                 _out_data_interpretation);
      }
      else {
        data_out.add_data_vector(_out_data,
                                 _out_data_name);
      }
    }

    DoFHandlerType &_out_dof_handler;
    Vector<float> _out_data;
    std::vector<std::string> _out_data_name;
    std::vector<DataComponentInterpretation::DataComponentInterpretation> _out_data_interpretation;
  };


  template<int dim, typename TriaType, typename DoFHandlerType>
  class DataOutput {
  /**
   * @brief Interface class to handle Data output. This class collects all the data and writes the output at the end of each iteration.
   * 
   */
  public:
    /**
     * @brief Construct a new DataOutput object
     */
    DataOutput(Data &, MPI_Comm &, TimerOutput &, const std::string &, TriaType &);

    /**
     * @brief Destroy the DataOutput object
     */
    ~DataOutput();

    /**
     * @brief Function to push data and names which will be used to generate output.
     * 
     * @tparam VecType 
     * @param data 
     * @param names 
     * @param data_terpretation 
     * @param ptr_dof_handler 
     */
    template<typename VecType>
    void PushDataName(VecType &data, std::vector<std::string> &names,
                      std::vector<DataComponentInterpretation::DataComponentInterpretation> &data_terpretation,
                      DoFHandlerType *ptr_dof_handler);

    /**
     * @brief Function to write output. For time independent problem exclude the parameter.
     * @param cycle 
     */
    void WriteDataOutput(int cycle = -1);

    /**
     * @brief Function to check if path exists.
     * @param s 
     * @return true 
     * @return false 
     */
    bool IsPathExist(const std::string &s);

    /**
     * @brief Function to switch off output.
     */
    inline void SwitchOffOutput() { _output_on = false; }

  private:
    MPI_Comm &mpi_communicator;
    TimerOutput &compute_timer;
    const std::string &prb_name;
    const std::string &destination;
    std::string vtu_folder = "VTUs/";
    const std::string _prefix;
    TriaType &_tria;
    unsigned int _poly_degree;
    unsigned int this_mpi_process = 0;
    unsigned int n_mpi_processes = 0;
    std::vector<Vector < float>> _out_data;
    std::vector<std::vector<std::string>> _out_data_names;
    std::vector<std::vector<DataComponentInterpretation::DataComponentInterpretation>> _out_data_interpretation;
    std::vector<std::unique_ptr<OutputDataHandler<dim, DoFHandlerType>>> _out_helper_ptr;

    std::vector<std::pair<double, std::string> > _time_and_name_history;

    bool _output_on = true;

    /**
     * @brief Struct to generate filename.
     */
    struct Filename {
      /// Generates vtu file name
      static std::string get_filename_vtu(std::string analysis_name,
                                          unsigned int subdomain,
                                          unsigned int cycle,
                                          const unsigned int n_digits = 5) {
        std::ostringstream filename_vtu;
        filename_vtu
            << analysis_name
            << (std::to_string(dim) + "d")
            << "."
            << Utilities::int_to_string(subdomain, n_digits)
            << "."
            << Utilities::int_to_string(cycle, n_digits)
            << ".vtu";
        return filename_vtu.str();
      }

      /// Generates pvtu file name
      static std::string get_filename_pvtu(std::string analysis_name,
                                           unsigned int timestep,
                                           const unsigned int n_digits = 5) {
        std::ostringstream filename_vtu;
        filename_vtu
            << analysis_name
            << (std::to_string(dim) + "d")
            << "."
            << Utilities::int_to_string(timestep, n_digits)
            << ".pvtu";
        return filename_vtu.str();
      }

      /// Generates pvd file name
      static std::string get_filename_pvd(std::string analysis_name) {
        std::ostringstream filename_vtu;
        filename_vtu
            << analysis_name
            << (std::to_string(dim) + "d")
            << ".pvd";
        return filename_vtu.str();
      }
    } filename;

  }; // end of DataOutput class

  template<int dim, typename TriaType, typename DoFHandlerType>
  DataOutput<dim, TriaType, DoFHandlerType>::DataOutput(Data &data, MPI_Comm &mpi_communicator_,
                                                        TimerOutput &compute_timer_,
                                                        const std::string &prefix, TriaType &tria)
      : mpi_communicator(mpi_communicator_),
        compute_timer(compute_timer_),
        prb_name(data.analysis_name),
        destination(data.destination_path),
        _prefix(prefix),
        _tria(tria),
        _poly_degree(data.poly_degree) {
    this_mpi_process = (Utilities::MPI::this_mpi_process(mpi_communicator));
    n_mpi_processes = (Utilities::MPI::n_mpi_processes(mpi_communicator));

    if (this_mpi_process == 0) {
      bool exists = IsPathExist(destination);
      if (!exists)
        mkdir(destination.c_str(), ACCESSPERMS);

      // This is to create a folder for all the help vtu files,
      // This is make it easy by reducing the number of files visible in paraview.

      exists = IsPathExist((destination + vtu_folder));
      if (!exists)
        mkdir((destination + vtu_folder).c_str(), ACCESSPERMS);

    }

    int ierr = MPI_Barrier(mpi_communicator);
    AssertThrowMPI(ierr);

  }

  template<int dim, typename TriaType, typename DoFHandlerType>
  DataOutput<dim, TriaType, DoFHandlerType>::~DataOutput() {
    _out_data.clear();
    _out_data_names.clear();
    _out_data_interpretation.clear();
    _out_helper_ptr.clear();
  }

  template<int dim, typename TriaType, typename DoFHandlerType>
  template<typename VecType>
  void DataOutput<dim, TriaType, DoFHandlerType>::PushDataName(VecType &data, std::vector<std::string> &names,
                                                               std::vector<DataComponentInterpretation::DataComponentInterpretation> &data_interpretation,
                                                               DoFHandlerType *ptr_dof_handler) {
    TimerOutput::Scope t(compute_timer, "DataOutput");

    if (!_output_on)
      return;

    Vector<float> vec(data);
    if(vec.size() == 0)
      throw std::runtime_error("You tried to push an empty vector to output: Name = " + names[0]);

    _out_helper_ptr
        .emplace_back(new OutputDataHandler<dim, DoFHandlerType>(*ptr_dof_handler, vec, names, data_interpretation));
    _out_data.push_back(vec);
    _out_data_names.push_back(names);
    _out_data_interpretation.push_back(data_interpretation);
  }

// Below code implemented based on
// https://www.systutorials.com/how-to-test-a-file-or-directory-exists-in-c/
  template<int dim, typename TriaType, typename DoFHandlerType>
  bool DataOutput<dim, TriaType, DoFHandlerType>::IsPathExist(const std::string &s) {
    struct stat buffer;
    return (stat(s.c_str(), &buffer) == 0);
  }

  template<int dim, typename TriaType, typename DoFHandlerType>
  void DataOutput<dim, TriaType, DoFHandlerType>::WriteDataOutput(int cycle) {
    if (!_output_on)
      return;

    TimerOutput::Scope t(compute_timer, "DataOutput");

    DataOut <dim, DoFHandlerType> data_out;
    data_out.attach_triangulation(_tria);

    Vector<float> subdomain(_tria.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = _tria.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    Assert(_out_data.size() == _out_data_names.size(),
           ExcMessage(
               "Output data and names size mismatch, there are : \n"
               + Utilities::int_to_string(_out_data.size())
               + " data vectors and "
               + Utilities::int_to_string(_out_data_names.size())
               + " data names!"));

    // Loop over all the output data accumulated.
    for (auto &out : _out_helper_ptr)
      out->AddDataVector(data_out);

    data_out.build_patches();
    //data_out.build_patches(_poly_degree);

    std::string filename_vtu;
    if (cycle != -1)
      filename_vtu = filename.get_filename_vtu(destination + vtu_folder + prb_name + "-" + _prefix + "-",
                                               this_mpi_process,
                                               cycle);
    else
      filename_vtu = filename.get_filename_vtu(destination + vtu_folder + prb_name + "-" + _prefix + "-",
                                               this_mpi_process,
                                               0);
    std::ofstream output(filename_vtu.c_str());
    data_out.write_vtu(output);


    // At this point, all processors have written their own files to disk. We
    // could visualize them individually in Visit or Paraview, but in reality
    // we of course want to visualize the whole set of files at once. To this
    // end, we create a master file in each of the formats understood by Visit
    // (<code>.visit</code>) and Paraview (<code>.pvtu</code>) on the zeroth
    // processor that describes how the individual files are defining the
    // global data set.
    if (this->this_mpi_process == 0) {
      std::vector<std::string> filenames;
      std::string pvtu_master_filename;
      if (cycle != -1) {

        for (unsigned int i = 0; i < n_mpi_processes; ++i)
          filenames.push_back(filename.get_filename_vtu(prb_name + "-" + _prefix + "-",
                                                        i,
                                                        cycle).c_str());
        pvtu_master_filename =
            (filename.get_filename_pvtu(destination + vtu_folder + prb_name + "-" + _prefix + "-", cycle));
      } else {

        for (unsigned int i = 0; i < n_mpi_processes; ++i)
          filenames.push_back(filename.get_filename_vtu("./" + vtu_folder + prb_name + "-" + _prefix + "-",
                                                        i,
                                                        0).c_str());
        pvtu_master_filename =
            (filename.get_filename_pvtu(destination + prb_name + "-" + _prefix + "-", 0));
      }
      std::ofstream pvtu_master(pvtu_master_filename.c_str());
      data_out.write_pvtu_record(pvtu_master, filenames);

      // If cycle is not -1, then it is a time dependent problem and so we write the pvd file.
      if (cycle != -1) {
        // Time dependent data master file
//      static std::vector<std::pair<double, std::string> > time_and_name_history;
        _time_and_name_history.push_back(std::make_pair(cycle,
                                                        (filename.get_filename_pvtu(
                                                            "./" + vtu_folder + prb_name + "-" + _prefix +
                                                            "-", cycle))));
        const std::string filename_pvd(
            filename.get_filename_pvd(destination + prb_name + "-" + _prefix + "-"));
        std::ofstream pvd_output(filename_pvd.c_str());
        DataOutBase::write_pvd_record(pvd_output, _time_and_name_history);
      }
    }// end of if this_mpi_process==0

    // In time dependent problem, the constructor is called once
    // but the write function can be called multiple times, so we have
    // to manually clear this data.
    _out_data.clear();
    _out_data_names.clear();
    _out_data_interpretation.clear();
    _out_helper_ptr.clear();
  }

} // end of StructuralOptimization namespace
