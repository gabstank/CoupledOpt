#include <Utils/Utilities.h>

namespace StructuralOptimization {

  template<int dim>
  const SymmetricTensor<2, dim> StandardTensors<dim>::I = unit_symmetric_tensor<dim>();

  template<int dim>
  const SymmetricTensor<4, dim> StandardTensors<dim>::IxI() { return outer_product(I, I); }

  template<int dim>
  const SymmetricTensor<4, dim> StandardTensors<dim>::II() { return identity_tensor<dim>(); }

// string to vector of int
  void string_to_vector_of_int(const std::string &s,
                               std::vector<int> &int_vec) {
    int_vec.clear();
    std::string item;
    std::stringstream ss(s);

    while (std::getline(ss, item, ',')) {
      int_vec.push_back(std::stoi(item));
    }
  }

// string to vector of double
  void string_to_vector_of_double(const std::string &s,
                                  std::vector<double> &double_vec) {
    double_vec.clear();
    std::string item;
    std::stringstream ss(s);

    while (std::getline(ss, item, ',')) {
      double_vec.push_back(std::stof(item));
    }
  }

// string to vector of double
  void string_to_vector_of_strings(const std::string &s,
                                   std::vector<std::string> &string_vec) {
    string_vec.clear();
    std::string item;
    std::stringstream ss(s);

    while (std::getline(ss, item, ',')) {
      boost::algorithm::trim(item);
      string_vec.push_back(item);
    }
  }

  std::string center(const std::string s, const int w) {
    std::stringstream ss, spaces;
    int pad = w - s.size();                  // count excess room to pad
    for (int i = 0; i < pad / 2; ++i)
      spaces << " ";
    ss << spaces.str() << s << spaces.str(); // format with padding
    if (pad > 0 && pad % 2 != 0)                    // if pad odd #, add 1 more space
      ss << " ";
    return ss.str();
  }

  namespace DebugUtilities {

    template<typename VecType>
    void PrintVector(const VecType &vec, const std::string &vec_name) {
      std::cout << vec_name << ": " << std::endl;
      for (unsigned int i = 0; i < vec.size(); ++i)
        std::cout << vec[i] << " | ";
      std::cout << std::endl;
      std::cout << std::endl;
    }

    void PrintVectorDouble(const Vector<double> &vec, const std::string &vec_name) {
      std::cout << vec_name << ": " << std::endl;
      for (unsigned int i = 0; i < vec.size(); ++i)
        std::cout << vec[i] << " | ";
      std::cout << std::endl;
      std::cout << std::endl;
    }

  } // end of namespace DebugUtilities

  template<>
  void WriteAbaqusScript<2>(const DoFHandler<1, 2> &dof_handler_shape, std::string dest_path, std::string analysis_name) {
    Point<2> vertex_a, vertex_b;

    std::ostringstream filename;
    filename << dest_path << analysis_name << "_restart_abq_script.py";
    std::ofstream output(filename.str().c_str());

    output << "from abaqus import *" << std::endl;
    output << "from abaqusConstants import *" << std::endl;
    output << "import __main__" << std::endl;

    output << "import section" << std::endl;
    output << "import regionToolset" << std::endl;
    output << "import displayGroupMdbToolset as dgm" << std::endl;
    output << "import material" << std::endl;
    output << "import assembly" << std::endl;
    output << "import step" << std::endl;
    output << "import interaction" << std::endl;
    output << "import load" << std::endl;
    output << "import sketch" << std::endl;
    output << "import visualization" << std::endl;
    output << "import xyPlot" << std::endl;
    output << "import displayGroupOdbToolset as dgo" << std::endl;
    output << "import connectorBehavior" << std::endl;

    output << "session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, engineeringFeatures=ON, mesh=OFF)" << std::endl;
    output << "session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues( meshTechnique=OFF)" << std::endl;
    output << "session.viewports['Viewport: 1'].setValues(displayedObject=None) " << std::endl;
    output << "mdb.models['Model-1'].Material(name='Material-1') " << std::endl;
    output << "session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, engineeringFeatures=OFF) " << std::endl;
    output << "session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=ON) " << std::endl;
    output << "s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0) " << std::endl;
    output << "g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints " << std::endl;
    output << "s.setPrimaryObject(option=STANDALONE) " << std::endl;

    std::string mid_points;
    mid_points.append("pickedEdges = e.findAt( ");
    for(const auto &cell : dof_handler_shape.active_cell_iterators()){
      vertex_a = cell->vertex(0);
      vertex_b = cell->vertex(1);
      output << "s.Line(point1=(";
      output << vertex_a[0] << ", " << vertex_a[1] << "), point2=(";
      output << vertex_b[0] << ", " << vertex_b[1] << "))" << std::endl;
      mid_points.append("((");
      mid_points.append(std::to_string((vertex_a[0] + vertex_b[0])*0.5));
      mid_points.append(",");
      mid_points.append(std::to_string((vertex_a[1] + vertex_b[1])*0.5));
      mid_points.append(", 0.0), ),");
    }

    mid_points.erase(mid_points.end()-1);
    mid_points.append(" )");

    output << "p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)" << std::endl;
    output << "p = mdb.models['Model-1'].parts['Part-1']" << std::endl;
    output << "p.BaseShell(sketch=s)" << std::endl;
    output << "s.unsetPrimaryObject()" << std::endl;
    output << "p = mdb.models['Model-1'].parts['Part-1']" << std::endl;
    output << "session.viewports['Viewport: 1'].setValues(displayedObject=p)" << std::endl;
    output << "del mdb.models['Model-1'].sketches['__profile__']" << std::endl;
    output << "session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, engineeringFeatures=ON)" << std::endl;
    output << "session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=OFF)" << std::endl;
    output << "mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', material='Material-1', thickness=1.0) " << std::endl;
    output << "p = mdb.models['Model-1'].parts['Part-1']" << std::endl;
    output << "f = p.faces" << std::endl;

    output << "faces = f.getSequenceFromMask(mask=('[#1 ]', ), ) " << std::endl;
    output << "region = regionToolset.Region(faces=faces) " << std::endl;
    output << "p = mdb.models['Model-1'].parts['Part-1'] " << std::endl;
    output << "p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)" << std::endl;

    output << "session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(meshTechnique=ON)" << std::endl;
    output << "p = mdb.models['Model-1'].parts['Part-1']" << std::endl;
    output << "f = p.faces" << std::endl;
    output << "pickedRegions = f.getSequenceFromMask(mask=('[#1 ]', ), )" << std::endl;
    output << "p.setMeshControls(regions=pickedRegions, elemShape=QUAD)" << std::endl;

    output << "p = mdb.models['Model-1'].parts['Part-1']" << std::endl;
    output << "e = p.edges" << std::endl;
    output << mid_points << std::endl;
    output << "p.seedEdgeByNumber(edges=pickedEdges, number=1, constraint=FINER)" << std::endl;

    output << "p = mdb.models['Model-1'].parts['Part-1']" << std::endl;
    output << "p.generateMesh()" << std::endl;
    output << "a = mdb.models['Model-1'].rootAssembly" << std::endl;
    output << "session.viewports['Viewport: 1'].setValues(displayedObject=a)" << std::endl;
    output << "a = mdb.models['Model-1'].rootAssembly" << std::endl;
    output << "a.DatumCsysByDefault(CARTESIAN)" << std::endl;
    output << "p = mdb.models['Model-1'].parts['Part-1']" << std::endl;
    output << "a.Instance(name='Part-1-1', part=p, dependent=ON)" << std::endl;
    std::string mod_analysis_name = analysis_name + "-restart";
    mod_analysis_name.erase(std::remove(mod_analysis_name.begin(), mod_analysis_name.end(),'.'), mod_analysis_name.end()); // abquas cannot run script if name contains period.
    output << "mdb.Job(name='" << mod_analysis_name <<
           "', model='Model-1', description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, \
      memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, \
      modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', parallelizationMethodExplicit=DOMAIN, numDomains=1, \
      activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=1)" << std::endl;
    output << "mdb.jobs['" << mod_analysis_name <<"'].writeInput(consistencyChecking=OFF)" << std::endl;

    output.close();
  }

  template<>
  void WriteAbaqusScript<3>(const DoFHandler<2, 3> &dof_handler_shape, std::string dest_path, std::string analysis_name) {
    (void) dof_handler_shape, (void) dest_path, (void) analysis_name;
    //TODO: not implemented for 3D.
  }


} // end of namespace StructuralOptimization


template
class StructuralOptimization::StandardTensors<2>;

template
class StructuralOptimization::StandardTensors<3>;

