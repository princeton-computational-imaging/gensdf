#include <igl/read_triangle_mesh.h>
#include <igl/centroid.h>
#include <igl/signed_distance.h>
#include <igl/random_points_on_mesh.h>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>

void recenter_mesh(
  Eigen::MatrixXd & V,
  Eigen::MatrixXi & F)
{
  Eigen::RowVector3d C; // centroid of V
  igl::centroid(V,F,C);
  V.rowwise() -= C;
}

void normalize_mesh_bbox(
  Eigen::MatrixXd & V)
{
  // Get bounding box diagonal length
  double bbd = (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();
  V =  V/bbd;
}

void compute_signed_distances(
  Eigen::MatrixXd & V,
  Eigen::MatrixXi & F,
  Eigen::MatrixXd & Q,
  Eigen::VectorXd & S,
  Eigen::MatrixXd & C)
{
  Eigen::VectorXi I;
  Eigen::MatrixXd N;
  igl::SignedDistanceType sign_type = igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER;
  igl::signed_distance(Q, V, F, sign_type, S, I, C, N);
}

void get_pc(
  const std::string & csv_file_name,
  Eigen::MatrixXd & pc)
{
  std::ifstream file(csv_file_name.c_str());
  std::string line, val;
  int row = 0;
  while (std::getline(file, line)) 
  {
    std::stringstream s(line);
    double value;
    int column = 0;
    Eigen::RowVector3d point;
    while (std::getline(s, val, ','))
    {
      point(column) = std::stod(val);
      column += 1;
    }
    pc.row(row) = point;
    row += 1;
  }
}

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

void write_matrix_to_csv(
  const std::string & filename,
  Eigen::MatrixXd & T)
{
  std::ofstream file(filename.c_str());
  file << T.format(CSVFormat);
}

int main(int argc, char * argv[])
{
  std::cout << "Reading the input mesh...\n";
  std::string mesh_file = argv[1]; // mesh file path
  Eigen::MatrixXd mesh_v; // mesh vertices
  Eigen::MatrixXi mesh_f; // mesh face
  igl::read_triangle_mesh(mesh_file, mesh_v, mesh_f);

  std::cout << "Recentering the input mesh...\n";
  recenter_mesh(mesh_v, mesh_f);

  std::cout << "Normalizing the input mesh...\n";
  normalize_mesh_bbox(mesh_v);

  std::cout << "Reading query points...\n";
 
  std::string query_file = argv[2];
  Eigen::MatrixXd query_points(50000,3);
  get_pc(query_file, query_points);
  

  std::cout << "Computing signed distances...\n";
  Eigen::VectorXd SDF_values;
  Eigen::MatrixXd C;
  compute_signed_distances(mesh_v, mesh_f, query_points, SDF_values, C);

  Eigen::MatrixXd SDFData = Eigen::MatrixXd::Zero(query_points.rows(), 4); // each row: (x, y, z, SDF value)
  
  // Top "num_samples_from_surface" rows: points on the surface
  //SDFData.leftCols(3) = query_points;

  // merge points and sdf values
  query_points.conservativeResize(query_points.rows(), 4);
  query_points.col(3) = SDF_values;

  SDFData.conservativeResize(query_points.rows(), 4);
  SDFData.bottomRows(query_points.rows()) = query_points;

  std::cout << "Saving results...\n";
  std::string save_data_dir = argv[3];
  if(!std::__fs::filesystem::is_directory(save_data_dir))
    std::__fs::filesystem::create_directory(save_data_dir);
  std::string sdf_file = save_data_dir + "/sdf_gt.csv";
  write_matrix_to_csv(sdf_file, SDFData);

  return 0;
}