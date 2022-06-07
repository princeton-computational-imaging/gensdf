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

  int num_samples = 500000;
  int num_samples_from_surface = (int)(47 * num_samples / 100);
  int num_samples_near_surface = num_samples - num_samples_from_surface;

  std::cout << "Sampling points...\n";
  Eigen::MatrixXd B, pc; // B: barycentric coord, pc: point cloud
  Eigen::MatrixXi FI; // face index
  igl::random_points_on_mesh(num_samples_from_surface, mesh_v, mesh_f, B, FI, pc);

  // Noise vectors
  float variance = 0.005;
  float second_variance = variance / 10;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::normal_distribution<float> perturb_norm1(0, sqrt(variance));
  std::normal_distribution<float> perturb_norm2(0, sqrt(second_variance));

  std::cout << "Sampling query points...\n";
  Eigen::MatrixXd query_points(num_samples_from_surface * 2, 3); // query points
  for(int i = 0; i <num_samples_from_surface; i++)
  {
    query_points.row(i) = pc.row(i) + perturb_norm1(rng) * pc.row(i); // first query
    query_points.row(num_samples_from_surface+i) = pc.row(i) + perturb_norm2(rng) * pc.row(i); // second query
  }

  std::cout << "Computing signed distances...\n";
  Eigen::VectorXd SDF_values;
  Eigen::MatrixXd C;
  compute_signed_distances(mesh_v, mesh_f, query_points, SDF_values, C);

  Eigen::MatrixXd SDFData = Eigen::MatrixXd::Zero(pc.rows(), 4); // each row: (x, y, z, SDF value)
  
  // Top "num_samples_from_surface" rows: points on the surface
  SDFData.leftCols(3) = pc;

  // merge points and sdf values
  query_points.conservativeResize(query_points.rows(), 4);
  query_points.col(3) = SDF_values;

  // Bottom "query_points.rows()" rows: points near the surface
  SDFData.conservativeResize(SDFData.rows() + query_points.rows(), 4);
  SDFData.bottomRows(query_points.rows()) = query_points;

  std::cout << "Saving results...\n";
  std::string save_data_dir = argv[2];
  if(!std::__fs::filesystem::is_directory(save_data_dir))
    std::__fs::filesystem::create_directory(save_data_dir);
  std::string sdf_file = save_data_dir + "/sdf_data.csv";
  write_matrix_to_csv(sdf_file, SDFData);

  return 0;
}