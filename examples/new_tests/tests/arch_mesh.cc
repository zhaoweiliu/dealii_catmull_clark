#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>

#include <fstream>
#include <iostream>
#include <cmath>

using namespace dealii;

/**
 * @class ShallowArchMapping
 * 将平面矩形网格映射为浅拱形
 */
template <int dim>
class ShallowArchMapping : public Function<dim>
{
public:
    ShallowArchMapping(double length, double arch_height, double thickness)
        : Function<dim>(dim)
        , L(length)
        , H0(arch_height)
        , h(thickness)
    {}

    virtual double value(const Point<dim> &p,
                         const unsigned int component) const override
    {
        if (component == 0)  // x-coordinate
        {
            return p[0];  // x不变
        }
        else if (component == 1)  // y-coordinate
        {
            const double x = p[0];
            const double y = p[1];
            
            // 拱形函数: 中心拱高为H0的余弦曲线
            double arch_profile = 0.0;
            if (dim == 2)
            {
                // 二维: 简单拱形
                arch_profile = (H0/2.0) * (1.0 - std::cos(2.0 * numbers::PI * x / L));
            }
            else if (dim == 3)
            {
                // 三维: 柱面拱形，在宽度方向保持平坦
                arch_profile = (H0/2.0) * (1.0 - std::cos(2.0 * numbers::PI * x / L));
            }
            
            // 映射: 从平面到拱形
            // 使用线性插值: 底部(y=0)不变，顶部(y=h)有最大拱高
            return y + arch_profile * (y / h);
        }
        else if (dim == 3 && component == 2)  // z-coordinate
        {
            return p[2];  // z不变
        }
        
        return 0.0;
    }

private:
    const double L;   // 长度
    const double H0;  // 初始拱高
    const double h;   // 厚度
};

/**
 * 生成浅拱形网格的主要函数
 */
template <int dim, int spacedim = dim>
void create_shallow_arch_mesh(
    Triangulation<dim, spacedim> &triangulation,
    const Point<spacedim> &bottom_left,
    const Point<spacedim> &top_right,
    const double arch_height,
    const std::vector<unsigned int> &n_subdivisions)
{
    // 1. 首先生成矩形网格
    GridGenerator::subdivided_hyper_rectangle(
        triangulation,
        n_subdivisions,
        bottom_left,
        top_right);
    
    // 2. 将网格变形为拱形
    ShallowArchMapping<spacedim> mapping(
        top_right[0] - bottom_left[0],  // 长度L
        arch_height,                    // 拱高H0
        top_right[1] - bottom_left[1]); // 厚度h
    
    // 创建临时向量存储顶点
    std::vector<Point<spacedim>> vertices = triangulation.get_vertices();
    
    // 应用映射
    for (auto &vertex : vertices)
    {
        Tensor<1, spacedim> displacement;
        for (unsigned int d = 0; d < spacedim; ++d)
        {
            displacement[d] = mapping.value(vertex, d) - vertex[d];
        }
        vertex += displacement;
    }
    
    // 更新顶点
    triangulation.set_vertices(vertices);
    
    // 3. 设置边界ID
    typename Triangulation<dim, spacedim>::face_iterator face;
    for (face = triangulation.begin_face(); face != triangulation.end_face(); ++face)
    {
        if (face->at_boundary())
        {
            const Point<spacedim> face_center = face->center();
            
            // 左边界 (x = 0): 固定约束
            if (std::abs(face_center[0] - bottom_left[0]) < 1e-6)
            {
                face->set_boundary_id(1);
            }
            // 右边界 (x = L): 固定约束
            else if (std::abs(face_center[0] - top_right[0]) < 1e-6)
            {
                face->set_boundary_id(2);
            }
            // 底部 (y = 0): 电极1
            else if (std::abs(face_center[1] - bottom_left[1]) < 1e-6)
            {
                face->set_boundary_id(3);
            }
            // 顶部 (y = h): 电极2
            else if (std::abs(face_center[1] - top_right[1]) < 1e-6)
            {
                face->boundary_id = 4;
            }
            // 前后面 (如果是3D)
            else if (dim == 3)
            {
                if (std::abs(face_center[2] - bottom_left[2]) < 1e-6)
                {
                    face->boundary_id = 5;  // 前面
                }
                else if (std::abs(face_center[2] - top_right[2]) < 1e-6)
                {
                    face->boundary_id = 6;  // 后面
                }
            }
        }
    }
}

/**
 * 生成具有预拉伸的拱形网格
 */
template <int dim, int spacedim = dim>
void create_prestressed_arch_mesh(
    Triangulation<dim, spacedim> &triangulation,
    const double length,
    const double width,
    const double thickness,
    const double arch_height,
    const double prestretch_ratio,
    const unsigned int n_subdivisions_x,
    const unsigned int n_subdivisions_y,
    const unsigned int n_subdivisions_z = 1)
{
    // 1. 创建初始平面网格
    Point<spacedim> bottom_left, top_right;
    
    if (dim == 2)
    {
        bottom_left = Point<spacedim>(0.0, 0.0);
        top_right = Point<spacedim>(length, thickness);
        
        std::vector<unsigned int> subdivisions = {n_subdivisions_x, n_subdivisions_y};
        create_shallow_arch_mesh<dim, spacedim>(
            triangulation, bottom_left, top_right, arch_height, subdivisions);
    }
    else if (dim == 3)
    {
        bottom_left = Point<spacedim>(0.0, 0.0, 0.0);
        top_right = Point<spacedim>(length, thickness, width);
        
        std::vector<unsigned int> subdivisions = {
            n_subdivisions_x, n_subdivisions_y, n_subdivisions_z};
        
        // 暂时不应用拱形映射
        GridGenerator::subdivided_hyper_rectangle(
            triangulation, subdivisions, bottom_left, top_right);
        
        // 然后应用三维拱形映射
        std::vector<Point<spacedim>> vertices = triangulation.get_vertices();
        
        for (auto &vertex : vertices)
        {
            const double x = vertex[0];
            const double y = vertex[1];
            const double z = vertex[2];
            
            // 拱形轮廓：余弦函数
            double arch_profile = (arch_height/2.0) * 
                (1.0 - std::cos(2.0 * numbers::PI * x / length));
            
            // 在y方向（厚度方向）应用拱形
            // 顶部(y=thickness)有最大拱高，底部(y=0)不变
            vertex[1] = y + arch_profile * (y / thickness);
        }
        
        triangulation.set_vertices(vertices);
    }
    
    // 2. 应用预拉伸
    if (prestretch_ratio > 1.0)
    {
        std::vector<Point<spacedim>> vertices = triangulation.get_vertices();
        
        for (auto &vertex : vertices)
        {
            // 在x方向应用预拉伸
            vertex[0] = vertex[0] * prestretch_ratio;
        }
        
        triangulation.set_vertices(vertices);
    }
    
    // 3. 应用边界标记
    typename Triangulation<dim, spacedim>::cell_iterator cell;
    for (cell = triangulation.begin(); cell != triangulation.end(); ++cell)
    {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if (cell->face(face)->at_boundary())
            {
                const Point<spacedim> face_center = cell->face(face)->center();
                
                // 左边界 (x = 0)
                if (std::abs(face_center[0] - 0.0) < 1e-6)
                {
                    cell->face(face)->set_boundary_id(0);  // 固定约束
                }
                // 右边界 (x = L * prestretch_ratio)
                else if (std::abs(face_center[0] - length * prestretch_ratio) < 1e-6)
                {
                    cell->face(face)->set_boundary_id(1);  // 位移约束
                }
                // 底部 (y = 0)
                else if (std::abs(face_center[1] - 0.0) < 1e-6)
                {
                    cell->face(face)->set_boundary_id(2);  // 电极1
                }
                // 顶部 (y = thickness)
                else if (std::abs(face_center[1] - thickness) < 1e-6)
                {
                    cell->face(face)->set_boundary_id(3);  // 电极2
                }
                // 3D情况：z方向的面
                else if (dim == 3)
                {
                    if (std::abs(face_center[2] - 0.0) < 1e-6)
                    {
                        cell->face(face)->set_boundary_id(4);  // 前面
                    }
                    else if (std::abs(face_center[2] - width) < 1e-6)
                    {
                        cell->face(face)->set_boundary_id(5);  // 后面
                    }
                }
            }
        }
    }
}

/**
 * 输出网格信息
 */
template <int dim, int spacedim>
void print_mesh_info(const Triangulation<dim, spacedim> &triangulation)
{
    std::cout << "\n=== 网格信息 ===" << std::endl;
    std::cout << "维度: " << dim << "D" << std::endl;
    std::cout << "单元数: " << triangulation.n_active_cells() << std::endl;
    std::cout << "顶点数: " << triangulation.n_vertices() << std::endl;
    
    // 统计边界ID
    std::set<types::boundary_id> boundary_ids;
    typename Triangulation<dim, spacedim>::cell_iterator cell;
    
    for (cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
    {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if (cell->face(face)->at_boundary())
            {
                boundary_ids.insert(cell->face(face)->boundary_id());
            }
        }
    }
    
    std::cout << "边界ID: ";
    for (auto id : boundary_ids)
    {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    // 计算拱形顶点坐标
    double min_y = 1e10, max_y = -1e10;
    double arch_center_y = 0.0;
    
    for (const auto &vertex : triangulation.get_vertices())
    {
        if (vertex[1] < min_y) min_y = vertex[1];
        if (vertex[1] > max_y) max_y = vertex[1];
        
        // 找到中心点 (x = L/2)
        if (std::abs(vertex[0] - 0.05) < 1e-3)  // 假设L=0.1m
        {
            arch_center_y = vertex[1];
        }
    }
    
    std::cout << "厚度范围: " << max_y - min_y << " m" << std::endl;
    std::cout << "中心拱高: " << arch_center_y << " m" << std::endl;
    std::cout << "================\n" << std::endl;
}

/**
 * 保存网格到不同格式
 */
template <int dim, int spacedim>
void save_mesh(const Triangulation<dim, spacedim> &triangulation, 
               const std::string &base_filename)
{
    // 保存为VTU格式 (用于ParaView可视化)
    std::string vtu_filename = base_filename + ".vtu";
    std::ofstream vtu_out(vtu_filename);
    GridOut grid_out;
    grid_out.write_vtu(triangulation, vtu_out);
    std::cout << "网格已保存为: " << vtu_filename << std::endl;
    
    // 保存为MSH格式 (用于Gmsh)
    std::string msh_filename = base_filename + ".msh";
    std::ofstream msh_out(msh_filename);
    grid_out.write_msh(triangulation, msh_out);
    std::cout << "网格已保存为: " << msh_filename << std::endl;
    
    // 保存为EPS格式 (2D图形)
    if (dim == 2)
    {
        std::string eps_filename = base_filename + ".eps";
        std::ofstream eps_out(eps_filename);
        grid_out.write_eps(triangulation, eps_out);
        std::cout << "网格已保存为: " << eps_filename << std::endl;
    }
}

/**
 * 生成示例网格
 */
void generate_example_meshes()
{
    // 示例1: 2D浅拱形，无预拉伸
    {
        std::cout << "\n=== 生成示例1: 2D浅拱形，无预拉伸 ===" << std::endl;
        
        Triangulation<2, 3> tria_2d;
        
        const double L = 0.1;        // 长度 100mm
        const double h = 0.001;      // 厚度 1mm
        const double H0 = 0.005;     // 拱高 5mm
        const double prestretch = 1.0;  // 无预拉伸
        
        create_prestressed_arch_mesh<2, 3>(
            tria_2d, L, 0.0, h, H0, prestretch, 20, 2);
        
        print_mesh_info(tria_2d);
        save_mesh(tria_2d, "shallow_arch_2d_no_prestretch");
    }
    
    // 示例2: 2D浅拱形，5%预拉伸
    {
        std::cout << "\n=== 生成示例2: 2D浅拱形，5%预拉伸 ===" << std::endl;
        
        Triangulation<2, 3> tria_2d_pre;
        
        const double L = 0.1;        // 长度 100mm
        const double h = 0.001;      // 厚度 1mm
        const double H0 = 0.005;     // 拱高 5mm
        const double prestretch = 1.05;  // 5%预拉伸
        
        create_prestressed_arch_mesh<2, 3>(
            tria_2d_pre, L, 0.0, h, H0, prestretch, 20, 2);
        
        print_mesh_info(tria_2d_pre);
        save_mesh(tria_2d_pre, "shallow_arch_2d_prestretch_5percent");
    }
    
    // 示例3: 3D浅拱形
    {
        std::cout << "\n=== 生成示例3: 3D浅拱形 ===" << std::endl;
        
        Triangulation<3, 3> tria_3d;
        
        const double L = 0.1;        // 长度 100mm
        const double W = 0.01;       // 宽度 10mm
        const double h = 0.001;      // 厚度 1mm
        const double H0 = 0.005;     // 拱高 5mm
        const double prestretch = 1.05;  // 5%预拉伸
        
        create_prestressed_arch_mesh<3, 3>(
            tria_3d, L, W, h, H0, prestretch, 20, 2, 5);
        
        print_mesh_info(tria_3d);
        save_mesh(tria_3d, "shallow_arch_3d");
    }
}

/**
 * 生成具有不同参数的网格用于参数研究
 */
void generate_parameter_study_meshes()
{
    std::cout << "\n=== 生成参数研究网格 ===" << std::endl;
    
    // 基本参数
    const double L = 0.1;      // 长度
    const double h = 0.001;    // 厚度
    const double prestretch = 1.05;  // 预拉伸
    
    // 不同拱高
    std::vector<double> arch_heights = {0.002, 0.005, 0.008, 0.012};  // 2, 5, 8, 12mm
    
    for (size_t i = 0; i < arch_heights.size(); ++i)
    {
        Triangulation<2, 3> tria;
        
        std::cout << "\n生成拱高 H0 = " << arch_heights[i]*1000 << " mm 的网格" << std::endl;
        
        create_prestressed_arch_mesh<2, 3>(
            tria, L, 0.0, h, arch_heights[i], prestretch, 20, 2);
        
        std::string filename = "arch_height_" + 
            std::to_string(static_cast<int>(arch_heights[i]*1000)) + "mm";
        
        save_mesh(tria, filename);
    }
}

/**
 * 主函数
 */
int main()
{
    try
    {
        // 生成示例网格
        generate_example_meshes();
        
        // 生成参数研究网格
        generate_parameter_study_meshes();
        
        std::cout << "\n所有网格已成功生成！" << std::endl;
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "运行过程中发生异常: " << std::endl
                  << exc.what() << std::endl
                  << "中止！" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "未知异常!" << std::endl
                  << "中止!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    
    return 0;
}