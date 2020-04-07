//
// Copyright (C) 2018 Marco Morandini
// Copyright (C) 2018 Wenguo Zhu
//
//----------------------------------------------------------------------
//
//    This file is part of Anba.
//
//    Anba is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    Hanba is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with Anba.  If not, see <https://www.gnu.org/licenses/>.
//
//----------------------------------------------------------------------
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <cmath>
#include <vector>
#include <memory>

#include <Eigen/Core>
#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

namespace anba {
class Material
{
    private:
        mutable Eigen::Matrix<double, 6, 6, Eigen::RowMajor> transformMatrix;
	const double rho;
    protected:
        mutable Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matModulus;
        mutable Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matRotatedStressModulus;
    public:
    // std::bool isActive;

    Material& operator=(Material&) = delete;  // Disallow copying
    Material(const Material&) = delete;

    Material(const double _rho) : rho(_rho)
    {
        transformMatrix = Eigen::MatrixXd::Zero(6, 6);
        matModulus = Eigen::MatrixXd::Zero(6, 6);
        matRotatedStressModulus = Eigen::MatrixXd::Zero(6, 6);
    }
    
    const double Rho() const {
        return rho;
    }
    virtual const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>&
    ComputeElasticModulus(const double& alpha, const double& beta) const = 0;
    virtual const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>&
    ComputeRotatedStressElasticModulus(const double& alpha, const double& beta) const = 0;
    
    virtual ~Material() = default;

    const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& 
    TransformationMatrix(const double& alpha, const double& beta) const
    {
        constexpr double pi = 3.14159265358979323846;
        constexpr double pi180 = pi / 180.;

/*        // alpha->fiber plane oriention; beta->fiber oriention.
        const double a = std::sin(alpha*pi180);
        const double b = std::cos(alpha*pi180);
        const double s = std::sin(beta*pi180);
        const double c = std::cos(beta*pi180);
	const double s2 = s * s;
	const double c2 = c * c;
	const double a2 = a * a;
	const double b2 = b * b;

        transformMatrix(0, 0) = b2 * c2;
        transformMatrix(0, 1) = a2 * c2;
        transformMatrix(0, 2) = s2;
        transformMatrix(0, 3) = 2*a*c*s;
        transformMatrix(0, 4) = 2 * b*c*s;
        transformMatrix(0, 5) = 2.0 * a*b*c2;

        transformMatrix(1, 0) = a2;
        transformMatrix(1, 1) = b2;
        transformMatrix(1, 5) = -2.0 * a * b;

        transformMatrix(2, 0) = b2*s2;
        transformMatrix(2, 1) = a2 * s2;
        transformMatrix(2, 2) = c2;
        transformMatrix(2, 3) = -2*a*c*s;
        transformMatrix(2, 4) = -2.0 * b*c*s;
        transformMatrix(2, 5) = 2.0 * a*b*s2;

        transformMatrix(3, 0) = a*b*s;
        transformMatrix(3, 1) = -a*b*s;
        transformMatrix(3, 3) = b*c;
        transformMatrix(3, 4) = -a*c;
        transformMatrix(3, 5) = -(b2-a2)*s;

        transformMatrix(4, 0) = -b2*c*s;
        transformMatrix(4, 1) = -a2*c*s;
        transformMatrix(4, 2) = c*s;
        transformMatrix(4, 3) = a*c2*s2;
        transformMatrix(4, 4) = b*c2*s2;
        transformMatrix(4, 5) = -2*a*b*c*s;

        transformMatrix(5, 0) = -a*b*c;
        transformMatrix(5, 1) = a*b*c;
        transformMatrix(5, 3) = b*s;
        transformMatrix(5, 4) = -a*s;
        transformMatrix(5, 5) = (b2-a2)*c;
*/

	
        // alpha->fiber plane oriention; beta->fiber oriention.
        const double sn_a = -std::sin(alpha*pi180);
        const double cn_a = std::cos(alpha*pi180);
        const double sn_b = -std::sin(beta*pi180);
        const double cn_b = std::cos(beta*pi180);

        transformMatrix(0, 0) = cn_a * cn_a * cn_b * cn_b;
        transformMatrix(0, 1) = sn_a * sn_a;
        transformMatrix(0, 2) = cn_a * cn_a * sn_b * sn_b;
        transformMatrix(0, 3) = -2.0 * cn_a * sn_a * sn_b;
        transformMatrix(0, 4) = -2.0 * cn_a * cn_a * sn_b * cn_b;
        transformMatrix(0, 5) = 2.0 * cn_a * sn_a * cn_b;

        transformMatrix(1, 0) = sn_a * sn_a * cn_b * cn_b;
        transformMatrix(1, 1) = cn_a * cn_a;
        transformMatrix(1, 2) = sn_a * sn_a * sn_b * sn_b;
        transformMatrix(1, 3) = 2.0 * cn_a * sn_a * sn_b;
        transformMatrix(1, 4) = -2.0 * sn_a * sn_a * sn_b * cn_b;
        transformMatrix(1, 5) = -2.0 * cn_a * sn_a * cn_b;

        transformMatrix(2, 0) = sn_b * sn_b;
        transformMatrix(2, 2) = cn_b * cn_b;
        transformMatrix(2, 4) = 2.0 * cn_b * sn_b;

        transformMatrix(3, 0) = -sn_a * sn_b * cn_b;
        transformMatrix(3, 2) = sn_a * sn_b * cn_b;
        transformMatrix(3, 3) = cn_a * cn_b;
        transformMatrix(3, 4) = -sn_a * cn_b * cn_b + sn_a * sn_b * sn_b;
        transformMatrix(3, 5) = cn_a * sn_b;
			   
        transformMatrix(4, 0) = cn_a * sn_b * cn_b;
        transformMatrix(4, 2) = -cn_a * sn_b * cn_b;
        transformMatrix(4, 3) = sn_a * cn_b;
        transformMatrix(4, 4) = -cn_a * sn_b * sn_b + cn_a* cn_b * cn_b;
        transformMatrix(4, 5) = sn_a * sn_b;
			   
        transformMatrix(5, 0) = -sn_a * cn_a * cn_b * cn_b;
        transformMatrix(5, 1) = cn_a * sn_a;
        transformMatrix(5, 2) = -sn_a * cn_a * sn_b * sn_b;
        transformMatrix(5, 3) = -cn_a * cn_a * sn_b + sn_a *sn_a * sn_b;
        transformMatrix(5, 4) = 2.0 * sn_a * sn_b * cn_a * cn_b;
        transformMatrix(5, 5) = cn_a * cn_a * cn_b - sn_a * sn_a * cn_b;

        return transformMatrix;
    }
};

class IsotropicMaterial : public Material
{
private:
    const Eigen::Matrix<double, 2, 1> matMechanicProp;

public:
    IsotropicMaterial& operator=(IsotropicMaterial&) = delete;  // Disallow copying
    IsotropicMaterial(const IsotropicMaterial&) = delete;
    
    IsotropicMaterial(const Eigen::Matrix<double, 2, 1> _matMechanicProp, const double _rho):
    Material(_rho), matMechanicProp(_matMechanicProp)
    {

        const double E = matMechanicProp(0);
        const double nu = matMechanicProp(1);
        const double G = E / (2 * (1 + nu));

        const double delta = E / (1. + nu) / (1 - 2.*nu);
        const double diag = (1. - nu) * delta;
        const double off_diag = nu * delta;
        
	matModulus(0, 0) = diag;
        matModulus(0, 1) = off_diag;
        matModulus(0, 2) = off_diag;

        matModulus(1, 0) = off_diag;
        matModulus(1, 1) = diag;
        matModulus(1, 2) = off_diag;

        matModulus(2, 0) = off_diag;
        matModulus(2, 1) = off_diag;
        matModulus(2, 2) = diag;

        matModulus(3, 3) = G;
        matModulus(4, 4) = G;
        matModulus(5, 5) = G;
    }
    
    virtual const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>&
    ComputeElasticModulus(const double& alpha, const double& beta) const override {
    	return matModulus;
    }
    virtual const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>&
    ComputeRotatedStressElasticModulus(const double& alpha, const double& beta) const override {
        const Eigen::Matrix<double,6,6, Eigen::RowMajor>& TM = TransformationMatrix(alpha, beta);
	matRotatedStressModulus = matModulus * TM.transpose();
	return matRotatedStressModulus;
    }
    
    virtual ~IsotropicMaterial() = default;
};

class OrthotropicMaterial : public Material
{
private:
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> matMechanicProp;
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> matLocalModulus;

public:
    OrthotropicMaterial& operator=(OrthotropicMaterial&) = delete;  // Disallow copying
    OrthotropicMaterial(const OrthotropicMaterial&) = delete;

    OrthotropicMaterial(const Eigen::Matrix<double,3,3, Eigen::RowMajor> _matMechanicProp, const double _rho):
    Material(_rho), matMechanicProp(_matMechanicProp)
    {

        const double e_xx = matMechanicProp(0,0);
        const double e_yy = matMechanicProp(0,1);
        const double e_zz = matMechanicProp(0,2);
        const double g_yz = matMechanicProp(1,0);
        const double g_xz = matMechanicProp(1,1);
        const double g_xy = matMechanicProp(1,2);
        const double nu_zy = matMechanicProp(2,0);
        const double nu_zx = matMechanicProp(2,1);
        const double nu_xy = matMechanicProp(2,2);
        // Calculate the other 3 poisson ratios.
        const double nu_yx = e_yy * nu_xy / e_xx;
        const double nu_xz = e_xx * nu_zx / e_zz;
        const double nu_yz = e_yy * nu_zy / e_zz;

        matLocalModulus = Eigen::MatrixXd::Zero(6, 6);

        const double delta = (1.0-nu_xy*nu_yx-nu_yz*nu_zy-nu_xz*nu_zx-2.0*nu_yx*nu_zy*nu_xz) / (e_xx*e_yy*e_zz);
        matLocalModulus(0, 0) = (1.0-nu_yz*nu_zy)/(e_yy*e_zz*delta);
        matLocalModulus(0, 1) = (nu_xy+nu_zy*nu_xz)/(e_xx*e_zz*delta);
        matLocalModulus(0, 2) = (nu_xz+nu_xy*nu_yz)/(e_xx*e_yy*delta);

        matLocalModulus(1, 0) = matLocalModulus(0, 1);
        matLocalModulus(1, 1) = (1-nu_xz*nu_zx)/(e_xx*e_zz*delta);
        matLocalModulus(1, 2) = (nu_yz+nu_yx*nu_xz)/(e_xx*e_yy*delta);

        matLocalModulus(2, 0) = matLocalModulus(0, 2);
        matLocalModulus(2, 1) = matLocalModulus(1, 2);
        matLocalModulus(2, 2) = (1-nu_xy*nu_yx)/(e_xx*e_yy*delta);

        matLocalModulus(3, 3) = g_yz;
        matLocalModulus(4, 4) = g_xz;
        matLocalModulus(5, 5) = g_xy;
    }
    
    virtual const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>&
    ComputeElasticModulus(const double& alpha, const double& beta) const override {
        const Eigen::Matrix<double,6,6, Eigen::RowMajor>& TM = TransformationMatrix(alpha, beta);
	matModulus = TM * matLocalModulus * TM.transpose();
    	return matModulus;
    }
    virtual const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>&
    ComputeRotatedStressElasticModulus(const double& alpha, const double& beta) const override {
        const Eigen::Matrix<double,6,6, Eigen::RowMajor>& TM = TransformationMatrix(alpha, beta);
	matRotatedStressModulus = matLocalModulus * TM.transpose();
	return matRotatedStressModulus;
    }
    
    virtual ~OrthotropicMaterial() = default;
};


class ElasticModulus : public dolfin::Expression
{
private:
    const std::vector<std::shared_ptr<const Material>> matsLibrary;
    const std::shared_ptr<const dolfin::MeshFunction<std::size_t>> material_id;
    const std::shared_ptr<const dolfin::MeshFunction<double>> plane_orientation;
    const std::shared_ptr<const dolfin::MeshFunction<double>> fiber_orientation;

public:
    ElasticModulus& operator=(ElasticModulus&) = delete;  // Disallow copying
    ElasticModulus(const ElasticModulus&) = delete;

    // Constructor.
    ElasticModulus(
    	const std::vector<std::shared_ptr<const Material>> _matsLibrary, 
    	const std::shared_ptr<const dolfin::MeshFunction<std::size_t>> _material_id,
	const std::shared_ptr<const dolfin::MeshFunction<double>> _plane_orientation,
	const std::shared_ptr<const dolfin::MeshFunction<double>> _fiber_orientation) : 
		dolfin::Expression(36),
		matsLibrary(_matsLibrary),
		material_id(_material_id),
		plane_orientation(_plane_orientation),
		fiber_orientation(_fiber_orientation)
    {}

    // Eval at every cell.
    void eval(Eigen::Ref<Eigen::VectorXd> values, const Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
    {
        size_t mat_id = (*material_id)[c.index];
        double alpha = (*plane_orientation)[c.index];
        double beta = (*fiber_orientation)[c.index];
        auto transformedStiffness = matsLibrary[mat_id]->ComputeElasticModulus(alpha, beta);

        // Assign elsticity matrix to local vertex values.
        values(0) = transformedStiffness(0, 0);
        values(1) = transformedStiffness(0, 1);
        values(2) = transformedStiffness(0, 2);
        values(3) = transformedStiffness(0, 3);
        values(4) = transformedStiffness(0, 4);
        values(5) = transformedStiffness(0, 5);
        values(6) = transformedStiffness(1, 0);
        values(7) = transformedStiffness(1, 1);
        values(8) = transformedStiffness(1, 2);
        values(9) = transformedStiffness(1, 3);
        values(10) = transformedStiffness(1, 4);
        values(11) = transformedStiffness(1, 5);
        values(12) = transformedStiffness(2, 0);
        values(13) = transformedStiffness(2, 1);
        values(14) = transformedStiffness(2, 2);
        values(15) = transformedStiffness(2, 3);
        values(16) = transformedStiffness(2, 4);
        values(17) = transformedStiffness(2, 5);
        values(18) = transformedStiffness(3, 0);
        values(19) = transformedStiffness(3, 1);
        values(20) = transformedStiffness(3, 2);
        values(21) = transformedStiffness(3, 3);
        values(22) = transformedStiffness(3, 4);
        values(23) = transformedStiffness(3, 5);
        values(24) = transformedStiffness(4, 0);
        values(25) = transformedStiffness(4, 1);
        values(26) = transformedStiffness(4, 2);
        values(27) = transformedStiffness(4, 3);
        values(28) = transformedStiffness(4, 4);
        values(29) = transformedStiffness(4, 5);
        values(30) = transformedStiffness(5, 0);
        values(31) = transformedStiffness(5, 1);
        values(32) = transformedStiffness(5, 2);
        values(33) = transformedStiffness(5, 3);
        values(34) = transformedStiffness(5, 4);
        values(35) = transformedStiffness(5, 5);
    }
}; // class

class RotatedStressElasticModulus : public dolfin::Expression
{
private:
    const std::vector<std::shared_ptr<const Material>> matsLibrary;
    const std::shared_ptr<const dolfin::MeshFunction<std::size_t>> material_id;
    const std::shared_ptr<const dolfin::MeshFunction<double>> plane_orientation;
    const std::shared_ptr<const dolfin::MeshFunction<double>> fiber_orientation;

public:
    RotatedStressElasticModulus& operator=(RotatedStressElasticModulus&) = delete;  // Disallow copying
    RotatedStressElasticModulus(const RotatedStressElasticModulus&) = delete;

    // Constructor.
    RotatedStressElasticModulus(
	const std::vector<std::shared_ptr<const Material>> _matsLibrary, 
	const std::shared_ptr<const dolfin::MeshFunction<std::size_t>> _material_id,
	const std::shared_ptr<const dolfin::MeshFunction<double>> _plane_orientation,
	const std::shared_ptr<const dolfin::MeshFunction<double>> _fiber_orientation) : 
		dolfin::Expression(36),
		matsLibrary(_matsLibrary),
		material_id(_material_id),
		plane_orientation(_plane_orientation),
		fiber_orientation(_fiber_orientation)
    {}

    // Eval at every cell.
    void eval(Eigen::Ref<Eigen::VectorXd> values, const Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
    {
        size_t mat_id = (*material_id)[c.index];
        double alpha = (*plane_orientation)[c.index];
        double beta = (*fiber_orientation)[c.index];
        auto transformedStiffness = matsLibrary[mat_id]->ComputeRotatedStressElasticModulus(alpha, beta);

        // Assign elsticity matrix to local vertex values.
        values(0) = transformedStiffness(0, 0);
        values(1) = transformedStiffness(0, 1);
        values(2) = transformedStiffness(0, 2);
        values(3) = transformedStiffness(0, 3);
        values(4) = transformedStiffness(0, 4);
        values(5) = transformedStiffness(0, 5);
        values(6) = transformedStiffness(1, 0);
        values(7) = transformedStiffness(1, 1);
        values(8) = transformedStiffness(1, 2);
        values(9) = transformedStiffness(1, 3);
        values(10) = transformedStiffness(1, 4);
        values(11) = transformedStiffness(1, 5);
        values(12) = transformedStiffness(2, 0);
        values(13) = transformedStiffness(2, 1);
        values(14) = transformedStiffness(2, 2);
        values(15) = transformedStiffness(2, 3);
        values(16) = transformedStiffness(2, 4);
        values(17) = transformedStiffness(2, 5);
        values(18) = transformedStiffness(3, 0);
        values(19) = transformedStiffness(3, 1);
        values(20) = transformedStiffness(3, 2);
        values(21) = transformedStiffness(3, 3);
        values(22) = transformedStiffness(3, 4);
        values(23) = transformedStiffness(3, 5);
        values(24) = transformedStiffness(4, 0);
        values(25) = transformedStiffness(4, 1);
        values(26) = transformedStiffness(4, 2);
        values(27) = transformedStiffness(4, 3);
        values(28) = transformedStiffness(4, 4);
        values(29) = transformedStiffness(4, 5);
        values(30) = transformedStiffness(5, 0);
        values(31) = transformedStiffness(5, 1);
        values(32) = transformedStiffness(5, 2);
        values(33) = transformedStiffness(5, 3);
        values(34) = transformedStiffness(5, 4);
        values(35) = transformedStiffness(5, 5);
    }
}; // class

class TransformationMatrix : public dolfin::Expression
{
private:
    const std::vector<std::shared_ptr<const Material>> matsLibrary;
    const std::shared_ptr<const dolfin::MeshFunction<std::size_t>> material_id;
    const std::shared_ptr<const dolfin::MeshFunction<double>> plane_orientation;
    const std::shared_ptr<const dolfin::MeshFunction<double>> fiber_orientation;

public:
    TransformationMatrix& operator=(TransformationMatrix&) = delete;  // Disallow copying
    TransformationMatrix(const TransformationMatrix&) = delete;

    // Constructor.
    TransformationMatrix(
	const std::vector<std::shared_ptr<const Material>> _matsLibrary,
	const std::shared_ptr<const dolfin::MeshFunction<std::size_t>> _material_id,
	const std::shared_ptr<const dolfin::MeshFunction<double>> _plane_orientation,
	const std::shared_ptr<const dolfin::MeshFunction<double>> _fiber_orientation) :
		dolfin::Expression(36),
		matsLibrary(_matsLibrary),
		material_id(_material_id),
		plane_orientation(_plane_orientation),
		fiber_orientation(_fiber_orientation)
    {}

    // Eval at every cell.
    void eval(Eigen::Ref<Eigen::VectorXd> values, const Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
    {
        size_t mat_id = (*material_id)[c.index];
        double alpha = (*plane_orientation)[c.index];
        double beta = (*fiber_orientation)[c.index];
        auto rotMatrix = matsLibrary[mat_id]->TransformationMatrix(alpha, beta);

        // Assign elsticity matrix to local vertex values.
        values(0) = rotMatrix(0, 0);
        values(1) = rotMatrix(0, 1);
        values(2) = rotMatrix(0, 2);
        values(3) = rotMatrix(0, 3);
        values(4) = rotMatrix(0, 4);
        values(5) = rotMatrix(0, 5);
        values(6) = rotMatrix(1, 0);
        values(7) = rotMatrix(1, 1);
        values(8) = rotMatrix(1, 2);
        values(9) = rotMatrix(1, 3);
        values(10) = rotMatrix(1, 4);
        values(11) = rotMatrix(1, 5);
        values(12) = rotMatrix(2, 0);
        values(13) = rotMatrix(2, 1);
        values(14) = rotMatrix(2, 2);
        values(15) = rotMatrix(2, 3);
        values(16) = rotMatrix(2, 4);
        values(17) = rotMatrix(2, 5);
        values(18) = rotMatrix(3, 0);
        values(19) = rotMatrix(3, 1);
        values(20) = rotMatrix(3, 2);
        values(21) = rotMatrix(3, 3);
        values(22) = rotMatrix(3, 4);
        values(23) = rotMatrix(3, 5);
        values(24) = rotMatrix(4, 0);
        values(25) = rotMatrix(4, 1);
        values(26) = rotMatrix(4, 2);
        values(27) = rotMatrix(4, 3);
        values(28) = rotMatrix(4, 4);
        values(29) = rotMatrix(4, 5);
        values(30) = rotMatrix(5, 0);
        values(31) = rotMatrix(5, 1);
        values(32) = rotMatrix(5, 2);
        values(33) = rotMatrix(5, 3);
        values(34) = rotMatrix(5, 4);
        values(35) = rotMatrix(5, 5);
    }
}; // class

class MaterialDensity : public dolfin::Expression
{
private:
    const std::vector<std::shared_ptr<const Material>> matsLibrary;
    const std::shared_ptr<const dolfin::MeshFunction<std::size_t>> material_id;

public:
    MaterialDensity& operator=(MaterialDensity&) = delete;  // Disallow copying
    MaterialDensity(const MaterialDensity&) = delete;

    // Constructor.
    MaterialDensity(
        const std::vector<std::shared_ptr<const Material>> _matsLibrary,
        const std::shared_ptr<const dolfin::MeshFunction<std::size_t>> _material_id) :
		dolfin::Expression(1),
		matsLibrary(_matsLibrary),
		material_id(_material_id)
    {}

    // Eval at every cell.
    void eval(Eigen::Ref<Eigen::VectorXd> values, const Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
    {
        size_t mat_id = (*material_id)[c.index];

        // Assign density.
        values(0) = matsLibrary[mat_id]->Rho();
    }
}; // class


} // namespace anba

//

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<anba::Material, std::shared_ptr<anba::Material>> Material
      (m, "Material", "Base material class");
    Material
    	.def("TransformationMatrix", &anba::Material::TransformationMatrix);

    pybind11::class_<anba::IsotropicMaterial, std::shared_ptr<anba::IsotropicMaterial>, anba::Material> 
      (m, "IsotropicMaterial", "Isotropic material class")
    .def(
        pybind11::init<const Eigen::Matrix<double, 2, 1>, const double>(),
        pybind11::arg("prop"),
        pybind11::arg("rho") = 0.
    );

    pybind11::class_<anba::OrthotropicMaterial, std::shared_ptr<anba::OrthotropicMaterial>, anba::Material> 
      (m, "OrthotropicMaterial", "Orthotropic material class")
    .def(
        pybind11::init<const Eigen::Matrix<double,3,3, Eigen::RowMajor>, const double>(),
        pybind11::arg("prop"),
        pybind11::arg("rho") = 0.
    );

    pybind11::class_<anba::ElasticModulus, std::shared_ptr<anba::ElasticModulus>, dolfin::Expression>
      (m, "ElasticModulus", "ElasticModulus expression")
    .def(pybind11::init<const std::vector<std::shared_ptr<const anba::Material>>, 
    	const std::shared_ptr<const dolfin::MeshFunction<std::size_t>>,
	const std::shared_ptr<const dolfin::MeshFunction<double>>,
	const std::shared_ptr<const dolfin::MeshFunction<double>>>()
    );

    pybind11::class_<anba::RotatedStressElasticModulus, std::shared_ptr<anba::RotatedStressElasticModulus>, dolfin::Expression>
      (m, "RotatedStressElasticModulus", "RotatedStressElasticModulus expression")
    .def(pybind11::init<const std::vector<std::shared_ptr<const anba::Material>>,
	const std::shared_ptr<const dolfin::MeshFunction<std::size_t>>,
	const std::shared_ptr<const dolfin::MeshFunction<double>>,
	const std::shared_ptr<const dolfin::MeshFunction<double>>>()
    );

    pybind11::class_<anba::TransformationMatrix, std::shared_ptr<anba::TransformationMatrix>, dolfin::Expression>
      (m, "TransformationMatrix", "TransformationMatrix expression")
    .def(pybind11::init<const std::vector<std::shared_ptr<const anba::Material>>,
	const std::shared_ptr<const dolfin::MeshFunction<std::size_t>>,
	const std::shared_ptr<const dolfin::MeshFunction<double>>,
	const std::shared_ptr<const dolfin::MeshFunction<double>>>()
    );

    pybind11::class_<anba::MaterialDensity, std::shared_ptr<anba::MaterialDensity>, dolfin::Expression>
      (m, "MaterialDensity", "MaterialDensity expression")
    .def(pybind11::init<const std::vector<std::shared_ptr<const anba::Material>>,
        const std::shared_ptr<const dolfin::MeshFunction<std::size_t>>>()
    );
}

