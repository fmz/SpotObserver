#include "four-pcs-gpu-diagonal.h"

#include "four-pcs.h"

namespace SOb {

namespace {
inline Eigen::Vector3d toEigen(float3 v) { return {v.x, v.y, v.z}; }
}  // namespace

HostDiagonalPairing diagonalPairingAndRatiosHost(const float3 base_points[4]) {
    static const int orderings[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};

    for (auto& ordering : orderings) {
        int a = ordering[0], b = ordering[1], c = ordering[2], d = ordering[3];
        Eigen::Vector3d pa = toEigen(base_points[a]), pb = toEigen(base_points[b]);
        Eigen::Vector3d pc = toEigen(base_points[c]), pd = toEigen(base_points[d]);

        Eigen::Matrix<double, 3, 2> coeff;
        coeff.col(0) = pb - pa;
        coeff.col(1) = -(pd - pc);
        Eigen::Vector3d rhs = pc - pa;
        Eigen::Vector2d solution = coeff.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
        double ratio_a = solution(0), ratio_b = solution(1);

        if (ratio_a >= -0.05 && ratio_a <= 1.05 && ratio_b >= -0.05 && ratio_b <= 1.05) {
            Eigen::Vector3d crossing = pa + ratio_a * (pb - pa);
            Eigen::Vector3d crossing_check = pc + ratio_b * (pd - pc);
            if ((crossing - crossing_check).norm() < 0.02) {
                HostDiagonalPairing result;
                result.found = true;
                result.order[0] = a; result.order[1] = b; result.order[2] = c; result.order[3] = d;
                result.ratio_a = ratio_a;
                result.ratio_b = ratio_b;
                result.diag_a = (pb - pa).norm();
                result.diag_b = (pd - pc).norm();
                return result;
            }
        }
    }
    return HostDiagonalPairing{};
}

}  // namespace SOb
