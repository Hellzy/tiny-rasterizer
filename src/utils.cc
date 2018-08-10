#include "utils.hh"

void cam_project_point(const cam_t& cam, point_t* p)
{
    double rot_mat[] =
    {
        cam.dir_x.x, cam.dir_x.y, cam.dir_x.z,
        cam.dir_y.x, cam.dir_y.y, cam.dir_y.z,
        cam.dir_z.x, cam.dir_z.y, cam.dir_z.z
    };

    *p -= cam.pos;

    double trans_mat[] = { p->x, p->y, p->z };
    double out_mat[3] = { 0 };

    mat_mult<double, 3>(rot_mat, trans_mat, out_mat);

    p->x = out_mat[0];
    p->y = out_mat[1];
    p->z = out_mat[2];
}
