import os, os.path as osp
import sys

def obj2urdf(obj_filename, obj_filename_full, obj_name, save_dir=None, cwd=False, scaling=None, dry_run=False):
    if scaling is None:
        scale = [1.0, 1.0, 1.0]
    else:
        assert isinstance(scaling, list), 'scaling must be a 3-element list'
        assert len(scaling) == 3, 'scaling must be a 3-element list'
        scale = scaling

    obj_fname = obj_filename.split('/')[-1] + '.urdf'

    urdf_str = '''<?xml version="1.0" ?>
<robot name="%s">
    <link concave="yes" name="base_link">
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value="0.0"/>
            <inertia ixx="1e-08" ixy="0" ixz="0" iyy="1e-08" iyz="0" izz="1e-08"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="%s" scale="%f %f %f"/>
            </geometry>
        </visual>

        <collision concave="yes">
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="%s" scale="%f %f %f"/>
            </geometry>
        </collision>
    </link>
</robot>
    ''' % (obj_name, obj_filename, scale[0], scale[1], scale[2], obj_filename, scale[0], scale[1], scale[2])

    obj_fname = obj_filename_full.split('/')[-1] + '.urdf'
    if save_dir is None:
        if not cwd:
            save_dir = '/'.join(obj_filename.split('/')[:-1])
        else:
            save_dir = os.getcwd()

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    urdf_fname = osp.join(save_dir, obj_fname)
    print(f'URDF filename: {urdf_fname}')
    if dry_run:
        pass
    else:
        with open(urdf_fname, 'w') as f:
            f.write(urdf_str)
    return urdf_str, urdf_fname


# bookshelf_obj_fnames = [fn for fn in os.listdir('.') if (fn.endswith('.obj') and ('dec' not in fn))]
obj_dirname = sys.argv[1]

dry_run = False
if len(sys.argv) > 2:
    dry_run_flag = sys.argv[2]
    dry_run = dry_run_flag == '-d' or dry_run_flag == '--dry_run' or dry_run_flag == '--dry-run'

obj_dirname_full = osp.join(os.getcwd(), obj_dirname)
obj_fnames = [fn for fn in os.listdir(obj_dirname_full) if (fn.endswith('.obj') and ('dec' not in fn))]
obj_fnames_full = [osp.join(obj_dirname_full, fn) for fn in obj_fnames]

for i, fn in enumerate(obj_fnames):
    # _, urdf_fname = obj2urdf(fn, fn.split('.obj')[0], save_dir='urdfs') #, dry_run=True)
    _, urdf_fname = obj2urdf(
        obj_filename=fn, 
        obj_filename_full=obj_fnames_full[i], 
        obj_name=fn.split('.obj')[0], 
        save_dir=osp.join(obj_dirname, 'urdfs'), 
        dry_run=dry_run)
