import sys

import numpy as np
import math
import stl
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import stltovoxel
from PIL import Image
from skimage import io


# find the max dimensions, so we can know the bounding box, getting the height,
# width, length (because these are the step size)...
def find_mins_maxs(obj):
    minx = obj.x.min()
    maxx = obj.x.max()
    miny = obj.y.min()
    maxy = obj.y.max()
    minz = obj.z.min()
    maxz = obj.z.max()
    return minx, maxx, miny, maxy, minz, maxz


def translate(_solid, step, padding, multiplier, axis):
    if 'x' == axis:
        items = 0, 3, 6
    elif 'y' == axis:
        items = 1, 4, 7
    elif 'z' == axis:
        items = 2, 5, 8
    else:
        raise RuntimeError('Unknown axis %r, expected x, y or z' % axis)

    # _solid.points.shape == [:, ((x, y, z), (x, y, z), (x, y, z))]
    _solid.points[:, items] += (step * multiplier) + (padding * multiplier)


def copy_obj(obj, dims, num_rows, num_cols, num_layers):
    w, l, h = dims
    copies = []
    for layer in range(num_layers):
        for row in range(num_rows):
            for col in range(num_cols):
                # skip the position where original being copied is
                if row == 0 and col == 0 and layer == 0:
                    continue
                _copy = mesh.Mesh(obj.data.copy())
                # pad the space between objects by 10% of the dimension being
                # translated
                if col != 0:
                    translate(_copy, w, w / 10., col, 'x')
                if row != 0:
                    translate(_copy, l, l / 10., row, 'y')
                if layer != 0:
                    translate(_copy, h, h / 10., layer, 'z')
                copies.append(_copy)
    return copies

def plot_combined():
    # Load an existing STL file
    your_mesh = mesh.Mesh.from_file('D:/mestrado/Material TCC Douglas/3D/06_06/Matriz.stl')


    volume, cog, inertia = your_mesh.get_mass_properties()
    print("Volume                                  = {0}".format(volume))
    print("Position of the center of gravity (COG) = {0}".format(cog))
    print("Inertia matrix at expressed at the COG  = {0}".format(inertia[0,:]))
    print("                                          {0}".format(inertia[1,:]))
    print("                                          {0}".format(inertia[2,:]))



    # Using an existing stl file:
    main_body = mesh.Mesh.from_file('D:/mestrado/Material TCC Douglas/3D/06_06/Matriz.stl')



    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(main_body)
    w1 = maxx - minx
    l1 = maxy - miny
    h1 = maxz - minz
    copies = copy_obj(main_body, (w1, l1, h1), 2, 2, 1)

    # I wanted to add another related STL to the final STL
    twist_lock = mesh.Mesh.from_file('D:/mestrado/Material TCC Douglas/3D/06_06/Fibras.stl')
    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(twist_lock)
    w2 = maxx - minx
    l2 = maxy - miny
    h2 = maxz - minz
    # translate(twist_lock, w1, w1 / 1., 3, 'x')
    # copies2 = copy_obj(twist_lock, (w2, l2, h2), 2, 2, 1)
    combined = mesh.Mesh(np.concatenate([main_body.data, twist_lock.data]))

    volume, cog, inertia = combined.get_mass_properties()
    print("Volume                                  = {0}".format(volume))
    print("Position of the center of gravity (COG) = {0}".format(cog))
    print("Inertia matrix at expressed at the COG  = {0}".format(inertia[0,:]))
    print("                                          {0}".format(inertia[1,:]))
    print("                                          {0}".format(inertia[2,:]))



    # Create a new plot
    figure = pyplot.figure()
    axes = figure.add_subplot(projection='3d')

    # Render the cube
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(combined.vectors))

    # Auto scale to the mesh size
    scale = combined.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()


def stl_to_numpy(filename):
    meshes = []
    mesh_obj = mesh.Mesh.from_file(filename)
    org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
    meshes.append(org_mesh)

    voxels, scale, shift = stltovoxel.convert.convert_meshes(meshes)
    voxels = voxels.astype(bool)
    out = []
    for z in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for x in range(voxels.shape[2]):
                if voxels[z][y][x]:
                    point = (np.array([x, y, z]) / scale) + shift
                    out.append(point)
    return np.asarray(out)




def stl_to_tiff(input_filename, output_filename):
    # Load the STL file
    your_mesh = mesh.Mesh.from_file(input_filename)

    # Convert the mesh to a numpy array
    imarray = np.array(your_mesh)

    # Create a PIL image from the numpy array
    im = Image.fromarray(imarray)

    # Save the image as a TIFF
    im.save(output_filename)

if __name__ == '__main__':
    # plot_combined()
    input_filename = 'D:/mestrado/Material TCC Douglas/3D/06_06/Matriz.stl'
    output_filename = 'D:/mestrado/Material TCC Douglas/3D/06_06/Matriz.tiff'
    # numpyArrayStl = stl_to_numpy('D:/mestrado/Material TCC Douglas/3D/06_06/Matriz.stl')
    # print("Shape numpyArrayStl     = {0}".format(numpyArrayStl.shape))
    # stl_to_tiff(input_filename, output_filename)

    your_mesh = mesh.Mesh.from_file(input_filename)

    # Convert the mesh to a numpy array
    imarray = np.array(your_mesh)
    print("Shape numpyArrayStl     = {0}".format(imarray.shape))

    image = io.imread('D:/mestrado/sandstone_data_for_ML/data_for_3D_Unet/train_images_256_256_256.tif')
    print("Shape imageTrain     = {0}".format(image.shape))
    # img_patches = patchify(image, (64, 64, 64), step=64)  # Step=64 for 64 patches means no overlap

