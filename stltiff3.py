import os
import io
import trimesh
import numpy as np
import tifffile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def rasterize_mesh_to_voxel(mesh, pitch):
    # Rasteriza a malha em uma grade volumétrica
    voxelized = mesh.voxelized(pitch)
    return voxelized.matrix

def rasterize_slice(vertices, resolution=(256, 256)):
    # Criar uma imagem em branco
    image = np.zeros(resolution, dtype=np.uint8)

    vertices = (((vertices + 1) / 2) * (np.array(resolution) - 1)).astype(int)

    # Converter vértices para índices de pixel
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Garantir que os índices estejam dentro dos limites da imagem
    x = np.clip(x, 0, resolution[0] - 1)
    y = np.clip(y, 0, resolution[1] - 1)

    # Preencher a imagem nos locais dos vértices
    image[y, x] = 255

    return image

def process_slice(mesh, z, resolution, bounds):
    """
    Processa uma única fatia do mesh.

    Args:
        mesh (trimesh.base.Trimesh): O objeto de malha.
        z (float): Coordenada Z para a fatia.
        resolution (tuple): Resolução da imagem de saída.

    Returns:
        numpy.ndarray: Imagem rasterizada da fatia.
    """

    x_min, x_max, y_min, y_max = bounds

    # Cortar a fatia
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None or len(section.vertices) < 2:
        # Se a seção não existir ou tiver menos de 2 vértices, adicionar uma fatia em branco
        return np.zeros(resolution, dtype=np.uint8)


    # Extrair os vértices da fatia e normalizar
    slice_2d, _ = section.to_planar()
    vertices_xy = slice_2d.vertices

    # Calcular as dimensões da fatia
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Centralizar os vértices da fatia
    vertices_centered = vertices_xy - [x_min, y_min]

    # Escalar os vértices para a resolução da imagem, mantendo as proporções
    scaling_factor = max(resolution[0] / x_range, resolution[1] / y_range)
    vertices_scaled = (vertices_centered * scaling_factor).astype(int)

    return rasterize_slice(vertices_scaled, resolution)



def generate_slices(mesh, num_slices, resolution=(256, 256)):
    """
    Gera fatias 2D a partir de um objeto de malha (mesh) ao longo do eixo Z e retorna um array 3D das fatias.

    Args:
        mesh (trimesh.base.Trimesh): O objeto de malha.
        num_slices (int): O número de fatias a serem geradas.
        resolução (tuple): Resolução das fatias 2D.

    Returns:
        numpy.ndarray: O array 3D das fatias do mesh.
    """
    # Encontrar os limites globais do mesh
    z_min = mesh.bounds[0][2]
    z_max = mesh.bounds[1][2]
    x_min = mesh.bounds[0][0]
    x_max = mesh.bounds[1][0]
    y_min = mesh.bounds[0][1]
    y_max = mesh.bounds[1][1]
    bounds = (x_min, x_max, y_min, y_max)

    # Calcular a altura de cada fatia
    slice_height = (z_max - z_min) / num_slices
    # Utilizar ThreadPoolExecutor para processar fatias em paralelo
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_slice, mesh, z_min + i * slice_height, resolution, bounds) for i in
                   range(num_slices)]
        slices = [f.result() for f in tqdm(futures, desc='Gerando fatias')]

    # Empilhar as fatias em uma matriz tridimensional
    stacked_slices = np.stack(slices, axis=0)
    return stacked_slices

def generate_slices_from_meshes(mesh1, mesh2=None, num_slices=256):
    """
    Gera fatias 2D a partir de um ou dois objetos de malha (meshes) e retorna o array 3D combinado das fatias.

    Args:
        mesh1 (trimesh.base.Trimesh): O primeiro objeto de malha.
        mesh2 (trimesh.base.Trimesh, optional): O segundo objeto de malha. Padrão é None.
        num_slices (int, optional): O número de fatias a serem geradas para cada mesh. Padrão é 256.

    Returns:
        numpy.ndarray: O array 3D combinado das fatias dos meshes.
    """
    # Gerar fatias para o primeiro mesh
    slices_mesh1 = generate_slices(mesh1, num_slices)

    if mesh2 is not None:
        # Gerar fatias para o segundo mesh
        slices_mesh2 = generate_slices(mesh2, num_slices)

        # Combinar as fatias dos dois meshes
        stacked_slices_combined = np.maximum(slices_mesh1, slices_mesh2)
    else:
        # Se não houver segundo mesh, apenas usar as fatias do primeiro mesh
        stacked_slices_combined = slices_mesh1

    return stacked_slices_combined

def save_voxels_as_tif(volume, filename):
    # Salvar a grade volumétrica como um arquivo TIFF
    tifffile.imwrite(filename, volume.astype(np.uint8))

def save_slices_as_tif(stacked_slices, filename):
    # Salvar a stack de fatias como um arquivo TIFF
    tifffile.imwrite(filename, stacked_slices.astype(np.uint8))

if __name__ == '__main__':
    # Carregar o arquivo STL
    path_3D = 'D:/mestrado/Material TCC Douglas/3D'
    amostra = '03_09'
    path_amostra = os.path.join(path_3D, amostra)
    path_matriz_stl = os.path.join(path_amostra, 'Matriz.stl')
    path_fibras_stl = os.path.join(path_amostra, 'Fibras.stl')
    output_matriz_tif = os.path.join(path_amostra, 'output_matriz_bounds.tif')
    output_fibras_tif = os.path.join(path_amostra, 'output_fibras_bounds.tif')
    output_combined_tif = os.path.join(path_amostra, 'output_combined_matriz_fibras_bounds.tif')

    # Definir o pitch para voxelização (resolução)
    pitch = 1.0

    # print("carregando mesh1")
    # mesh1 = trimesh.load(path_matriz_stl)

    print("carregando mesh2")
    mesh2 = trimesh.load(path_fibras_stl)

    print("gerando fibras")
    # fibras_voxel = rasterize_mesh_to_voxel(mesh2, pitch)
    # save_voxels_as_tif(fibras_voxel, output_fibras_tif)
    fibras_slices = generate_slices(mesh2, num_slices=256, resolution=(256, 256))
    save_slices_as_tif(fibras_slices, output_fibras_tif)

    # Gerar fatias a partir de um único mesh
    # print("gerando matriz")
    # matriz_slices = generate_slices(mesh1, num_slices=384, resolution=(768, 768))
    # save_slices_as_tif(matriz_slices, output_matriz_tif)

    print('GEROU!')
