import os
import trimesh
import numpy as np
import tifffile
import time


def normalize_coordinates(vertices, min_vals, max_vals, resolution):
    # Normalizar as coordenadas dos vértices em relação à resolução da imagem
    scale = resolution / (max_vals - min_vals)
    normalized_vertices = (vertices - min_vals) * scale
    # Arredondar para inteiros e converter para tipo inteiro
    return np.clip(np.round(normalized_vertices), 0, resolution - 1).astype(int)


def generate_slices(mesh, num_slices, resolution=(256, 256)):
    """
    Gera fatias 2D a partir de um objeto de malha (mesh) ao longo do eixo Z e retorna um array 3D das fatias empilhadas.

    Args:
        mesh (trimesh.base.Trimesh): O objeto de malha.
        num_slices (int): O número de fatias a serem geradas.
        resolução (tuple): Resolução das fatias 2D.

    Returns:
        numpy.ndarray: O array 3D das fatias do mesh empilhadas.
    """
    # Encontrar os limites globais do mesh
    z_min, z_max = mesh.bounds[:, 2]

    # Calcular a altura de cada fatia
    z_levels = np.linspace(z_min, z_max, num_slices + 1)

    # Gerar as seções multiplanares
    sections = mesh.section_multiplane(plane_origin=(0, 0, 0),
                                       plane_normal=[0, 0, 1],
                                       heights=z_levels)

    # Inicializar listas para armazenar todos os vértices e as imagens 2D
    all_vertices = []
    all_images = []

    # Coletar todos os vértices das seções para normalização global
    for section in sections:
        if section is not None and len(section.vertices) > 0:
            vertices = section.vertices[:, :2]
            all_vertices.append(vertices)

    # Se não houver vértices, retornar None
    if not all_vertices:
        return None

    # Empilhar todos os vértices para calcular os valores globais
    stacked_vertices = np.vstack(all_vertices)

    # Calcular os valores mínimos e máximos globais
    global_min_vals = stacked_vertices.min(axis=0)
    global_max_vals = stacked_vertices.max(axis=0)

    # Gerar as imagens normalizadas
    for vertices in all_vertices:
        # Normalizar as coordenadas dos vértices
        normalized_vertices = normalize_coordinates(vertices, global_min_vals, global_max_vals, resolution[0])
        # Criar uma matriz de pixels
        image_array = np.zeros(resolution, dtype=np.uint8)
        # Ativar os pixels correspondentes às coordenadas normalizadas
        np.add.at(image_array, (normalized_vertices[:, 1], normalized_vertices[:, 0]), 255)
        all_images.append(image_array)

    # Empilhar as imagens 2D
    stacked_slices = np.stack(all_images, axis=0)

    return stacked_slices

def generate_slices_from_meshes(mesh1, mesh2=None, num_slices=256, resolution=(256, 256)):
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
    slices_mesh1 = generate_slices(mesh1, num_slices, resolution)

    if mesh2 is not None:
        # Gerar fatias para o segundo mesh
        slices_mesh2 = generate_slices(mesh2, num_slices, resolution)

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
    output_matriz_tif = os.path.join(path_amostra, 'output_matriz4.tif')
    output_fibras_tif = os.path.join(path_amostra, 'output_fibras4.tif')
    output_combined_tif = os.path.join(path_amostra, 'output_combined_matriz_fibras4.tif')

    print("carregando mesh1")
    mesh1 = trimesh.load(path_matriz_stl)

    print("carregando mesh2")
    mesh2 = trimesh.load(path_fibras_stl)
    # Marcar o tempo de início
    start_time = time.time()
    print("gerando fibras")

    fibras_slices = generate_slices_from_meshes(mesh2, num_slices=256, resolution=(256, 256))
    # Marcar o tempo de término
    end_time = time.time()
    # Calcular o tempo total de execução

    save_slices_as_tif(fibras_slices, output_fibras_tif)

    matriz_slices = generate_slices_from_meshes(mesh1, num_slices=256, resolution=(256, 256))
    save_slices_as_tif(matriz_slices, output_matriz_tif)


    combined_slices = generate_slices_from_meshes(mesh1, mesh2, num_slices=256, resolution=(256, 256))
    save_slices_as_tif(combined_slices, output_combined_tif)
    execution_time = end_time - start_time
    print('GEROU!')
    print(f"Tempo de execução: {execution_time:.2f} segundos")
