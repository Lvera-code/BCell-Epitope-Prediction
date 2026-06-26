import os
import glob
import torch
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from torch_geometric.data import Data

# =====================================================================
# CONFIGURACIÓN DE DICCIONARIOS BIOQUÍMICOS
# =====================================================================
escala_hidrofobicidad = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}
escala_carga = {'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1, 'ASP': -1.0, 'GLU': -1.0}

def procesar_un_pdb(ruta_pdb):
    """
    Parsea un único archivo PDB y lo transforma en un Grafo de PyTorch Geometric.
    Contiene un sistema de control de calidad estricto.
    """
    parser = PDBParser(QUIET=True)
    nombre_id = os.path.basename(ruta_pdb).split('.')[0]
    
    estructura = parser.get_structure(nombre_id, ruta_pdb)
    
    # Cálculo obligatorio de SASA
    sr = ShrakeRupley()
    sr.compute(estructura, level="A")
    
    lista_nodos = []
    
    for modelo in estructura:
        for cadena in modelo:
            # Filtro defensivo: Solo procesamos la cadena principal A para esta prueba
            if cadena.get_id() == 'A':
                for residuo in cadena:
                    nombre_res = residuo.get_resname()
                    
                    # Ignorar solventes y moléculas de agua intrusas
                    if nombre_res in ['HOH', 'WAT']:
                        continue
                        
                    hidro = escala_hidrofobicidad.get(nombre_res, 0.0)
                    carga = escala_carga.get(nombre_res, 0.0)
                    
                    for atomo in residuo:
                        if atomo.get_name() == 'CA': # Filtrado por Carbono Alfa
                            coor = atomo.get_coord()
                            sasa = getattr(atomo, 'sasa', 0.0)
                            
                            # Vector hexadimensional
                            lista_nodos.append([coor[0], coor[1], coor[2], sasa, hidro, carga])
    
    if len(lista_nodos) == 0:
        raise ValueError(f"La cadena A de {nombre_id} no contiene Carbonos Alfa válidos.")
        
    tensor_x = torch.tensor(lista_nodos, dtype=torch.float)
    
    # CONSTRUCCIÓN MATRIZ DE CONTACTO (EDGE_INDEX) AUTOMÁTICA
    # Distancia de corte: 8.0 Angstroms
    posiciones = tensor_x[:, :3]
    matriz_distancias = torch.cdist(posiciones, posiciones)
    indices_conexión = (matriz_distancias < 8.0).nonzero(as_tuple=False).t()
    
    # Creamos el objeto Data oficial de PyTorch Geometric
    grafo_proteina = Data(x=tensor_x, edge_index=indices_conexión)
    return grafo_proteina

# =====================================================================
# BUCLE AUTOMATIZADO CON TOLERANCIA A FALLOS
# =====================================================================
if __name__ == "__main__":
    # Buscamos todos los PDBs en el directorio actual
    archivos_pdb = glob.glob("*.pdb") + glob.glob("*.ent")
    print(f"Se han detectado {len(archivos_pdb)} archivos para procesar.")
    
    dataset_final = []
    
    for ruta in archivos_pdb:
        print(f"Procesando: {ruta}...", end="")
        try:
            # Bloque de seguridad: Si una proteína falla, el bucle continúa
            grafo = procesar_un_pdb(ruta)
            dataset_final.append(grafo)
            print(" [OK]")
        except Exception as e:
            print(f" [FALLO] -> Motivo: {e}")
            
    print(f"\nProceso finalizado. Grafos generados con éxito: {len(dataset_final)}/{len(archivos_pdb)}")
    
    # Guardamos la factoría de datos en un archivo binario de alta velocidad
    if dataset_final:
        torch.save(dataset_final, "dataset_estructural_output.pt")
        print("Dataset serializado y guardado como 'dataset_estructural_output.pt'")
