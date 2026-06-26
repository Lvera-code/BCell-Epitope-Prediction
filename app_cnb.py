import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score

# DICCIONARIO DE PRODUCCIÓN (Numeración Estándar Chothia)

RANGOS_CDR = {
    'H': {  
        'CDR1': range(26, 33),
        'CDR2': range(52, 57),
        'CDR3': range(95, 103)
    },
    'L': {  
        'CDR1': range(24, 35),
        'CDR2': range(50, 57),
        'CDR3': range(89, 98)
    }
}

def verificar_si_es_cdr(id_cadena, numero_residuo):
    """Identifica si un residuo pertenece a los bucles hipervariables del anticuerpo."""
    if id_cadena not in RANGOS_CDR: 
        return False
    for name_cdr, rango_residuos in RANGOS_CDR[id_cadena].items():
        if numero_residuo in rango_residuos: 
            return True
    return False

# ARQUITECTURA DE LA RED NEURONAL DE GRAFOS (GCN 6D)

class RedEpitopos(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Entrada Hexadimensional: [X, Y, Z, SASA, Hidrofobicidad, Carga]
        self.conv1 = GCNConv(6, 16) 
        self.conv2 = GCNConv(16, 2)  # Clases: 0 (Estructura) o 1 (Sitio Activo/CDR)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# EXTRACCIÓN Y ENTRENAMIENTO 

if __name__ == "__main__":
    # Nota: Sustituir por la ruta del complejo antígeno-anticuerpo real en el laboratorio
    ruta_pdb = "complejo_anticuerpo.pdb" 
    
    if not os.path.exists(ruta_pdb):
        print(f"[-] Modo de Producción Listo. Archivo '{ruta_pdb}' no detectado en el directorio local.")
        print("[!] Inserta un complejo PDB con cadenas H y L para iniciar el pipeline real el miércoles.")
        exit()

    parser = PDBParser(QUIET=True)
    estructura = parser.get_structure("complejo", ruta_pdb)
    
    # Inyección de SASA (Área de Superficie Accesible al Solvente)
    sr = ShrakeRupley()
    sr.compute(estructura, level="A")
    
    # Escalas bioquímicas fenomenológicas
    escala_hidrofobicidad = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}
    escala_carga = {'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1, 'ASP': -1.0, 'GLU': -1.0}
    
    lista_nodos = []
    lista_etiquetas_y = []
    
    for modelo in estructura:
        for cadena in modelo:
            id_cad = cadena.get_id()
            
            # FILTRO DE PRODUCCIÓN: Evaluamos las cadenas del anticuerpo
            if id_cad in ['H', 'L']:
                for residuo in cadena:
                    nombre_res = residuo.get_resname()
                    num_res = residuo.get_id()[1] # Extracción del número correlativo
                    
                    # Extracción de descriptores químicos
                    hidro = escala_hidrofobicidad.get(nombre_res, 0.0)
                    carga = escala_carga.get(nombre_res, 0.0)
                    
                    for atomo in residuo:
                        if atomo.get_name() == 'CA': # Nodo centrado en Carbono Alfa
                            coor = atomo.get_coord()
                            sasa = getattr(atomo, 'sasa', 0.0)
                            
                            # Ensamblaje del vector hexadimensional
                            lista_nodos.append([coor[0], coor[1], coor[2], sasa, hidro, carga])
                            
                            # Mapeo inmuno-estructural
                            es_cdr = verificar_si_es_cdr(id_cad, num_res)
                            lista_etiquetas_y.append(1 if es_cdr else 0)

    # Conversión estructural a tensores de alta velocidad
    mi_tensor_x = torch.tensor(lista_nodos, dtype=torch.float)
    mi_tensor_y = torch.tensor(lista_etiquetas_y, dtype=torch.long)
    
    # Generación matricial de conectividad por proximidad espacial (Radio < 8.0 Å)
    posiciones = mi_tensor_x[:, :3]
    matriz_distancias = torch.cdist(posiciones, posiciones)
    edge_index = (matriz_distancias < 8.0).nonzero(as_tuple=False).t()
    
    # Cizalla: Penalización asimétrica por desequilibrio de clases
    num_negativos = (mi_tensor_y == 0).sum().float()
    num_positivos = (mi_tensor_y == 1).sum().float()
    
    if num_positivos == 0:
        print("[-] Error: El PDB no contiene residuos en las posiciones de CDR indexadas.")
        exit()
        
    peso_1 = num_negativos / num_positivos
    pesos_clases = torch.tensor([1.0, peso_1])
    
    print(f"[+] Complejo estructurado: {mi_tensor_x.shape}")
    print(f"[+] Castigo dinámico asignado a la clase minoritaria: {peso_1:.2f}")
    
    # Inicialización del motor de Deep Learning
    modelo = RedEpitopos()
    optimizador = optim.Adam(modelo.parameters(), lr=0.01)
    funcion_error = nn.NLLLoss(weight=pesos_clases)
    
    # Bucle de Entrenamiento (200 Épocas)

    modelo.train()
    for epoca in range(200):
        optimizador.zero_grad()
        predicciones = modelo(mi_tensor_x, edge_index)
        loss = funcion_error(predicciones, mi_tensor_y)
        loss.backward()
        optimizador.step()
        
    # Evaluación de Grado Clínico
    modelo.eval()
    with torch.no_grad():
        predicciones_finales = modelo(mi_tensor_x, edge_index)
        clases_predichas = predicciones_finales.argmax(dim=1)
        
    y_real = mi_tensor_y.cpu().numpy()
    y_pred = clases_predichas.cpu().numpy()
    
    precision = precision_score(y_real, y_pred, zero_division=0)
    recall = recall_score(y_real, y_pred, zero_division=0)
    
    print(f"Precisión (Falsos Positivos): {precision:.2f}")
    print(f"Exhaustividad (Recall Real):   {recall:.2f}")
