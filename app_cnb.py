import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score

# 1. DICCIONARIOS Y UTILIDADES BIOQUÍMICAS

# Numeración Estándar Chothia
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

# 2. ARQUITECTURA DE LA RED NEURONAL (GCN)

class RedEpitopos(torch.nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        # Entrada Hexadimensional: [X, Y, Z, SASA, Hidrofobicidad, Carga]
        self.conv1 = GCNConv(6, hidden_dim) 
        self.conv2 = GCNConv(hidden_dim, 2)  # Clases: 0 (Estructura) o 1 (Sitio Activo/CDR)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 3. INFRAESTRUCTURA DE PRODUCCIÓN

def setup_logging(output_dir):
    """Configura el sistema de logs dobles: consola y archivo para SLURM."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train_production.log')
    
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Historial de ejecución redirigido a: {log_file}")

def parse_args():
    """Define los puntos de entrada dinámicos por terminal."""
    parser = argparse.ArgumentParser(description="CNB Pipeline: GCN para Predicción de Interfaces de Unión.")
    
    parser.add_argument('--pdb_path', type=str, default='complejo_anticuerpo.pdb', help='Ruta al archivo PDB a procesar.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directorio para logs y pesos del modelo.')
    parser.add_argument('--epochs', type=int, default=200, help='Número de épocas de entrenamiento.')
    parser.add_argument('--lr', type=float, default=0.01, help='Tasa de aprendizaje (Learning Rate).')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Dimensión de las capas ocultas de la GCN.')
    
    return parser.parse_args()

# 4. MOTOR DE EXTRACCIÓN Y ENTRENAMIENTO

def train_model(args):
    ruta_pdb = args.pdb_path
    
    if not os.path.exists(ruta_pdb):
        logging.error(f"Modo de Producción Listo. Archivo '{ruta_pdb}' no detectado.")
        logging.error("Inserta un complejo PDB válido para iniciar el pipeline.")
        sys.exit(1)

    logging.info(f"Extrayendo topología y descriptores de: {ruta_pdb}")
    parser = PDBParser(QUIET=True)
    estructura = parser.get_structure("complejo", ruta_pdb)
    
    # Inyección de SASA
    sr = ShrakeRupley()
    sr.compute(estructura, level="A")
    
    escala_hidrofobicidad = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}
    escala_carga = {'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.1, 'ASP': -1.0, 'GLU': -1.0}
    
    lista_nodos = []
    lista_etiquetas_y = []
    
    for modelo in estructura:
        for cadena in modelo:
            id_cad = cadena.get_id()
            if id_cad in ['H', 'L']:
                for residuo in cadena:
                    nombre_res = residuo.get_resname()
                    num_res = residuo.get_id()[1]
                    
                    hidro = escala_hidrofobicidad.get(nombre_res, 0.0)
                    carga = escala_carga.get(nombre_res, 0.0)
                    
                    for atomo in residuo:
                        if atomo.get_name() == 'CA': 
                            coor = atomo.get_coord()
                            sasa = getattr(atomo, 'sasa', 0.0)
                            
                            lista_nodos.append([coor[0], coor[1], coor[2], sasa, hidro, carga])
                            es_cdr = verificar_si_es_cdr(id_cad, num_res)
                            lista_etiquetas_y.append(1 if es_cdr else 0)

    # Conversión a tensores
    mi_tensor_x = torch.tensor(lista_nodos, dtype=torch.float)
    mi_tensor_y = torch.tensor(lista_etiquetas_y, dtype=torch.long)
    
    # Matriz de conectividad (Radio < 8.0 Å)
    posiciones = mi_tensor_x[:, :3]
    matriz_distancias = torch.cdist(posiciones, posiciones)
    edge_index = (matriz_distancias < 8.0).nonzero(as_tuple=False).t()
    
    # Desequilibrio de clases
    num_negativos = (mi_tensor_y == 0).sum().float()
    num_positivos = (mi_tensor_y == 1).sum().float()
    
    if num_positivos == 0:
        logging.error("El PDB no contiene residuos en las posiciones de CDR indexadas.")
        sys.exit(1)
        
    peso_1 = num_negativos / num_positivos
    pesos_clases = torch.tensor([1.0, peso_1])
    
    logging.info(f"Complejo estructurado: {mi_tensor_x.shape[0]} nodos generados.")
    logging.info(f"Castigo dinámico asignado a clase minoritaria (CDR): {peso_1:.2f}")
    
    # Inicialización del modelo
    modelo = RedEpitopos(hidden_dim=args.hidden_dim)
    optimizador = optim.Adam(modelo.parameters(), lr=args.lr)
    funcion_error = nn.NLLLoss(weight=pesos_clases)
    
    logging.info(f"Iniciando entrenamiento: {args.epochs} épocas | LR: {args.lr} | Dimensión oculta: {args.hidden_dim}")
    
    modelo.train()
    for epoca in range(args.epochs):
        optimizador.zero_grad()
        predicciones = modelo(mi_tensor_x, edge_index)
        loss = funcion_error(predicciones, mi_tensor_y)
        loss.backward()
        optimizador.step()
        
        if (epoca + 1) % 50 == 0 or epoca == 0:
            logging.info(f"Época [{epoca+1:03d}/{args.epochs}] | Pérdida: {loss.item():.4f}")
            
    # Evaluación
    modelo.eval()
    with torch.no_grad():
        predicciones_finales = modelo(mi_tensor_x, edge_index)
        clases_predichas = predicciones_finales.argmax(dim=1)
        
    y_real = mi_tensor_y.cpu().numpy()
    y_pred = clases_predichas.cpu().numpy()
    
    precision = precision_score(y_real, y_pred, zero_division=0)
    recall = recall_score(y_real, y_pred, zero_division=0)
    
    logging.info("=== RESULTADOS DEL MODELO ===")
    logging.info(f"Precisión (Control Falsos Positivos): {precision:.4f}")
    logging.info(f"Exhaustividad (Recall Real):          {recall:.4f}")
    
    # Guardado de pesos
    checkpoint_path = os.path.join(args.output_dir, 'gcn_cnb_model.pt')
    torch.save(modelo.state_dict(), checkpoint_path)
    logging.info(f"Pesos del modelo guardados en: {checkpoint_path}")

# 5. PUNTO DE ENTRADA (MAIN)

def main():
    args = parse_args()
    setup_logging(args.output_dir)
    
    try:
        train_model(args)
    except Exception as e:
        logging.error(f"Fallo crítico en la ejecución del pipeline: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
