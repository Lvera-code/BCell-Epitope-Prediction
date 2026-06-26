import os
import torch
from Bio.PDB import PDBParser

# =====================================================================
# DICCIONARIO TÁCTICO: Definición de CDRs (Numeración estándar Chothia)
# =====================================================================
RANGOS_CDR = {
    'H': {  # Cadena Pesada (Heavy)
        'CDR1': range(26, 33),
        'CDR2': range(52, 57),
        'CDR3': range(95, 103)
    },
    'L': {  # Cadena Ligera (Light)
        'CDR1': range(24, 35),
        'CDR2': range(50, 57),
        'CDR3': range(89, 98)
    }
}

def verificar_si_es_cdr(id_cadena, numero_residuo):
    """
    Determina si un residuo específico forma parte del sitio de unión (CDR).
    """
    if id_cadena not in RANGOS_CDR:
        return False # No es una cadena del anticuerpo (puede ser el antígeno)
        
    for name_cdr, rango_residuos in RANGOS_CDR[id_cadena].items():
        if numero_residuo in rango_residuos:
            return True
    return False

# =====================================================================
# PARSEO E INYECCIÓN DE ETIQUETAS DE INMUNOLOGÍA
# =====================================================================
parser = PDBParser(QUIET=True)
# Simulamos la carga de un complejo antígeno-anticuerpo (debes tener el PDB en la carpeta)
ruta_ejemplo = "1CRN.pdb" 

if os.path.exists(ruta_ejemplo):
    estructura = parser.get_structure("complejo", ruta_ejemplo)
    
    lista_etiquetas_y = []
    
    for modelo in estructura:
        for cadena in modelo:
            id_cad = cadena.get_id() # Puede ser 'H', 'L' o la del antígeno 'A'
            
            for residuo in cadena:
                # El id del residuo en BioPython devuelve una tupla: (' ', numero_residuo, ' ')
                num_res = residuo.get_id()[1] 
                
                for atomo in residuo:
                    if atomo.get_name() == 'CA':
                        # Determinamos si este Carbono Alfa está en una CDR
                        es_zona_activa = verificar_si_es_cdr(id_cad, num_res)
                        
                        # Asignamos la etiqueta objetivo para la IA: 1 si es CDR, 0 si es estructura
                        etiqueta = 1 if es_zona_activa else 0
                        lista_etiquetas_y.append(etiqueta)
                        
    tensor_y = torch.tensor(lista_etiquetas_y, dtype=torch.long)
    print(f"Tensor Y de Inmunogenicidad creado con éxito. Tamaño: {tensor_y.shape}")
    print(f"Átomos detectados en zona CDR (Clase 1): {int((tensor_y == 1).sum())}")
else:
    print(f"Falta el archivo {ruta_ejemplo} para realizar el mapeo de inmunidad.")
    