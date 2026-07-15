"""Config raiz de pytest: solo garantiza que la raiz del repo este en sys.path
para que los tests puedan importar 'pipeline' y 'src.*' sin instalar el
paquete. No requiere ningun binario externo (BepiPred/EpiDope/BLAST+/
NetMHCIIpan): los tests en tests/ cubren unicamente la logica pura de cada
fase (parseo, fusion, calculo de umbrales), nunca invocan un subprocess real.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
